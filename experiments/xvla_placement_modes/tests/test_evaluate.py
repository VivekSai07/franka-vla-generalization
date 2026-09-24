"""CPU-only evaluator tests: rosters, gates, offline probes, and verify_final tamper checks."""
import importlib
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'xvla_broad_workspace_long'/'tests'))
from test_contract import frame


def _module():
    return importlib.import_module('experiments.xvla_placement_modes.evaluate')


FINAL = _module().FINAL_UPDATE


def _fake_plan():
    """Full broad-workspace plan; identity stays deterministic across imports."""
    from experiments.broad_workspace_data import plan as p
    return p.make_plan()


class RosterTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_broad_roster_is_36_layouts_by_2_seeds_layout_major(self):
        e = self.e
        trials = e.evaluation_plan_broad()
        self.assertEqual(len(trials), 128)
        self.assertEqual(len(e.MODE_LAYOUTS), 16)
        self.assertEqual(len(e.MODE_SEEDS), 8)
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 129)))
        expected = [(l, s) for l in e.MODE_LAYOUTS for s in e.MODE_SEEDS]
        self.assertEqual([(t['layout_id'], t['seed']) for t in trials], expected)

    def test_roster_is_balanced_and_groups_are_labelled(self):
        e = self.e
        trials = e.evaluation_plan_broad()
        self.assertEqual(len(e.FAILING_LAYOUTS), len(e.CONTROL_LAYOUTS))
        self.assertEqual(set(e.FAILING_LAYOUTS) & set(e.CONTROL_LAYOUTS), set())
        groups = {t['group'] for t in trials}
        self.assertEqual(groups, {'failing', 'control'})
        self.assertEqual(sum(t['group'] == 'failing' for t in trials), 64)
        self.assertEqual(sum(t['group'] == 'control' for t in trials), 64)

    def test_every_roster_layout_is_an_approved_reserved_layout(self):
        e = self.e
        approved = set(e.broad_layout_ids())
        for lid in e.MODE_LAYOUTS:
            self.assertIn(lid, approved, f'{lid} is not one of the reserved holdout layouts')

    def test_frozen_seeds_are_retained_and_derived_seeds_are_reproducible(self):
        import hashlib
        e = self.e
        self.assertEqual(e.MODE_SEEDS[:2], e.FROZEN_SEEDS)
        regenerated = tuple(int.from_bytes(hashlib.sha256(f'placement_modes:{i}'.encode()).digest()[:4], 'big')
                            for i in range(6))
        self.assertEqual(e.DERIVED_SEEDS, regenerated)
        self.assertEqual(len(set(e.MODE_SEEDS)), 8)
        for s in e.MODE_SEEDS:
            self.assertTrue(0 <= s < 2**32)

    def test_seed_trial_rejects_unapproved_seeds(self):
        e = self.e
        for bad in (12345, 'x', None, 3.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                e.seed_trial_broad(bad)

    def test_this_protocol_is_not_a_capability_gate(self):
        e = self.e
        rows = [dict(seed=t['seed'], layout_id=t['layout_id'], task_success=True, outcome='success')
                for t in e.evaluation_plan_broad()]
        s = e.summarize_broad(rows)
        self.assertEqual(s['successes'], 128)
        self.assertFalse(s['capability_passed'], 'a measurement protocol must never claim a gate')
        self.assertTrue(s['is_measurement_protocol'])
        self.assertEqual(s['required_successes'], 0)
        self.assertEqual(len(s['per_layout']), 16)
        self.assertEqual({g['group'] for g in s['per_layout'].values()}, {'failing', 'control'})
        self.assertTrue(all(g['trials'] == 8 for g in s['per_layout'].values()))

    def test_output_directory_cannot_collide_with_a_sealed_evaluation(self):
        e = self.e
        name = e.evaluation_dirname(e.FINAL_UPDATE)
        self.assertEqual(name, 'evaluation_placement_modes')
        self.assertNotEqual(name, 'evaluation')
        self.assertTrue(e.evaluation_dirname(4000).startswith('evaluation_placement_modes_step_'))

    def test_modes_are_restricted_to_the_measurement_protocol(self):
        self.assertEqual(self.e.MODES, ('placement_modes',))

    def test_broad_layout_ids_disjoint_from_training_and_validation(self):
        plan = _fake_plan()
        used = {episode['layout_id'] for episode in plan['episodes']}
        reserved = set(self.e.broad_layout_ids())
        self.assertEqual(reserved & used, set(), 'evaluation layouts must never appear in training/validation')
        self.assertEqual(len(reserved), 36)

    def test_fixed_parity_roster_is_20_seeds_with_matching_ordinals(self):
        trials = self.e.evaluation_plan_fixed_parity()
        self.assertEqual(len(trials), 20)
        self.assertEqual([t['seed'] for t in trials], list(range(20260918, 20260938)))
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 21)))
        for t in trials:
            self.assertEqual(t['source_episode_index'], 7)
            self.assertEqual(t['reset_file'], 'source.npz')
            self.assertEqual(t['seed'], 20260917+t['ordinal'])


class SeedGateTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_seed_trial_broad_accepts_only_frozen_policy_seeds(self):
        approved = self.e.broad_policy_seeds()
        self.assertEqual(len(approved), 2)
        for seed in approved: self.e.seed_trial_broad(seed)
        for bad in (0, -1, 20260918, 20260937, 7):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                self.e.seed_trial_broad(bad)
        with self.assertRaises(ValueError): self.e.seed_trial_broad(str(approved[0]))
        with self.assertRaises(ValueError): self.e.seed_trial_broad(True)

    def test_seed_trial_fixed_parity_accepts_only_twenty_approved_seeds(self):
        for seed in range(20260918, 20260938): self.e.seed_trial_fixed_parity(seed)
        for bad in (20260917, 20260938, 7):
            with self.assertRaises(ValueError): self.e.seed_trial_fixed_parity(bad)
        self.e.seed_trial_fixed_parity(20260918); a = (random.random(), np.random.rand(), torch.rand(()).item())
        self.e.seed_trial_fixed_parity(20260918); b = (random.random(), np.random.rand(), torch.rand(()).item())
        self.assertEqual(a, b)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def _broad_rows(self):
        trials = self.e.evaluation_plan_broad()
        return [dict(layout_id=t['layout_id'], seed=t['seed'],
                     task_success=True, outcome='success') for t in trials]

    def _fixed_rows(self):
        return [dict(seed=20260918+i, task_success=True, outcome='success') for i in range(20)]

    def test_broad_summarize_gate_at_65_of_72(self):
        """No gate exists here: a measurement protocol must not be readable as a capability claim."""
        rows = self._broad_rows()
        summary = self.e.summarize_broad(rows)
        self.assertEqual(summary['successes'], 128)
        self.assertEqual(summary['required_successes'], 0)
        self.assertFalse(summary['capability_passed'])
        for i in range(8):
            rows[i].update(task_success=False, outcome='timeout')
        summary = self.e.summarize_broad(rows)
        self.assertEqual(summary['successes'], 120)
        self.assertFalse(summary['capability_passed'])
        self.assertAlmostEqual(summary['success_rate'], 120/128)

    def test_broad_summarize_rejects_reordered_or_short_rosters(self):
        rows = self._broad_rows()
        for bad in (rows[:-1], rows[:-1]+[rows[0]], rows+[rows[0]],
                    [{**r, 'outcome': 'nonfinite_action'} for r in rows]):
            self.assertFalse(self.e.summarize_broad(bad)['capability_passed'])
            with self.assertRaises(ValueError): self.e.require_complete_trials_broad(bad)

    def test_fixed_parity_summarize_gate_at_18_of_20(self):
        rows = self._fixed_rows()
        self.assertTrue(self.e.summarize_fixed_parity(rows)['capability_passed'])
        rows[0].update(task_success=False, outcome='timeout')
        rows[1].update(task_success=False, outcome='timeout')
        self.assertTrue(self.e.summarize_fixed_parity(rows)['capability_passed'])
        rows[2].update(task_success=False, outcome='timeout')
        self.assertFalse(self.e.summarize_fixed_parity(rows)['capability_passed'])


class TrainingBytesTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_training_bytes_excludes_evaluation_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'checkpoints'/'step_005000').mkdir(parents=True)
            (root/'checkpoints'/'step_005000'/'model.safetensors').write_bytes(b'weights')
            (root/'training_stats.json').write_text('{}')
            (root/'result.json').write_text('{}')
            baseline = self.e.training_bytes(root)
            self.assertEqual(set(baseline), {'checkpoints/step_005000/model.safetensors',
                                             'training_stats.json', 'result.json'})
            (root/'evaluation').mkdir()
            (root/'evaluation'/'result.json').write_text('{"mutating": true}')
            (root/'evaluation'/'broad_holdout').mkdir()
            (root/'evaluation'/'broad_holdout'/'result.json').write_text('{}')
            self.assertEqual(self.e.training_bytes(root), baseline)


class VerifyFinalTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()
        self.c = self.e.c

    def _fixture(self):
        tmp = tempfile.TemporaryDirectory(); root = Path(tmp.name)
        output = root/'main'; output.mkdir()
        scratch = root/'scratch'; scratch.mkdir()
        admission = root/'admission'; admission.mkdir()
        (output/'training_stats.json').write_text(json.dumps({'action': {'mean': [0.]*8}}))
        (admission/'result.json').write_text(json.dumps({'training_stats': {'action': {'mean': [0.]*8}}}))
        (scratch/'result.json').write_text(json.dumps({'kind': 'scratch'}))
        (output/'initial_config').mkdir()
        config = dict(type='xvla', chunk_size=30, n_action_steps=10, n_obs_steps=1, dtype='float32',
                      input_features={k: dict(shape=list(s)) for k, s in self.c.OBS.items()},
                      output_features={'action': dict(shape=[8])})
        (output/'initial_config'/'config.json').write_text(json.dumps(config))
        (output/'checkpoints').mkdir()
        (output/'checkpoints'/self.c.checkpoint_directory(FINAL)).mkdir()
        (output/'checkpoints'/self.c.checkpoint_directory(FINAL)/'config.json').write_text(json.dumps(config))
        binding = {'dataset_fingerprint': 'x'*64}
        report = {**self.c.recipe(False), 'status': 'complete', 'completed_updates': FINAL,
                  'selected_checkpoint_update': FINAL, 'final_data_binding_passed': True, 'binding': binding,
                  'training_rows': self.c.TRAINING_ROWS,
                  'complete_training_chunks': self.c.COMPLETE_STARTS,
                  'scratch_used_for_initialization': False,
                  'cpu_admission_sha256': self.c.sha256(admission/'result.json'),
                  'training_stats_sha256': self.c.sha256(output/'training_stats.json'),
                  'scratch_result_sha256': self.c.sha256(scratch/'result.json')}
        (output/'result.json').write_text(json.dumps(report))
        return tmp, output, scratch, admission, binding, report, config

    def test_verify_final_accepts_current_report(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints') as vc, \
                    patch.object(self.c, 'validate_scratch') as vs:
                self.assertEqual(self.e.verify_final(output, binding),
                                 output/'checkpoints'/self.c.checkpoint_directory(FINAL))
                vc.assert_called_once_with(report, output, False)
                self.assertEqual(vs.call_count, 1)
        finally: tmp.cleanup()

    def test_verify_final_rejects_tampered_completed_updates(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            report['completed_updates'] = 4999
            (output/'result.json').write_text(json.dumps(report))
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints'), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()

    def test_verify_final_rejects_changed_training_stats_hash(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            report['training_stats_sha256'] = 'x'*64
            (output/'result.json').write_text(json.dumps(report))
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints'), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()

    def test_verify_final_rejects_changed_scratch_seal(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            report['scratch_result_sha256'] = 'x'*64
            (output/'result.json').write_text(json.dumps(report))
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints'), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()

    def test_verify_final_rejects_checkpoint_hash_drift(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints', side_effect=ValueError('checkpoint hash drift')), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()

    def test_verify_final_rejects_changed_final_config_schema(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            broken = {**config, 'n_action_steps': 20}
            (output/'checkpoints'/self.c.checkpoint_directory(FINAL)/'config.json').write_text(json.dumps(broken))
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints'), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()

    def test_verify_final_rejects_wrong_config_type(self):
        tmp, output, scratch, admission, binding, report, config = self._fixture()
        try:
            broken = {**config, 'type': 'smolvla'}
            (output/'checkpoints'/self.c.checkpoint_directory(FINAL)/'config.json').write_text(json.dumps(broken))
            with patch.object(self.c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(self.c, 'SCRATCH', scratch), \
                    patch.object(self.c, 'verify_checkpoints'), \
                    patch.object(self.c, 'validate_scratch'):
                with self.assertRaises(ValueError): self.e.verify_final(output, binding)
        finally: tmp.cleanup()


class RunOverwriteGuardTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_run_refuses_to_overwrite_existing_evaluation_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'main'
            (output/self.e.evaluation_dirname(self.e.FINAL_UPDATE)).mkdir(parents=True)
            (output/'training_stats.json').write_text('{}')
            with patch.object(self.e, 'verify_final', side_effect=AssertionError('should not reach')), \
                    patch.object(self.e.c, 'prepare_data', side_effect=AssertionError('should not reach')):
                with self.assertRaises(FileExistsError): self.e.run(output)

    def test_run_never_writes_into_a_sealed_evaluation_directory(self):
        """Regression: this protocol shares an output root with sealed capability evaluations."""
        e = self.e
        for update in e.c.checkpoint_updates(False):
            name = e.evaluation_dirname(update)
            self.assertTrue(name.startswith('evaluation_placement_modes'))
            self.assertNotEqual(name, 'evaluation')
            self.assertNotEqual(name, f'evaluation_step_{update:06d}')

    def test_run_rejects_unknown_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'main'; output.mkdir()
            with self.assertRaises(ValueError): self.e.run(output, mode='unauthorized')


class VariedEntryTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_reserved_layouts_exceed_the_pilot_workspace_bound(self):
        """The pilot geometry gate caps |y| at 0.40; F7 reaches 0.45.

        Regression for the first eval run, which died on evaluation_000 because
        restore_scene applied pilot_integrity bounds to F7 scenes.
        """
        bank = self._real_bank()
        ys = []
        for lid in self.e.broad_layout_ids():
            xy = bank[lid]['layout']['cube_xy_m']
            ys.append(max(abs(xy[1]), abs(xy[3])))
        self.assertGreater(max(ys), 0.40, 'holdout must actually exercise the wide envelope')
        self.assertGreater(sum(1 for y in ys if y > 0.40), 0)
        source = (Path(__file__).resolve().parents[1]/'evaluate.py').read_text()
        self.assertNotIn('from validation.evaluate_pilot import restore_scene', source)
        self.assertIn('_validate_broad_geometry', source)

    def _real_bank(self):
        return self.e._load_bank_manifest()

    def test_broad_entry_binds_materialized_hash_and_scene_hash(self):
        approved = list(self.e.broad_layout_ids())
        bank = {lid: {'scene_hash': 'bb'*32, 'layout': {'cube_xy_m': [.30, -.30, .30, .30]}} for lid in approved}
        entry = self.e.broad_entry(approved[0], bank, 'aa'*32)
        self.assertEqual(entry['file'], f'{approved[0]}.npz')
        self.assertEqual(entry['sha256'], 'aa'*32)
        self.assertEqual(entry['scene_hash'], 'bb'*32)
        self.assertEqual(entry['colors'], ['red', 'green'])
        self.assertEqual(entry['pick'], 'red')
        self.assertEqual(entry['place'], 'green')
        with self.assertRaises(ValueError):
            self.e.broad_entry('train_000', bank, 'aa'*32)

    def test_materialize_broad_scene_rejects_a_bank_hash_mismatch(self):
        # The derive-and-prove gate is what replaces local-varied's stored scene npz.
        import tempfile
        from pathlib import Path as _P
        plan = self.e._plan()
        bank = self.e._load_bank_manifest()
        lid = self.e.broad_layout_ids()[0]
        tampered = {**bank, lid: {**bank[lid], 'arrays_sha256': {**bank[lid]['arrays_sha256'], 'qpos': 'cc'*32}}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.e._materialize_broad_scene(_P(tmp), lid, plan, tampered)
            self.assertFalse((_P(tmp)/f'{lid}.npz').exists(), 'no scene may be written when the proof fails')


class OfflineProbeTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_offline_probes_iterate_all_200_episodes_with_three_frames_each(self):
        class Dataset:
            def __getitem__(self, i): return frame(i//528)

        class Tiny(torch.nn.Module):
            calls = 0
            def predict_action_chunk(self, batch):
                self.calls += 1
                if set(batch) != {'observation.state', 'observation.images.image', 'observation.images.image2', 'task'}:
                    raise AssertionError('prediction input leakage')
                return torch.ones(1, 30, 8)

        def _pre(observation):
            return {**observation, 'action': None}

        model = Tiny(); model.train()
        result = self.e.offline_probes(model, _pre, lambda x: x, Dataset(), device='cpu')
        self.assertEqual(model.calls, 200*3)  # 160 train + 40 validation episodes.
        self.assertEqual((result['train']['probes'], result['validation']['probes']), (160*3, 40*3))
        self.assertEqual({r['frame'] for r in result['rows']}, {60, 107, 347})
        self.assertTrue(model.training)  # restored


class CLITests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_evaluate_flag_defaults_to_both_and_accepts_explicit_modes(self):
        self.assertEqual(self.e.parse_args([]).mode, 'placement_modes')
        self.assertEqual(self.e.parse_args(['--mode=placement_modes']).mode, 'placement_modes')

    def test_evaluate_flag_is_mutually_exclusive_and_rejects_unknown_modes(self):
        for argv in (['--mode=weekly'], ['--mode'], ['--evaluate'],
                     ['--mode=broad_holdout'], ['--mode=fixed_parity'], ['--mode=both']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit): self.e.parse_args(argv)

    def test_main_dispatches_evaluate_to_evaluate_run(self):
        e = self.e
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e.c, 'OUTPUT', Path(tmp)/'main'), \
                patch('experiments.xvla_placement_modes.evaluate.run') as run:
            e.main(['--mode=placement_modes'])
            run.assert_called_once_with(Path(tmp)/'main', 'placement_modes', FINAL)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e.c, 'OUTPUT', Path(tmp)/'main'), \
                patch('experiments.xvla_placement_modes.evaluate.run') as run:
            e.main([])
            run.assert_called_once_with(Path(tmp)/'main', 'placement_modes', FINAL)

    def test_checkpoint_flag_selects_an_earlier_recorded_checkpoint(self):
        e = self.e
        self.assertEqual(e.parse_args(['--checkpoint', '4000']).checkpoint, 4000)
        for argv in (['--checkpoint', '2500'], ['--checkpoint', '7777'], ['--checkpoint', 'final']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit): e.parse_args(argv)
        # This protocol must never write where a sealed capability evaluation lives.
        self.assertEqual(e.evaluation_dirname(FINAL), 'evaluation_placement_modes')
        self.assertEqual(e.evaluation_dirname(4000), 'evaluation_placement_modes_step_004000')
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e.c, 'OUTPUT', Path(tmp)/'main'), \
                patch('experiments.xvla_placement_modes.evaluate.run') as run:
            e.main(['--mode=placement_modes', '--checkpoint=4000'])
            run.assert_called_once_with(Path(tmp)/'main', 'placement_modes', 4000)


if __name__ == '__main__': unittest.main()
