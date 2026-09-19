"""CPU-only evaluator tests: rosters, gates, offline probes, and verify_final tamper checks."""
import importlib
import inspect
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'smolvla_local_varied'/'tests'))
from test_contract import frame


def _module():
    return importlib.import_module('experiments.smolvla_local_varied_eval.evaluate')


def _fake_plan():
    """Full local-varied plan; identity stays deterministic across imports."""
    from experiments.local_varied_data import plan as p
    return p.make_plan()


class RosterTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_varied_roster_is_24_layouts_by_3_seeds_layout_major(self):
        trials = self.e.evaluation_plan_varied()
        plan = _fake_plan()
        self.assertEqual(len(trials), 72)
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 73)))
        # Layout-major, seed-minor cartesian.
        expected = [(layout, seed) for layout in plan['evaluation_layout_ids'] for seed in plan['policy_seeds']]
        self.assertEqual([(t['layout_id'], t['seed']) for t in trials], expected)
        self.assertEqual({t['seed'] for t in trials}, set(plan['policy_seeds']))
        self.assertEqual({t['layout_id'] for t in trials}, set(plan['evaluation_layout_ids']))

    def test_varied_layout_ids_disjoint_from_training_and_validation(self):
        plan = _fake_plan()
        used = {episode['layout_id'] for episode in plan['episodes']}
        reserved = set(self.e.varied_layout_ids())
        self.assertEqual(reserved & used, set(), 'evaluation layouts must never appear in training/validation')
        self.assertEqual(len(reserved), 24)

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

    def test_seed_trial_varied_accepts_only_three_frozen_policy_seeds(self):
        approved = self.e.varied_policy_seeds()
        self.assertEqual(len(approved), 3)
        for seed in approved: self.e.seed_trial_varied(seed)
        for bad in (0, -1, 20260918, 20260937, 7):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                self.e.seed_trial_varied(bad)
        with self.assertRaises(ValueError): self.e.seed_trial_varied(str(approved[0]))
        with self.assertRaises(ValueError): self.e.seed_trial_varied(True)  # bool is not int

    def test_seed_trial_fixed_parity_accepts_only_twenty_approved_seeds(self):
        for seed in range(20260918, 20260938): self.e.seed_trial_fixed_parity(seed)
        for bad in (20260917, 20260938, 7):
            with self.assertRaises(ValueError): self.e.seed_trial_fixed_parity(bad)
        # Deterministic across calls.
        self.e.seed_trial_fixed_parity(20260918); a = (random.random(), np.random.rand(), torch.rand(()).item())
        self.e.seed_trial_fixed_parity(20260918); b = (random.random(), np.random.rand(), torch.rand(()).item())
        self.assertEqual(a, b)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def _varied_rows(self):
        trials = self.e.evaluation_plan_varied()
        return [dict(layout_id=t['layout_id'], seed=t['seed'],
                     task_success=True, outcome='success') for t in trials]

    def _fixed_rows(self):
        return [dict(seed=20260918+i, task_success=True, outcome='success') for i in range(20)]

    def test_varied_summarize_gate_at_65_of_72(self):
        rows = self._varied_rows()
        summary = self.e.summarize_varied(rows)
        self.assertTrue(summary['capability_passed'])
        self.assertEqual(summary['required_successes'], 65)
        self.assertEqual(summary['successes'], 72)
        # Exactly one below the gate.
        rows[0].update(task_success=False, outcome='timeout')
        rows[1].update(task_success=False, outcome='timeout')
        rows[2].update(task_success=False, outcome='timeout')
        rows[3].update(task_success=False, outcome='timeout')
        rows[4].update(task_success=False, outcome='timeout')
        rows[5].update(task_success=False, outcome='timeout')
        rows[6].update(task_success=False, outcome='timeout')
        rows[7].update(task_success=False, outcome='timeout')
        summary = self.e.summarize_varied(rows)
        self.assertEqual(summary['successes'], 64)
        self.assertFalse(summary['capability_passed'])

    def test_varied_summarize_rejects_reordered_or_short_rosters(self):
        rows = self._varied_rows()
        for bad in (rows[:-1], rows[:-1]+[rows[0]], rows+[rows[0]],
                    [{**r, 'outcome': 'nonfinite_action'} for r in rows]):
            self.assertFalse(self.e.summarize_varied(bad)['capability_passed'])
            with self.assertRaises(ValueError): self.e.require_complete_trials_varied(bad)

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
            (root/'evaluation'/'varied_holdout').mkdir()
            (root/'evaluation'/'varied_holdout'/'result.json').write_text('{}')
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
        config = dict(type='smolvla', chunk_size=50, n_action_steps=10, n_obs_steps=1,
                      input_features={k: dict(shape=list(s)) for k, s in self.c.OBS.items()},
                      output_features={'action': dict(shape=[8])})
        (output/'initial_config'/'config.json').write_text(json.dumps(config))
        (output/'checkpoints').mkdir()
        (output/'checkpoints'/self.c.checkpoint_directory(5000)).mkdir()
        (output/'checkpoints'/self.c.checkpoint_directory(5000)/'config.json').write_text(json.dumps(config))
        binding = {'dataset_fingerprint': 'x'*64}
        report = {**self.c.recipe(False), 'status': 'complete', 'completed_updates': 5000,
                  'selected_checkpoint_update': 5000, 'final_data_binding_passed': True, 'binding': binding,
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
                                 output/'checkpoints'/self.c.checkpoint_directory(5000))
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
            (output/'checkpoints'/self.c.checkpoint_directory(5000)/'config.json').write_text(json.dumps(broken))
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
            (output/'evaluation').mkdir(parents=True)
            (output/'training_stats.json').write_text('{}')
            # verify_final would run only if mkdir passed; must fail first.
            with patch.object(self.e, 'verify_final', side_effect=AssertionError('should not reach')), \
                    patch.object(self.e.c, 'prepare_data', side_effect=AssertionError('should not reach')):
                with self.assertRaises(FileExistsError): self.e.run(output)

    def test_run_rejects_unknown_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'main'; output.mkdir()
            with self.assertRaises(ValueError): self.e.run(output, mode='unauthorized')


class VariedEntryTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_varied_entry_binds_snapshot_hash_and_scene_hash(self):
        approved = list(self.e.varied_layout_ids())
        snapshots = {f'scenes/{lid}.npz': 'aa'*32 for lid in approved}
        bank = {lid: {'scene_hash': 'bb'*32, 'layout': {'offsets_m': [0.001, 0.002, -.003, -.004]}} for lid in approved}
        entry = self.e.varied_entry(approved[0], snapshots, bank)
        self.assertEqual(entry['file'], f'{approved[0]}.npz')
        self.assertEqual(entry['sha256'], 'aa'*32)
        self.assertEqual(entry['scene_hash'], 'bb'*32)
        self.assertEqual(entry['colors'], ['red', 'green'])
        self.assertEqual(entry['pick'], 'red')
        self.assertEqual(entry['place'], 'green')
        # Unapproved layout id (any train layout) rejected.
        with self.assertRaises(ValueError):
            self.e.varied_entry('train_000', snapshots, bank)


class OfflineProbeTests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_offline_probes_iterate_all_60_episodes_with_three_frames_each(self):
        class Dataset:
            def __getitem__(self, i): return frame(i//528)

        class Tiny(torch.nn.Module):
            calls = 0
            def predict_action_chunk(self, batch):
                self.calls += 1
                if set(batch) != {'observation.state', 'observation.images.image', 'observation.images.image2', 'task'}:
                    raise AssertionError('prediction input leakage')
                return torch.ones(1, 50, 8)

        model = Tiny(); model.train()
        result = self.e.offline_probes(model, lambda x: x, lambda x: x, Dataset(), device='cpu')
        self.assertEqual(model.calls, 60*3)  # 48 train + 12 validation episodes.
        self.assertEqual((result['train']['probes'], result['validation']['probes']), (48*3, 12*3))
        self.assertEqual({r['frame'] for r in result['rows']}, {60, 107, 347})
        self.assertTrue(model.training)  # restored


class CLITests(unittest.TestCase):
    def setUp(self):
        self.e = _module()

    def test_evaluate_flag_defaults_to_both_and_accepts_explicit_modes(self):
        self.assertEqual(self.e.parse_args([]).mode, 'both')
        self.assertEqual(self.e.parse_args(['--mode=varied_holdout']).mode, 'varied_holdout')
        self.assertEqual(self.e.parse_args(['--mode=fixed_parity']).mode, 'fixed_parity')
        self.assertEqual(self.e.parse_args(['--mode=both']).mode, 'both')

    def test_evaluate_flag_is_mutually_exclusive_and_rejects_unknown_modes(self):
        for argv in (['--mode=weekly'], ['--mode'], ['--evaluate']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit): self.e.parse_args(argv)

    def test_main_dispatches_evaluate_to_evaluate_run(self):
        e = self.e
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e.c, 'OUTPUT', Path(tmp)/'main'), \
                patch('experiments.smolvla_local_varied_eval.evaluate.run') as run:
            e.main(['--mode=fixed_parity'])
            run.assert_called_once_with(Path(tmp)/'main', 'fixed_parity')
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e.c, 'OUTPUT', Path(tmp)/'main'), \
                patch('experiments.smolvla_local_varied_eval.evaluate.run') as run:
            e.main([])
            run.assert_called_once_with(Path(tmp)/'main', 'both')


if __name__ == '__main__': unittest.main()
