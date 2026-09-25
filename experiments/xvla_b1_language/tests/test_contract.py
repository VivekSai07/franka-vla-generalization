"""CPU contracts: X-VLA B1 language-grounding adapter, real seal admission, provenance enforcement."""
import ast
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.b1_language_data.plan import make_plan

# Fixtures are driven by the real plan so they cannot drift from the contract's expectation.
PLANNED = tuple(episode['instruction'] for episode in make_plan()['episodes'])
TASK_OF_EPISODE = tuple(sorted(set(PLANNED)).index(x) for x in PLANNED)


def metadata():
    return [dict(episode_index=i, length=528, tasks=[PLANNED[i]],
                 dataset_from_index=i*528, dataset_to_index=(i+1)*528) for i in range(200)]


def frame(episode=0):
    return {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
            'observation.state': torch.zeros(9),
            'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512),
            'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'contract.py').is_file(),
                        'X-VLA B1 language adapter missing')
        self.c = importlib.import_module('experiments.xvla_b1_language.contract')

    def test_counts_last_offset_and_no_validation_sampling(self):
        rows, starts = self.c.indices(metadata())
        # 160 * 528 rows, 160 * 499 complete-chunk starts (chunk_size=30 → 528-30+1=499).
        self.assertEqual((len(rows), len(starts)), (84480, 79840))
        self.assertEqual(rows, list(range(84480)))
        self.assertEqual(starts[-1], 159*528+498)
        self.assertTrue(all(i//528 < 160 and i % 528 <= 498 for i in starts))
        self.assertEqual(len(set(starts)), len(starts))

    def test_reject_fixed_layout_pilot_and_malformed_boundaries(self):
        for episodes in (metadata()[:20], metadata()[:24], metadata()+metadata()[:1]):
            with self.assertRaises(ValueError):
                self.c.indices(episodes)
        for change in ({'episode_index': 300}, {'length': 527}, {'tasks': ['wrong']},
                       {'dataset_from_index': 0}, {'dataset_to_index': 84480}):
            episodes = metadata()
            episodes[160].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.c.indices(episodes)

    def test_train_statistics_exclude_validation_and_retain_prior_quantiles(self):
        class Rows:
            def __getitem__(self, i):
                if i >= 84480:
                    raise AssertionError('validation accessed for stats')
                return dict(episode_index=i//528, task_index=TASK_OF_EPISODE[i//528],
                            action=torch.full((8,), float(i % 2)),
                            **{'observation.state': torch.full((9,), float(i % 2))})
        stats = self.c.selected_stats(Rows(), list(range(84480)), TASK_OF_EPISODE)
        self.assertEqual(set(stats['action']), {'mean', 'std', 'min', 'max'})
        self.assertEqual(stats['action']['mean'].tolist(), [.5]*8)
        self.assertEqual(stats['action']['std'].tolist(), [.5]*8)
        for rows in (list(range(84481)), [0]*84480, list(range(1, 84481))):
            with self.assertRaises(ValueError):
                self.c.selected_stats(Rows(), rows, TASK_OF_EPISODE)

    def test_whitelist_real_state_action_padding_and_train_membership(self):
        row = {**frame(), 'success': True, 'object_positions': [1, 2, 3]}
        selected = self.c.policy_row(row, training=True)
        self.assertEqual(set(selected), set(self.c.OBS)|{'task', 'action', 'action_is_pad'})
        for change in ({'episode_index': torch.tensor(160)}, {'task': 'wrong'},
                       {'observation.objects': torch.ones(3)}, {'action': torch.zeros(50, 8)},
                       {'action_is_pad': torch.ones(30, dtype=torch.bool)},
                       {'observation.state': torch.zeros(32)},
                       {'action': torch.full((30, 8), float('nan'))}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.c.policy_row({**row, **change}, training=True)
        # Validation episode legal only as observation-only inference row.
        self.c.policy_row(frame(199), training=False)
        with self.assertRaises(ValueError):
            self.c.policy_row(frame(199), training=True)
        with self.assertRaises(ValueError):
            self.c.collate_rows([frame(160)]*4)
        with self.assertRaises(ValueError):
            self.c.collate_rows([frame()]*3)

    def test_sampler_permutation_covers_every_training_start_once_and_only_training(self):
        _, starts = self.c.indices(metadata())
        self.assertEqual(len(starts), 160*499)
        self.assertEqual(len(set(starts)), 160*499)
        for s in starts:
            episode, offset = s//528, s % 528
            self.assertLess(episode, 160)
            self.assertLessEqual(offset, 498)
        self.assertFalse(any(s//528 >= 160 for s in starts))
        # Sampler must reproduce full epoch as permutation of the 79840 starts.
        import itertools
        it = self.c.sample_batches(starts)
        batches = list(itertools.islice(it, 79840//4))
        self.assertEqual(sorted(itertools.chain.from_iterable(batches)), starts)
        # Reproducibility: fresh iterator yields identical first epoch.
        it2 = self.c.sample_batches(starts)
        self.assertEqual(list(itertools.islice(it2, 79840//4)), batches)
        for bad in (starts[:100], starts[:-1]+[10560]):
            with self.assertRaises(ValueError):
                next(self.c.sample_batches(bad))

    def test_output_existing_mount_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                self.c.claim_output(root)
            with patch.object(self.c, 'is_mountpoint', return_value=True):
                self.c.claim_output(root)
                with self.assertRaises(FileExistsError):
                    self.c.claim_output(root)
            new = root/'new'
            self.c.claim_output(new)
            self.assertTrue((new/'run_claim.json').is_file())
            link = root/'link'
            link.symlink_to(new, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.c.claim_output(link)

    def test_require_assets_accepts_real_bundle_and_rejects_tamper(self):
        # Use the real .cache/xvla_trial/bundle via xvla_trial.assets.require_assets.
        from experiments.xvla_trial.assets import require_assets, ASSETS
        report = require_assets(self.c.CACHE)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['assets'], ASSETS)
        # Simulate tamper by patching the manifest CACHE root to a temp with missing files.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'status': 'incomplete', 'assets': ASSETS, 'files': {}}))
            with self.assertRaises(ValueError):
                require_assets(root)

    def test_cpu_binding_semantic_stats_and_processor_tampering_rejected(self):
        c = self.c
        semantic = {'status': 'passed', 'training_ready': True}
        binding = {'semantic_sha256': c.digest_json(semantic)}
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4})
        c.validate_cpu(report, binding)
        for change in ({'binding': {}}, {'training_rows': 4752}, {'complete_training_chunks': 9980},
                       {'train_episodes': list(range(9))}, {'training_stats': {}},
                       {'processor_check': {}}, {'semantic_gate': {}}, {'gpu_executed': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_cpu({**report, **change}, binding)

    def test_later_admission_rehashes_without_full_semantic_repeat(self):
        c = self.c
        semantic = {'status': 'passed', 'training_ready': True}
        binding = {'semantic_sha256': c.digest_json(semantic)}
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cpu.json'
            path.write_text(json.dumps(report))
            with patch.object(c, 'CPU_ADMISSION', path), \
                    patch.object(c, 'current_binding', return_value=binding), \
                    patch('experiments.b1_language_data.validate.require_broad_workspace_pass',
                          side_effect=AssertionError('semantic must not repeat')):
                self.assertEqual(c.admission(), binding)
                with patch.object(c, 'current_binding', return_value={**binding, 'assets': 'changed'}):
                    with self.assertRaises(ValueError):
                        c.admission()

    def _synthetic_parent(self, tmp, seal, valid):
        parent = Path(tmp)/'parent'
        parent.mkdir()
        (parent/'seal.json').write_text(json.dumps({'fingerprint': seal}))
        (parent/'plan.json').write_text('{}')
        (parent/'validation.json').write_text(json.dumps(valid))
        (parent/'source.npz').write_bytes(b'reset')
        (parent/'snapshots.json').write_text('{}')
        return parent

    def _require_sealed_parent(self):
        """The parent is produced by collection + admission; these tests are live only after that."""
        if self.c.SEAL == 'PENDING_B1_ADMISSION' or not (self.c.PARENT/'validation.json').exists():
            self.skipTest('B1 parent not collected/sealed yet')

    def test_current_binding_accepts_real_parent_and_matches_sealed_provenance(self):
        # Uses the real sealed parent; the recorded snapshot must equal a fresh provenance().
        c = self.c
        self._require_sealed_parent()
        from experiments.b1_language_data.scenes import provenance
        fresh = provenance()
        binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
        self.assertEqual(binding['b1_language_provenance'], fresh)
        recorded = json.loads((c.PARENT/'validation.json').read_text())['provenance']
        self.assertEqual(recorded, fresh)

    def test_current_binding_rejects_tampered_seal_plan_and_semantic_fields(self):
        c = self.c
        self._require_sealed_parent()
        # Recorded provenance must agree with the patched provenance() or the new check trips.
        semantic = {**json.loads((c.PARENT/'validation.json').read_text()), 'provenance': {'v': 1}}
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._synthetic_parent(tmp, c.SEAL, semantic)
            with patch.object(c, 'PARENT', parent), \
                    patch.object(c, 'require_assets', return_value={'inventory': 'x'}), \
                    patch.object(c, 'source_hashes', return_value={}), \
                    patch('experiments.b1_language_data.scenes.provenance',
                          return_value={'v': 1}):
                # Baseline passes (with a synthetic empty plan.json — validate_plan is patched off).
                with patch.object(c, 'validate_plan'):
                    binding = c.current_binding()
                    self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
                # Tamper seal fingerprint.
                (parent/'seal.json').write_text(json.dumps({'fingerprint': 'x'*64}))
                with patch.object(c, 'validate_plan'), self.assertRaises(ValueError):
                    c.current_binding()
                (parent/'seal.json').write_text(json.dumps({'fingerprint': c.SEAL}))
                # Tamper plan.json → validate_plan raises.
                with patch.object(c, 'validate_plan', side_effect=ValueError('plan tampered')), \
                        self.assertRaises(ValueError):
                    c.current_binding()
                # Tamper validation.status.
                (parent/'validation.json').write_text(json.dumps({**semantic, 'status': 'failed'}))
                with patch.object(c, 'validate_plan'), self.assertRaises(ValueError):
                    c.current_binding()
                # Tamper training_ready.
                (parent/'validation.json').write_text(json.dumps({**semantic, 'training_ready': False}))
                with patch.object(c, 'validate_plan'), self.assertRaises(ValueError):
                    c.current_binding()
                # Wrong split ids.
                (parent/'validation.json').write_text(json.dumps({**semantic, 'train_ids': list(range(24))}))
                with patch.object(c, 'validate_plan'), self.assertRaises(ValueError):
                    c.current_binding()

    def test_current_binding_compares_provenance_snapshot(self):
        # Stronger than the local-varied contract, which had to waive this check.
        c = self.c
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn("semantic.get('provenance') != current", source)
        self._require_sealed_parent()
        semantic = json.loads((c.PARENT/'validation.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._synthetic_parent(tmp, c.SEAL, semantic)
            with patch.object(c, 'PARENT', parent), \
                    patch.object(c, 'require_assets', return_value={'inventory': 'x'}), \
                    patch.object(c, 'source_hashes', return_value={}), \
                    patch.object(c, 'validate_plan'), \
                    patch('experiments.b1_language_data.scenes.provenance',
                          return_value={'drifted': True}), \
                    self.assertRaises(ValueError):
                c.current_binding()

    def test_checkpoint_naming_is_step_prefixed_and_rejects_legacy(self):
        c = self.c
        self.assertEqual(c.checkpoint_directory(1), 'step_000001')
        self.assertEqual(c.checkpoint_directory(5000), 'step_005000')
        for update in (0, -1, 1.0, '1', True):
            with self.assertRaises(ValueError):
                c.checkpoint_directory(update)

    def test_recipe_and_checkpoint_schedule(self):
        c = self.c
        for scratch in (True, False):
            recipe = c.recipe(scratch)
            self.assertEqual((recipe['microbatch'], recipe['accumulation'], recipe['effective_batch']),
                             (4, 2, 8))
            self.assertEqual((recipe['chunk_size'], recipe['n_action_steps']), (30, 10))
            self.assertEqual(recipe['max_optimizer_updates'], 2 if scratch else 33400)
            self.assertEqual(recipe['seed'], 20260917)
            self.assertEqual(recipe['train_episodes'], list(range(160)))
            self.assertEqual(recipe['validation_episodes'], list(range(160, 200)))
            self.assertFalse(recipe['resume'])
            self.assertEqual(recipe['precision'], 'fp32_parameters_gradients_adam_bf16_autocast')
            self.assertEqual(recipe['optimizer_storage'], 'final_only')
        self.assertEqual(c.checkpoint_updates(True), [1, 2])
        self.assertEqual(c.checkpoint_updates(False), [1, 8000, 16700, 25000, 33400])
        # 33,400 x 8 = 267,200 chunks over 79,840 starts = 3.35 epochs, double the 86.1% run.
        self.assertAlmostEqual(33400*8/c.COMPLETE_STARTS, 3.346, places=2)
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(33400), 1e-5)

    def test_complete_chunk_starts_count_matches_spec(self):
        # Spec requirement: 160*499 = 79840 complete-chunk starts.
        _, starts = self.c.indices(metadata())
        self.assertEqual(len(starts), 160*499)
        self.assertEqual(len(starts), 79840)
        self.assertEqual(self.c.COMPLETE_STARTS, 79840)
        self.assertEqual(self.c.TRAINING_ROWS, 84480)

    def test_verify_checkpoints_requires_step_dirs_and_rejects_legacy(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'checkpoints').mkdir()
            for update in (1, 2):
                (out/f'checkpoints/{update:06d}').mkdir()  # legacy naming
            markers = [dict(status='complete', update=u, reload_fp32_weights_exact=True,
                            reload_max_action_error=0., model_sha256='x') for u in (1, 2)]
            report = dict(checkpoints=markers,
                          checkpoint_files={c.checkpoint_directory(u): {} for u in (1, 2)})
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, out, True)

    def test_forbidden_imports_absent(self):
        # New package must NOT depend on smolvla_local_varied or xvla_fixed_layout.
        for name in ('contract.py', 'runner.py'):
            source = (Path(__file__).parents[1]/name).read_text()
            for needle in ('from experiments.smolvla_local_varied',
                           'import experiments.smolvla_local_varied',
                           'from experiments.xvla_fixed_layout',
                           'import experiments.xvla_fixed_layout',
                           'from .evaluate'):
                self.assertNotIn(needle, source, msg=f'Forbidden import in {name}: {needle}')

    def test_explicit_source_files_only(self):
        names = self.c.SOURCE_FILES
        self.assertIn('experiments/xvla_trial/preflight.py', names)
        self.assertIn('experiments/xvla_single_task/contract.py', names)
        self.assertIn('experiments/xvla_b1_language/contract.py', names)
        self.assertIn('experiments/xvla_b1_language/runner.py', names)
        self.assertIn('experiments/b1_language_data/plan.py', names)
        self.assertFalse(any('smolvla_local_varied' in n for n in names))
        self.assertFalse(any('local_varied_data' in n for n in names))
        self.assertFalse(any('xvla_fixed_layout' in n for n in names))
        self.assertFalse(any('CPU_REPORT' in n or 'CPU_ADMISSION' in n for n in names))
        self.assertEqual(len(names), len(set(names)))

    def test_cpu_admission_and_output_paths_are_isolated(self):
        c = self.c
        self.assertEqual(c.OUTPUT.name, 'xvla_franka_stacking_b1_language_v1_20260925')
        self.assertEqual(c.SCRATCH.name, 'xvla_franka_stacking_b1_language_v1_20260925_preflight')
        self.assertEqual(c.ADMISSION.name, 'xvla_franka_stacking_b1_language_v1_20260925_admission')
        # Cache is the X-VLA bundle, not the SmolVLA benchmark_models cache.
        self.assertEqual(c.CACHE, c.ROOT/'.cache/xvla_trial/bundle')

    def test_scratch_proof_recipe_binding_and_checkpoint_bytes_bound(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_json(root/'training_stats.json', {'action': {'mean': [0.]*8}})
            (root/'checkpoints').mkdir()
            markers, files = [], {}
            for i, update in enumerate((1, 2)):
                folder = root/'checkpoints'/c.checkpoint_directory(update)
                folder.mkdir()
                for name in ('model.safetensors', 'config.json',
                             'policy_preprocessor.json', 'policy_postprocessor.json'):
                    (folder/name).write_text('bytes')
                if update == 2:
                    (folder/'training_state.pt').write_bytes(b'opt')
                marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                              reload_max_action_error=0.,
                              model_sha256=c.sha256(folder/'model.safetensors'))
                c.write_json(folder/'checkpoint_result.json', marker)
                if update == 2:
                    c.write_json(folder/'adam_reload.json',
                                 dict(adam_reload_fp32_exact=True, update=update,
                                      training_state_sha256=c.sha256(folder/'training_state.pt')))
                markers.append(marker)
                files[c.checkpoint_directory(update)] = c.inventory(folder)
            binding = {'fingerprint': 'x'}
            report = {**c.recipe(True), 'status': 'complete', 'completed_updates': 2,
                      'binding': binding, 'training_rows': 84480, 'complete_training_chunks': 79840,
                      'cpu_admission_sha256': 'cpu', 'final_data_binding_passed': True,
                      'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None,
                      'rollouts': 0, 'checkpoints': markers, 'checkpoint_files': files,
                      'training_stats_sha256': c.sha256(root/'training_stats.json')}
            c.validate_scratch(report, binding, root, {'action': {'mean': [0.]*8}}, 'cpu')
            for change in ({'completed_updates': 1}, {'cpu_admission_sha256': 'stale'},
                           {'binding': {}}, {'adam_reload_fp32_exact': False},
                           {'training_rows': 10560}, {'complete_training_chunks': 22992},
                           {'rollouts': 1}, {'checkpoints': [markers[0]]}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_scratch({**report, **change}, binding, root,
                                       {'action': {'mean': [0.]*8}}, 'cpu')

    def test_intermediate_checkpoint_must_not_carry_optimizer_state(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'checkpoints').mkdir()
            markers, files = [], {}
            for update in (1, 2):
                folder = out/'checkpoints'/c.checkpoint_directory(update)
                folder.mkdir()
                for name in ('model.safetensors', 'config.json',
                             'policy_preprocessor.json', 'policy_postprocessor.json',
                             'training_state.pt'):  # optimizer present on BOTH — illegal for update=1.
                    (folder/name).write_bytes(b'x')
                marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                              reload_max_action_error=0.,
                              model_sha256=c.sha256(folder/'model.safetensors'))
                c.write_json(folder/'checkpoint_result.json', marker)
                markers.append(marker)
                files[c.checkpoint_directory(update)] = c.inventory(folder)
            c.write_json(out/'checkpoints'/c.checkpoint_directory(2)/'adam_reload.json',
                         dict(adam_reload_fp32_exact=True, update=2,
                              training_state_sha256=c.sha256(out/'checkpoints/step_000002/training_state.pt')))
            files[c.checkpoint_directory(2)] = c.inventory(out/'checkpoints/step_000002')
            report = dict(checkpoints=markers, checkpoint_files=files)
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, out, True)


if __name__ == '__main__':
    unittest.main()
