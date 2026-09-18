"""CPU contracts: real adapter, hostile metadata and byte-bound admission."""
import ast
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

TASK = 'pick up the red cube and place it on top of the green cube'


def metadata():
    return [dict(episode_index=i, length=528, tasks=[TASK],
                 dataset_from_index=i*528, dataset_to_index=(i+1)*528) for i in range(24)]


def frame(episode=0):
    return {'episode_index': torch.tensor(episode), 'task': TASK,
            'observation.state': torch.zeros(9),
            'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512),
            'action': torch.zeros(50, 8), 'action_is_pad': torch.zeros(50, dtype=torch.bool)}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'contract.py').exists(), 'fixed SmolVLA adapter missing')
        self.c = importlib.import_module('experiments.smolvla_fixed_layout.contract')

    def test_counts_last_offset_and_no_validation_sampling(self):
        rows, starts = self.c.indices(metadata())
        self.assertEqual((len(rows), len(starts)), (10560, 9580))
        self.assertEqual(rows, list(range(10560)))
        self.assertEqual(starts[-1], 19*528+478)
        self.assertTrue(all(i//528 < 20 and i % 528 <= 478 for i in starts))

    def test_reject_pilot_nine_episodes_and_malformed_boundaries(self):
        for episodes in (metadata()[:9], metadata()+metadata()[:1]):
            with self.assertRaises(ValueError): self.c.indices(episodes)
        for change in ({'episode_index': 300}, {'length': 527}, {'tasks': ['wrong']},
                       {'dataset_from_index': 0}, {'dataset_to_index': 10560}):
            episodes = metadata(); episodes[20].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.c.indices(episodes)

    def test_train_statistics_exclude_validation_and_retain_prior_quantiles(self):
        class Rows:
            def __getitem__(self, i):
                if i >= 10560: raise AssertionError('validation accessed for stats')
                return dict(episode_index=i//528, task_index=0,
                            action=torch.full((8,), float(i % 2)),
                            **{'observation.state': torch.full((9,), float(i % 2))})
        stats = self.c.selected_stats(Rows(), list(range(10560)))
        self.assertEqual(set(stats['action']), {'mean', 'std', 'min', 'max', 'q01', 'q99'})
        self.assertEqual(stats['action']['mean'].tolist(), [.5]*8)
        self.assertEqual(stats['action']['std'].tolist(), [.5]*8)
        for rows in (list(range(10561)), [0]*10560, list(range(1, 10561))):
            with self.assertRaises(ValueError): self.c.selected_stats(Rows(), rows)

    def test_whitelist_real_state_action_padding_and_train_membership(self):
        row = {**frame(), 'success': True, 'object_positions': [1, 2, 3]}
        selected = self.c.policy_row(row, training=True)
        self.assertEqual(set(selected), set(self.c.OBS)|{'task', 'action', 'action_is_pad'})
        self.assertEqual(set(self.c.observation(selected)), set(self.c.OBS)|{'task'})
        for change in ({'episode_index': torch.tensor(20)}, {'task': 'wrong'},
                       {'observation.objects': torch.ones(3)}, {'action': torch.zeros(30, 8)},
                       {'action_is_pad': torch.ones(50, dtype=torch.bool)},
                       {'observation.state': torch.zeros(32)}, {'action': torch.full((50, 8), float('nan'))}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.c.policy_row({**row, **change}, training=True)
        self.c.policy_row(frame(23), training=False)
        with self.assertRaises(ValueError): self.c.collate_rows([frame(20)]*8)
        with self.assertRaises(ValueError): self.c.collate_rows([frame()]*4)

    def test_output_existing_mount_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError): self.c.claim_output(root)
            with patch.object(self.c, 'is_mountpoint', return_value=True):
                self.c.claim_output(root)
                with self.assertRaises(FileExistsError): self.c.claim_output(root)
            new = root/'new'; self.c.claim_output(new)
            self.assertTrue((new/'run_claim.json').is_file())
            link = root/'link'; link.symlink_to(new, target_is_directory=True)
            with self.assertRaises(ValueError): self.c.claim_output(link)

    def test_cache_manifest_and_all_file_bytes_are_required(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'policy').mkdir(); (root/'vlm').mkdir()
            for name in ('policy/config.json', 'policy/model.safetensors', 'vlm/config.json',
                         'vlm/tokenizer.json', 'vlm/tokenizer_config.json', 'vlm/preprocessor_config.json'):
                (root/name).write_bytes(b'fixture')
            manifest = {'policy': dict(repo_id='lerobot/smolvla_base', revision=c.REVISION, path=str(root/'policy')),
                        'vlm': dict(repo_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                                    revision=c.VLM_REVISION, path=str(root/'vlm'))}
            (root/'manifest.json').write_text(json.dumps(manifest))
            digest = c.digest_json(c.inventory(root))
            with patch.object(c, 'CACHE', root), patch.object(c, 'CACHE_DIGEST', digest):
                proof = c.require_assets()
                self.assertEqual(proof['inventory_digest'], digest)
                (root/'policy/model.safetensors').write_bytes(b'changed')
                with self.assertRaises(ValueError): c.require_assets()
                (root/'policy/model.safetensors').write_bytes(b'fixture')
                (root/'extra').write_text('unlisted')
                with self.assertRaises(ValueError): c.require_assets()

    def test_cpu_binding_semantic_stats_and_processor_tampering_rejected(self):
        c = self.c
        semantic = {'status': 'passed', 'training_ready': True}
        binding = {'semantic_sha256': c.digest_json(semantic)}
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 8})
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
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 8})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cpu.json'; path.write_text(json.dumps(report))
            with patch.object(c, 'CPU_ADMISSION', path), patch.object(c, 'current_binding', return_value=binding), \
                    patch('experiments.fixed_layout.validate.require_fixed_layout_pass', side_effect=AssertionError('repeat')):
                self.assertEqual(c.admission(), binding)
                with patch.object(c, 'current_binding', return_value={**binding, 'assets': 'changed'}):
                    with self.assertRaises(ValueError): c.admission()

    def test_actual_data_and_cpu_processor_no_model_no_physics(self):
        from experiments.smolvla_fixed_layout.runner import processor_check
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        c = self.c
        with patch.object(SmolVLAPolicy, '__init__', side_effect=AssertionError('model construction')), \
                patch('mujoco.mj_step', side_effect=AssertionError('physics')):
            ds, starts, stats = c.load_data()
            self.assertEqual((len(ds), len(starts)), (12672, 9580))
            self.assertEqual(len(stats['action']['q99']), 8)
            evidence = processor_check(ds, c.PROCESSOR_ROWS, stats)
            self.assertEqual(evidence['batch_size'], 8)
            self.assertEqual(evidence['action_shape'], [8, 50, 8])
            self.assertTrue(evidence['physical_action_roundtrip'])
            self.assertFalse(evidence['whole_policy_constructed'])

    def test_actual_binding_includes_all_new_files_and_no_xvla_weights(self):
        c = self.c
        with patch('experiments.fixed_layout.validate.require_fixed_layout_pass', side_effect=AssertionError('semantic')):
            binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], '403b8ec2d16094863faf611a13768e770dbf19a363c477f0f509fb67d8d8fba5')
        paths = {str(p.relative_to(c.ROOT)) for p in Path(__file__).parents[1].rglob('*') if p.is_file()}
        self.assertTrue(paths <= set(binding['sources']))
        self.assertIn('installed/lerobot/policies/smolvla/processor_smolvla.py', binding['sources'])
        self.assertFalse(any('xvla' in k for k in binding['sources']))
        self.assertEqual(len(binding['assets']['files']), 42)

    def test_direct_smolvla_imports_are_explicitly_hash_bound(self):
        import lerobot
        root = Path(lerobot.__file__).parent
        bound = self.c.source_hashes()
        for name in ('modeling_smolvla.py', 'smolvlm_with_expert.py', 'configuration_smolvla.py', 'processor_smolvla.py'):
            tree = ast.parse((root/'policies/smolvla'/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module.startswith('lerobot.'):
                    relative = Path(node.module.removeprefix('lerobot.').replace('.', '/'))
                    path = relative.with_suffix('.py') if (root/relative.with_suffix('.py')).is_file() else relative/'__init__.py'
                    self.assertIn('installed/lerobot/'+str(path), bound)
        for name in ('models/smolvlm/modeling_smolvlm.py', 'models/smolvlm/processing_smolvlm.py',
                     'models/llama/modeling_llama.py', 'tokenization_utils_fast.py'):
            self.assertIn('installed/transformers/'+name, bound)


if __name__ == '__main__': unittest.main()