"""Behavioral regression tests: paths, exclusivity, allowlists and precision."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file


class PackagingTests(unittest.TestCase):
    def modules(self):
        self.assertIsNotNone(importlib.util.find_spec('publication.package_baselines'),
                             'Portable packaging implementation is missing')
        self.assertIsNotNone(importlib.util.find_spec('publication.load_baseline'),
                             'Self-contained loader implementation is missing')
        from publication import package_baselines, load_baseline
        return package_baselines, load_baseline

    def test_rewrites_only_paths_and_device_not_stale_scheduler(self):
        package, _ = self.modules()
        config = dict(type='smolvla', device='cuda', load_vlm_weights=False,
                      vlm_model_name='/old/cache/vlm', scheduler_warmup_steps=1000,
                      scheduler_decay_steps=30000, scheduler_decay_lr=2.5e-6)
        pre = {'steps': [
            {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': '/old/cache/vlm'}},
            {'registry_name': 'device_processor', 'config': {'device': 'cuda'}},
            {'registry_name': 'normalizer_processor', 'config': {}, 'state_file': 'norm.safetensors'}]}
        post = {'steps': [{'registry_name': 'device_processor', 'config': {'device': 'cpu'}}]}
        originals = copy.deepcopy((config, pre, post))
        docs, changes = package.portable_documents(config, pre, post)
        self.assertEqual((config, pre, post), originals)
        expected = dict(config, device='cpu', vlm_model_name='assets/vlm')
        self.assertEqual(docs['config.json'], expected)
        self.assertEqual(docs['policy_preprocessor.json']['steps'][0]['config']['tokenizer_name'], 'assets/vlm')
        self.assertEqual(docs['policy_preprocessor.json']['steps'][1]['config']['device'], 'cpu')
        self.assertEqual(len(changes), 4)
        self.assertNotIn('/old/', json.dumps(changes))

    def test_unreviewed_absolute_path_is_rejected_not_silently_scrubbed(self):
        package, _ = self.modules()
        with self.assertRaises(ValueError):
            package.portable_documents({'type': 'xvla', 'device': 'cuda',
                                        'tokenizer_name': '/old/tokenizer',
                                        'surprise': '/home/someone/checkpoint'}, {'steps': []}, {'steps': []})

    def test_no_existing_or_partial_packages_root_can_be_reused(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'packages'
            package.claim_packages(root)
            marker = root / 'partial'
            marker.write_text('keep')
            with self.assertRaises(FileExistsError):
                package.claim_packages(root)
            self.assertEqual(marker.read_text(), 'keep')

    def test_claim_rejects_symlinked_parent(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'real').mkdir()
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaises(ValueError):
                package.claim_packages(root / 'alias' / 'packages')

    def test_asset_copy_ignores_internal_cache_and_weights(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'src', Path(tmp) / 'dst'
            source.mkdir()
            for name in package.XVLA_ASSETS:
                (source / name).write_text('{}')
            (source / '.cache').mkdir()
            (source / '.cache' / 'download-token').write_text('must not copy')
            (source / 'model.safetensors').write_text('must not copy')
            package.copy_assets(source, target, package.XVLA_ASSETS)
            self.assertEqual({p.name for p in target.iterdir()}, set(package.XVLA_ASSETS))
            self.assertNotEqual((source / 'config.json').stat().st_ino,
                                (target / 'config.json').stat().st_ino)

    def test_missing_required_asset_fails(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                package.copy_assets(Path(tmp), Path(tmp) / 'out', package.XVLA_ASSETS)

    def test_text_audit_rejects_credentials_and_machine_paths_without_echo(self):
        package, _ = self.modules()
        scan = getattr(package, 'audit_text', None)
        self.assertIsNotNone(scan, 'Release text scan is required')
        for text in ('value=' + 'hf_' + 'a' * 32,
                     'asset=/home/example/private/model', 'asset=/vla/local/model',
                     '-----BEGIN PRIVATE KEY-----'):
            with self.subTest(kind=text[:5]), self.assertRaises(ValueError) as caught:
                scan(text)
            self.assertNotIn(text, str(caught.exception))
        scan('assets/vlm; revision=' + 'a' * 40)

    def test_complete_staging_audits_rewrite_metadata_and_excludes_training_files(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / 'outputs/train/smolvla_fixed_layout_20260918/checkpoints/005000'
            source.mkdir(parents=True)
            package.write_json(source / 'config.json', dict(type='smolvla', device='cuda',
                               load_vlm_weights=False, vlm_model_name='/original/vlm',
                               scheduler_warmup_steps=1000))
            for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
                package.write_json(source / name, {'steps': []})
            save_file({'weight': torch.tensor([1.0001])}, source / 'model.safetensors')
            (source / 'optimizer.pt').write_bytes(b'never copy')
            (source / 'training_state.pt').write_bytes(b'never copy')
            package.write_json(source.parents[1] / 'result.json',
                               dict(completed_updates=5000, schedule={'warmup': 500}))
            cache = repo / '.cache/benchmark_models/vlm'
            cache.mkdir(parents=True)
            for name in package.SMOL_ASSETS:
                (cache / name).write_text('{}')
            package.write_json(cache.parent / 'manifest.json',
                               {'vlm': {'repo_id': 'example/vlm', 'revision': 'a' * 40}})
            packages = package.claim_packages(repo / 'packages')
            package.stage_model(repo, packages, 'smolvla')
            release = packages / 'smolvla'
            self.assertEqual((release / 'model.safetensors').read_bytes(),
                             (source / 'model.safetensors').read_bytes())
            self.assertFalse(list(release.rglob('*.pt')))
            manifest = json.loads((release / 'release_manifest.json').read_text())
            package.audit_json(manifest)
            self.assertEqual(manifest['actual_training_recipe']['schedule']['warmup'], 500)
            self.assertEqual(json.loads((release / 'config.json').read_text())['scheduler_warmup_steps'], 1000)
            self.assertEqual(json.loads((release / 'inventory.json').read_text()),
                             package.inventory(release, exclude=('inventory.json',)))

    def test_bundled_paths_resolve_relative_to_release_not_cwd(self):
        _, loader = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'assets' / 'vlm').mkdir(parents=True)
            self.assertEqual(loader.bundled_path(root, 'assets/vlm'), str(root / 'assets' / 'vlm'))
            for bad in ('/tmp', '../outside', 'remote/model', 'assets/../../outside'):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    loader.bundled_path(root, bad)

    def test_fp32_promotion_precedes_copy_and_frozen_bf16_is_unchanged(self):
        _, loader = self.modules()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trained = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
                self.frozen = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.safetensors'
            expected = {'trained': torch.tensor([1.0001, -0.9999]),
                        'frozen': torch.tensor([1.5, -2.5], dtype=torch.bfloat16)}
            save_file(expected, path)
            policy = Tiny()
            loader.strict_load_weights(policy, path, 'smolvla')
            self.assertEqual(policy.trained.dtype, torch.float32)
            self.assertEqual(policy.frozen.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(policy.trained, expected['trained']))
            self.assertTrue(torch.equal(policy.frozen, expected['frozen']))

    def test_missing_extra_or_wrong_dtype_weights_fail_strictly(self):
        _, loader = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.safetensors'
            for weights in ({}, {'weight': torch.ones(1, 2), 'extra': torch.zeros(1)},
                            {'weight': torch.ones(1, 2, dtype=torch.bfloat16)}):
                save_file(weights, path)
                with self.subTest(keys=list(weights)), self.assertRaises((ValueError, RuntimeError)):
                    loader.strict_load_weights(torch.nn.Linear(2, 1, bias=False), path, 'xvla')

    def test_streamed_compare_detects_dtype_value_and_key_corruption(self):
        package, _ = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.safetensors', Path(tmp) / 'b.safetensors'
            save_file({'a': torch.tensor([0.0, 1.0001])}, a)
            save_file({'a': torch.tensor([0.0, 1.0001])}, b)
            self.assertEqual(package.compare_safetensors(a, b)['tensors'], 1)
            for wrong in ({'a': torch.tensor([-0.0, 1.0001])},
                          {'a': torch.tensor([0.0, 1.0])},
                          {'a': torch.tensor([0.0, 1.0], dtype=torch.bfloat16)},
                          {'different': torch.zeros(2)}):
                save_file(wrong, b)
                with self.subTest(wrong=list(wrong)), self.assertRaises(ValueError):
                    package.compare_safetensors(a, b)


if __name__ == '__main__':
    unittest.main()