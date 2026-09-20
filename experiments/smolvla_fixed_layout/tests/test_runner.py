"""Synthetic CPU models; no pretrained model, CUDA or real training."""
from contextlib import ExitStack, nullcontext
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_contract import frame


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'runner.py').exists(), 'new runner missing')
        self.r = importlib.import_module('experiments.smolvla_fixed_layout.runner')
        self.c = self.r.c

    def test_recipe_lr_and_checkpoint_schedule(self):
        c = self.c
        for scratch in (True, False):
            recipe = c.recipe(scratch)
            self.assertEqual((recipe['microbatch'], recipe['accumulation'], recipe['effective_batch']), (8, 4, 32))
            self.assertEqual(recipe['max_optimizer_updates'], 2 if scratch else 5000)
            self.assertEqual(recipe['seed'], 20260917)
            self.assertFalse(recipe['resume'])
        self.assertEqual(c.checkpoint_updates(True), [1, 2])
        self.assertEqual(c.checkpoint_updates(False), [1, 1000, 2000, 3000, 4000, 5000])
        self.assertAlmostEqual(c.learning_rate(1), 2e-7)
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(5000), 1e-5)

    def test_absolute_reload_both_markers_reject_relative_only_and_nonfinite(self):
        for update in (1, 2):
            marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                          reload_max_action_error=0., model_sha256='hash')
            self.c.require_marker(marker, update)
            for error in (2e-5, -.1, float('nan'), float('inf'), None, True):
                with self.subTest(update=update, error=error), self.assertRaises(ValueError):
                    self.c.require_marker({**marker, 'reload_max_action_error': error}, update)
            with self.assertRaises(ValueError): self.c.require_marker({**marker, 'reload_fp32_weights_exact': False}, update)

    def test_real_adam_two_updates_and_roundtrip_keep_fp32(self):
        model = torch.nn.Linear(2, 1).bfloat16()
        self.c.h.promote_trainable_parameters(model)
        optimizer = self.r.make_optimizer(list(model.parameters()))
        for update in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(4): self.c.h.backward_microbatch(model(torch.ones(8, 2)).square().mean())
            optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            torch.save({'optimizer': optimizer.state_dict(), 'update': 2}, path)
            evidence = self.r.adam_roundtrip(optimizer, path, 2)
            self.assertTrue(evidence['adam_reload_fp32_exact'])
            self.assertEqual(evidence['update'], 2)
            state = torch.load(path, weights_only=False)
            first = next(iter(state['optimizer']['state'].values()))
            first['exp_avg'] = first['exp_avg'].bfloat16()
            torch.save(state, path)
            with self.assertRaises(ValueError): self.r.adam_roundtrip(optimizer, path, 2)

    def test_fp32_loader_promotes_before_strict_copy(self):
        from types import SimpleNamespace
        from safetensors.torch import save_model
        from validation.precision import load_fp32_checkpoint
        class Tiny(torch.nn.Module):
            def __init__(self, config=None):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        with tempfile.TemporaryDirectory() as tmp:
            model = Tiny(); self.c.h.promote_trainable_parameters(model)
            with torch.no_grad(): model.weight.fill_(1.000123)
            save_model(model, str(Path(tmp)/'model.safetensors'))
            with patch('lerobot.configs.policies.PreTrainedConfig.from_pretrained', return_value=SimpleNamespace(device='cpu')), \
                    patch('lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy', Tiny):
                restored = load_fp32_checkpoint(tmp)
            self.assertEqual(restored.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(model.weight, restored.weight))

    def test_prior_save_checkpoint_retains_training_state_and_fp32(self):
        from types import SimpleNamespace
        from safetensors.torch import save_model
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([1.000123]))
                self.config = SimpleNamespace(device='cpu')
            def predict_action_chunk(self, batch): return self.weight.expand(1, 50, 8)
            def save_pretrained(self, directory): save_model(self, str(directory/'model.safetensors'))
        class Processor:
            def __init__(self, name): self.name = name
            def __call__(self, value): return value
            def save_pretrained(self, directory): (directory/self.name).write_text('{}')
        model = Tiny(); optimizer = self.r.make_optimizer(list(model.parameters()))
        model.weight.grad = torch.ones_like(model.weight); optimizer.step()
        pre, post = Processor('policy_preprocessor.json'), Processor('policy_postprocessor.json')
        def load(path):
            from safetensors.torch import load_model
            new = Tiny(); load_model(new, str(path/'model.safetensors'), strict=True); return new
        generator = torch.Generator().manual_seed(1)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(torch.cuda, 'current_device', return_value=0), \
                patch.object(torch.random, 'fork_rng', side_effect=lambda **kw: nullcontext()), \
                patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                patch.object(self.c.h, 'load_fp32_checkpoint', side_effect=load), \
                patch('lerobot.policies.factory.make_pre_post_processors', return_value=(pre, post)):
            marker = self.c.h.save_checkpoint(model, pre, post, optimizer, [frame()], 0, Path(tmp),
                                               1, generator, generator.get_state(), 0, 4)
            self.c.require_marker(marker, 1)
            state = torch.load(Path(tmp)/'checkpoints/000001/training_state.pt', weights_only=False)
            self.assertFalse(state['resume_supported'])
            self.assertEqual(state['batches_in_epoch'], 4)
            self.assertIn('epoch_generator_state', state)
            self.assertTrue(model.training)

    def scratch_fixture(self, root):
        c = self.c
        stats = {'action': {'mean': [0.]*8}}
        c.write_json(root/'training_stats.json', stats)
        markers = []
        for update in (1, 2):
            folder = root/f'checkpoints/{update:06d}'; folder.mkdir(parents=True)
            (folder/'model.safetensors').write_bytes(b'weights')
            for name in ('training_state.pt', 'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                (folder/name).write_text('{}')
            marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                          reload_max_action_error=0., model_sha256=c.sha256(folder/'model.safetensors'))
            c.write_json(folder/'checkpoint_result.json', marker)
            c.write_json(folder/'adam_reload.json', dict(adam_reload_fp32_exact=True, update=update,
                         training_state_sha256=c.sha256(folder/'training_state.pt')))
            markers.append(marker)
        binding = {'data': 'sealed'}
        report = {**c.recipe(True), 'status': 'complete', 'completed_updates': 2, 'binding': binding,
                  'cpu_admission_sha256': 'cpu', 'training_rows': 10560, 'complete_training_chunks': 9580,
                  'training_stats_sha256': c.sha256(root/'training_stats.json'), 'checkpoints': markers,
                  'checkpoint_files': {f'{i:06d}': c.inventory(root/f'checkpoints/{i:06d}') for i in (1, 2)},
                  'final_data_binding_passed': True, 'adam_reload_fp32_exact': True,
                  'selected_checkpoint_update': None, 'rollouts': 0}
        return binding, stats, report

    def test_scratch_admission_all_checkpoint_files_stats_and_cpu_bound(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binding, stats, report = self.scratch_fixture(root)
            c.validate_scratch(report, binding, root, stats, 'cpu')
            for change in ({'completed_updates': 1}, {'cpu_admission_sha256': 'stale'}, {'binding': {}},
                           {'adam_reload_fp32_exact': False}, {'training_stats_sha256': 'changed'},
                           {'rollouts': 1}, {'checkpoint_files': {}}, {'microbatch': 4}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_scratch({**report, **change}, binding, root, stats, 'cpu')
            for i in (0, 1):
                bad = copy.deepcopy(report); bad['checkpoints'][i]['reload_max_action_error'] = .001
                with self.assertRaises(ValueError): c.validate_scratch(bad, binding, root, stats, 'cpu')
            (root/'checkpoints/000001/policy_preprocessor.json').write_text('changed')
            with self.assertRaises(ValueError): c.validate_scratch(report, binding, root, stats, 'cpu')

    def test_scratch_requires_real_adam_evidence_in_both_bound_markers(self):
        c = self.c
        for update in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); binding, stats, report = self.scratch_fixture(root)
                c.write_json(root/f'checkpoints/{update:06d}/adam_reload.json',
                             dict(adam_reload_fp32_exact=False, update=update, training_state_sha256='wrong'))
                # Even a consistently rebound inventory cannot stand in for actual PASS evidence.
                report['checkpoint_files'][f'{update:06d}'] = c.inventory(root/f'checkpoints/{update:06d}')
                with self.assertRaises(ValueError): c.validate_scratch(report, binding, root, stats, 'cpu')

    def test_exact_two_update_runner_with_real_cpu_adam_and_prior_saver(self):
        from types import SimpleNamespace
        from safetensors.torch import save_model, load_model
        from test_contract import metadata
        r, c = self.r, self.c
        self.assertTrue(hasattr(r, 'load_fresh_policy'), 'isolate original model load with architecture guard')
        class Dataset:
            meta = SimpleNamespace(episodes=metadata())
            sample = frame()
            def __getitem__(self, index): return {**self.sample, 'episode_index': torch.tensor(index//528)}
        class Config:
            def save_pretrained(self, path):
                path.mkdir(exist_ok=False)
                (path/'config.json').write_text('{}')
        config = Config()
        calls = []
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.config = config
                self.action_out_proj = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad(): self.action_out_proj.weight.fill_(1.)
                self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            def forward(self, batch):
                calls.append(tuple(batch['action'].shape))
                return self.action_out_proj.weight.square().mean(), {}
            def predict_action_chunk(self, batch): return self.action_out_proj.weight.expand(1, 50, 8)
            def save_pretrained(self, path):
                save_model(self, str(path/'model.safetensors')); (path/'config.json').write_text('{}')
        class Processor:
            def __init__(self, name): self.name = name
            def __call__(self, x): return x
            def save_pretrained(self, path): (path/self.name).write_text('{}')
        pre, post = Processor('policy_preprocessor.json'), Processor('policy_postprocessor.json')
        def reload(path):
            model = Tiny(); load_model(model, str(path/'model.safetensors'), strict=True); return model
        model, binding, stats = Tiny(), {'bound': 'data'}, {'action': {'mean': torch.zeros(8)}}
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); cpu = root/'cpu.json'; cpu.write_text('{}')
            out = root/'scratch'; out.mkdir()
            patches = [patch.object(c, 'CPU_ADMISSION', cpu),
                       patch.object(c, 'prepare_data', return_value=(Dataset(), c.indices(metadata())[1], stats, binding)),
                       patch.object(c, 'admission', return_value=binding), patch.object(c, 'make_config', return_value=config),
                       patch.object(r, 'load_fresh_policy', return_value=(model, {'trainable_parameters': 99880992})),
                       patch.object(r, 'require_resources', return_value={}), patch.object(r, 'require_disk'),
                       patch.object(torch.cuda, 'is_available', return_value=True),
                       patch.object(torch.cuda, 'is_bf16_supported', return_value=True),
                       patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
                       patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'max_memory_allocated', return_value=0),
                       patch.object(torch.cuda, 'current_device', return_value=0),
                       patch.object(torch.cuda, 'get_rng_state_all', return_value=[]),
                       patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
                       patch.object(torch.random, 'fork_rng', side_effect=lambda **k: nullcontext()),
                       patch.object(c.h, 'load_fp32_checkpoint', side_effect=reload),
                       patch('lerobot.policies.factory.make_pre_post_processors', return_value=(pre, post))]
            for item in patches: stack.enter_context(item)
            report = c.recipe(True)
            r.train(out, True, report)
            self.assertEqual(calls, [(8, 50, 8)]*8)
            self.assertEqual(report['completed_updates'], 2)
            self.assertEqual(report['last_metrics']['sampled_chunks'], 64)
            self.assertEqual([m['update'] for m in report['checkpoints']], [1, 2])
            c.validate_scratch(report, binding, out, c.statistics_json(stats), c.sha256(cpu))
            for update in (1, 2):
                state = torch.load(out/f'checkpoints/{update:06d}/training_state.pt', weights_only=False)
                self.assertEqual(state['batches_in_epoch'], update*4)
                self.assertEqual(next(iter(state['optimizer']['state'].values()))['step'].item(), update)
            self.assertLess(model.action_out_proj.weight.item(), 1.)
            self.assertIsNone(model.frozen.grad)

    def test_final_only_report_and_all_six_inventories_gate(self):
        from experiments.smolvla_fixed_layout import evaluate as e
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scratch = root/'scratch'; scratch.mkdir()
            binding, stats, proof = self.scratch_fixture(scratch)
            cpu = root/'cpu.json'; c.write_json(cpu, {'training_stats': stats})
            proof['cpu_admission_sha256'] = c.sha256(cpu); c.write_json(scratch/'result.json', proof)
            out = root/'main'; out.mkdir(); c.write_json(out/'training_stats.json', stats)
            config = dict(type='smolvla', chunk_size=50, n_action_steps=10, n_obs_steps=1,
                          input_features={k: {'shape': list(v)} for k, v in c.OBS.items()},
                          output_features={'action': {'shape': [8]}})
            (out/'initial_config').mkdir(); c.write_json(out/'initial_config/config.json', config)
            markers, files = [], {}
            for update in (1, 1000, 2000, 3000, 4000, 5000):
                path = out/f'checkpoints/{update:06d}'; path.mkdir(parents=True)
                for name in ('model.safetensors', 'training_state.pt', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                    (path/name).write_text('synthetic')
                c.write_json(path/'config.json', config)
                marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                              reload_max_action_error=0., model_sha256=c.sha256(path/'model.safetensors'))
                c.write_json(path/'checkpoint_result.json', marker)
                markers.append(marker); files[f'{update:06d}'] = c.inventory(path)
            report = {**c.recipe(False), 'status': 'complete', 'completed_updates': 5000, 'binding': binding,
                      'selected_checkpoint_update': 5000, 'final_data_binding_passed': True,
                      'training_rows': 10560, 'complete_training_chunks': 9580, 'scratch_used_for_initialization': False,
                      'cpu_admission_sha256': c.sha256(cpu), 'training_stats_sha256': c.sha256(out/'training_stats.json'),
                      'scratch_result_sha256': c.sha256(scratch/'result.json'), 'checkpoints': markers, 'checkpoint_files': files}
            with patch.object(c, 'CPU_ADMISSION', cpu), patch.object(c, 'SCRATCH', scratch):
                c.write_json(out/'result.json', report)
                self.assertEqual(e.verify_final(out, binding), out/'checkpoints/005000')
                for change in ({'selected_checkpoint_update': 1000}, {'completed_updates': 4999},
                               {'cpu_admission_sha256': 'old'}, {'scratch_used_for_initialization': True},
                               {'checkpoint_files': {}}, {'training_stats_sha256': 'old'}):
                    c.write_json(out/'result.json', {**report, **change})
                    with self.subTest(change=change), self.assertRaises(ValueError): e.verify_final(out, binding)

    def test_dispatch_scratch_no_eval_and_failure_preserves_artifact(self):
        r, c = self.r, self.c
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'SCRATCH', Path(tmp)/'scratch'), \
                patch.object(r, 'train', side_effect=ValueError('synthetic failure')), \
                patch('experiments.smolvla_fixed_layout.evaluate.run', side_effect=AssertionError('scratch eval')):
            with self.assertRaises(ValueError): r.main(['--preflight'])
            self.assertEqual(json.loads((c.SCRATCH/'failure.json').read_text())['status'], 'failed')
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'SCRATCH', Path(tmp)/'scratch'), \
                patch.object(r, 'train'), \
                patch('experiments.smolvla_fixed_layout.evaluate.run', side_effect=AssertionError('scratch eval')):
            r.main(['--preflight'])

    def test_cli_no_default_gpu_resume_override_or_abbreviation(self):
        for argv in ([], ['--resume'], ['--train', '--preflight'], ['--batch-size', '4'], ['--tr']):
            with self.assertRaises(SystemExit): self.r.parse_args(argv)
        self.assertTrue(self.r.parse_args(['--check']).check)

    def test_resources_fail_below_35_disk_or_12_gpu(self):
        from types import SimpleNamespace
        for disk, gpu in ((34, 40), (40, 11)):
            with patch.object(self.r.shutil, 'disk_usage', return_value=SimpleNamespace(free=disk*1024**3)), \
                    patch.object(torch.cuda, 'mem_get_info', return_value=(gpu*1024**3, 48*1024**3)):
                with self.assertRaises(RuntimeError): self.r.require_resources(Path('/tmp'))


if __name__ == '__main__': unittest.main()