"""Synthetic CPU models; no pretrained model, CUDA or real training (X-VLA local-varied)."""
from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_contract import frame, metadata


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'runner.py').is_file(),
                        'X-VLA local-varied runner missing')
        self.r = importlib.import_module('experiments.xvla_local_varied.runner')
        self.c = self.r.c

    def test_cli_no_default_gpu_resume_override_or_abbreviation(self):
        for argv in ([], ['--resume'], ['--train', '--preflight'], ['--batch-size', '4'],
                     ['--tr'], ['--check', '--preflight'], ['--evaluate'], ['--train', '--steps', '1']):
            with self.assertRaises(SystemExit):
                self.r.parse_args(argv)
        self.assertTrue(self.r.parse_args(['--check']).check)
        self.assertTrue(self.r.parse_args(['--preflight']).preflight)
        self.assertTrue(self.r.parse_args(['--train']).train)

    def test_resources_fail_below_preflight_15_and_train_60_disk_or_28_gpu(self):
        from types import SimpleNamespace
        # Preflight: 15 GiB disk, 28 GiB GPU
        for disk, gpu, preflight, passes in ((14, 40, True, False), (15, 40, True, True),
                                             (15, 27, True, False),
                                             (59, 40, False, False), (60, 40, False, True),
                                             (60, 27, False, False)):
            with patch.object(self.r.shutil, 'disk_usage',
                              return_value=SimpleNamespace(free=disk*1024**3)), \
                    patch.object(torch.cuda, 'mem_get_info',
                                 return_value=(gpu*1024**3, 48*1024**3)):
                if passes:
                    self.r.require_resources(Path('/tmp'), preflight)
                else:
                    with self.assertRaises(RuntimeError):
                        self.r.require_resources(Path('/tmp'), preflight)

    def test_processor_check_uses_real_processor_no_model_construction(self):
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        rows = {i: frame(0) for i in (0, 498, 499, 10530)}
        evidence = self.r.processor_check(rows, [0, 498, 499, 10530], stats)
        self.assertEqual(evidence['image_shape'], [4, 3, 3, 224, 224])
        self.assertTrue(evidence['physical_action_roundtrip'])
        self.assertFalse(evidence['whole_policy_constructed'])
        self.assertFalse(evidence['backward_executed'])

    def test_absolute_reload_marker_reject_relative_only_and_nonfinite(self):
        for update in (1, 2, 5000):
            marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                          reload_max_action_error=0., model_sha256='hash')
            self.c.require_marker(marker, update)
            for error in (2e-5, -.1, float('nan'), float('inf'), None, True):
                with self.subTest(update=update, error=error), self.assertRaises(ValueError):
                    self.c.require_marker({**marker, 'reload_max_action_error': error}, update)
            with self.assertRaises(ValueError):
                self.c.require_marker({**marker, 'reload_fp32_weights_exact': False}, update)

    def test_real_adam_two_updates_and_roundtrip_keep_fp32(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW([{'params': list(model.parameters()), 'name': 'other',
                                  'lr': 1e-4, 'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 1e-4}])
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            model(torch.ones(4, 2)).square().mean().backward()
            opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'training_state.pt'
            torch.save(opt.state_dict(), path)
            evidence = self.r.adam_roundtrip(opt, path, 2)
            self.assertTrue(evidence['adam_reload_fp32_exact'])
            self.assertEqual(evidence['update'], 2)
            # Tamper: change dtype of exp_avg → mismatch.
            saved = torch.load(path, weights_only=False)
            first = next(iter(saved['state'].values()))
            first['exp_avg'] = first['exp_avg'].bfloat16()
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                self.r.adam_roundtrip(opt, path, 2)

    def test_load_fresh_policy_requires_expected_parameter_budget(self):
        # Mock XVLAPolicy + load_weights so no real 3.5GB safetensors read happens.
        c = self.c
        config = object()
        good_params = {f'p{i}': torch.zeros(1) for i in range(901)}
        # Not exactly 879482456 numel, so overall check fails unless we monkeypatch that check too.
        with patch('lerobot.policies.xvla.modeling_xvla.XVLAPolicy', lambda cfg: object()), \
                patch.object(c.p, 'load_weights') as lw:
            # Case 1: policy exposes exactly 901 params/879482456 numel → passes.
            class GoodPolicy:
                def __init__(self):
                    self._np = {f'p{i}': torch.nn.Parameter(torch.zeros(879482456//901+(1 if i < 879482456%901 else 0)))
                                for i in range(901)}
                def named_parameters(self): return list(self._np.items())
                def get_optim_params(self): return dict(self._np)
            good = GoodPolicy()
            lw.return_value = good
            policy, params = self.r.load_fresh_policy(config)
            self.assertIs(policy, good)
            self.assertEqual(len(params), 901)
            # Case 2: fewer params → rejection.
            class BadPolicy(GoodPolicy):
                def __init__(self):
                    self._np = {f'p{i}': torch.nn.Parameter(torch.zeros(1)) for i in range(900)}
            lw.return_value = BadPolicy()
            with self.assertRaises(ValueError):
                self.r.load_fresh_policy(config)

    def _synthetic_checkpoint(self, root, updates, final_update, preflight):
        c = self.c
        markers, files = [], {}
        for update in updates:
            folder = root/'checkpoints'/c.checkpoint_directory(update)
            folder.mkdir(parents=True)
            for name in ('model.safetensors', 'config.json',
                         'policy_preprocessor.json', 'policy_postprocessor.json'):
                (folder/name).write_bytes(b'bytes')
            if update == final_update:
                (folder/'training_state.pt').write_bytes(b'opt')
            marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                          reload_max_action_error=0.,
                          model_sha256=c.sha256(folder/'model.safetensors'))
            c.write_json(folder/'checkpoint_result.json', marker)
            if preflight and update == final_update:
                c.write_json(folder/'adam_reload.json',
                             dict(adam_reload_fp32_exact=True, update=update,
                                  training_state_sha256=c.sha256(folder/'training_state.pt')))
            markers.append(marker)
            files[c.checkpoint_directory(update)] = c.inventory(folder)
        return markers, files

    def test_scratch_admission_all_checkpoint_files_stats_and_cpu_bound(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = {'action': {'mean': [0.]*8}}
            c.write_json(root/'training_stats.json', stats)
            markers, files = self._synthetic_checkpoint(root, [1, 2], 2, True)
            binding = {'data': 'sealed'}
            report = {**c.recipe(True), 'status': 'complete', 'completed_updates': 2,
                      'binding': binding, 'training_rows': 25344, 'complete_training_chunks': 23952,
                      'cpu_admission_sha256': 'cpu', 'final_data_binding_passed': True,
                      'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None,
                      'rollouts': 0, 'checkpoints': markers, 'checkpoint_files': files,
                      'training_stats_sha256': c.sha256(root/'training_stats.json')}
            c.validate_scratch(report, binding, root, stats, 'cpu')
            for change in ({'completed_updates': 1}, {'cpu_admission_sha256': 'stale'},
                           {'binding': {}}, {'adam_reload_fp32_exact': False},
                           {'training_stats_sha256': 'changed'}, {'rollouts': 1},
                           {'checkpoint_files': {}}, {'training_rows': 10560}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_scratch({**report, **change}, binding, root, stats, 'cpu')
            # Legacy naming regression must fail.
            (root/'checkpoints/000001').mkdir()
            with self.assertRaises(ValueError):
                c.validate_scratch(report, binding, root, stats, 'cpu')

    def test_final_five_thousand_report_and_all_six_step_inventories_gate(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = {'action': {'mean': [0.]*8}}
            c.write_json(root/'training_stats.json', stats)
            updates = [1, 1000, 2000, 3000, 4000, 5000]
            markers, files = self._synthetic_checkpoint(root, updates, 5000, False)
            report = dict(checkpoints=markers, checkpoint_files=files)
            c.verify_checkpoints(report, root, False)
            # Missing marker → rejection.
            bad = dict(checkpoints=markers[:-1], checkpoint_files=files)
            with self.assertRaises(ValueError):
                c.verify_checkpoints(bad, root, False)
            # Legacy 6-digit dir alongside step_005000 → rejection.
            (root/'checkpoints/005000').mkdir()
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, root, False)

    def test_train_two_updates_with_synthetic_policy_writes_step_prefixed_checkpoints(self):
        from types import SimpleNamespace
        from safetensors.torch import save_model, load_model
        c, r = self.c, self.r

        class Config:
            device = 'cuda'
            dtype = 'float32'
            chunk_size = 30
            action_feature = SimpleNamespace(shape=(8,))
            def save_pretrained(self, path):
                path.mkdir(exist_ok=True)
                (path/'config.json').write_text(json.dumps({'dtype': 'float32'}))
            def get_optimizer_preset(self):
                class Preset:
                    def build(inner_self, params):
                        return torch.optim.AdamW(
                            [{'params': list(params.values()), 'name': 'other',
                              'lr': 1e-4, 'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 1e-4}])
                return Preset()

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.action_out_proj = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.action_out_proj.weight.fill_(1.)
            def forward(self, batch):
                return self.action_out_proj.weight.square().mean(), {}
            def predict_action_chunk(self, batch):
                return self.action_out_proj.weight.expand(1, 30, 8)
            def save_pretrained(self, path):
                save_model(self, str(path/'model.safetensors'))
                self.config.save_pretrained(path)
            def get_optim_params(self):
                return dict(self.named_parameters())
            def to(self, device):
                return self

        class Processor:
            def __init__(self, name):
                self.name = name
            def __call__(self, value):
                return value
            def save_pretrained(self, path):
                (path/self.name).write_text('{}')

        pre, post = Processor('policy_preprocessor.json'), Processor('policy_postprocessor.json')
        stats = {'action': {'mean': torch.zeros(8)}}

        class Dataset:
            meta = SimpleNamespace(episodes=metadata())
            sample = frame()
            def __getitem__(self, index):
                return {**self.sample, 'episode_index': torch.tensor(index//528)}

        ds = Dataset()
        raw_starts = c.indices(metadata())[1]

        def _reload(checkpoint, expected):
            # Bytewise safetensors verify + prediction equal (synthetic; no config parse).
            new = Tiny()
            load_model(new, str(checkpoint/'model.safetensors'), strict=True)
            new.eval()
            with torch.inference_mode():
                actual = new.predict_action_chunk(None).cpu()
            if not torch.equal(actual, expected['prediction']):
                raise ValueError('reload mismatch')
            return 0.

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            out = root/'scratch'
            out.mkdir()
            cpu = root/'cpu.json'
            cpu.write_text('cpu')
            binding = {'bound': 'data'}
            tiny = Tiny()
            patches = [
                patch.object(c, 'CPU_ADMISSION', cpu),
                patch.object(c, 'prepare_data',
                             return_value=(ds, raw_starts, stats, binding)),
                patch.object(c, 'admission', return_value=binding),
                patch.object(c, 'make_config', return_value=Config()),
                patch.object(r, 'load_fresh_policy',
                             return_value=(tiny, tiny.get_optim_params())),
                patch.object(r, 'EXPECTED_PARAM_TENSORS', 1),
                patch.object(r, 'require_resources', return_value={}),
                patch.object(r, 'require_disk'),
                patch.object(c, 'verify_checkpoints'),
                patch.object(c, 'check_model_inputs',
                             return_value={'checked': True}),
                patch.object(c, 'set_learning_rate',
                             side_effect=lambda opt, u: [g.__setitem__('lr', 1e-4) for g in opt.param_groups]),
                patch.object(c, 'collate_rows',
                             side_effect=lambda rows: {**frame(), 'action': torch.zeros(4, 30, 8)}),
                patch.object(c, 'backward_microbatch',
                             side_effect=lambda loss: (loss/2).backward()),
                patch.object(c, 'observation', side_effect=lambda x: dict(x)),
                patch.object(c, 'sample_batches',
                             side_effect=lambda starts: iter(lambda: [starts[0]]*4, object())),
                patch.object(c, 'inventory',
                             side_effect=lambda directory: {p.name: 'x' for p in Path(directory).iterdir()}),
                patch.object(c.p, 'verify_precision'),
                patch.object(c.p, 'load_weights', side_effect=lambda pol, path: pol),
                patch.object(c.p, 'memory', return_value={}),
                patch.object(c.p, 'process_row', side_effect=lambda pre, obs: obs),
                patch.object(r, '_reload_and_verify', side_effect=_reload),
                patch.object(torch.cuda, 'is_available', return_value=True),
                patch.object(torch.cuda, 'is_bf16_supported', return_value=True),
                patch.object(torch.cuda, 'reset_peak_memory_stats'),
                patch.object(torch.cuda, 'synchronize'),
                patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
                patch.object(torch.cuda, 'get_device_name', return_value='fake'),
                patch.object(torch.cuda, 'get_device_properties', return_value=object()),
                patch.object(torch.cuda, 'mem_get_info',
                             return_value=(28*1024**3, 48*1024**3)),
                patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
                patch.object(torch.nn.utils, 'clip_grad_norm_',
                             side_effect=lambda ps, m, error_if_nonfinite=True: torch.tensor(0.5)),
                patch('lerobot.policies.xvla.processor_xvla.make_xvla_pre_post_processors',
                      return_value=(pre, post)),
                patch.object(torch, 'use_deterministic_algorithms'),
            ]
            for item in patches:
                stack.enter_context(item)
            report = {**c.recipe(True), 'binding': None}
            r.train(out, True, report)
            self.assertEqual(report['completed_updates'], 2)
            self.assertEqual([m['update'] for m in report['checkpoints']], [1, 2])
            for update in (1, 2):
                key = c.checkpoint_directory(update)
                self.assertTrue((out/'checkpoints'/key).is_dir())
                self.assertFalse((out/f'checkpoints/{update:06d}').exists())
            # Only FINAL update carries training_state.pt.
            self.assertFalse((out/'checkpoints/step_000001/training_state.pt').exists())
            self.assertTrue((out/'checkpoints/step_000002/training_state.pt').exists())
            # Adam reload evidence written for preflight final.
            self.assertTrue((out/'checkpoints/step_000002/adam_reload.json').is_file())

    def test_dispatch_does_not_launch_evaluation(self):
        r, c = self.r, self.c
        # No evaluate submodule in the X-VLA local-varied package; main() must never look for one.
        self.assertFalse((Path(__file__).parents[1]/'evaluate.py').exists())
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'SCRATCH', Path(tmp)/'scratch'), \
                patch.object(r, 'train', side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):
                r.main(['--preflight'])
            self.assertEqual(json.loads((c.SCRATCH/'failure.json').read_text())['status'], 'failed')
        # A pass-through --train must not invoke any evaluate.run.
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUTPUT', Path(tmp)/'main'), \
                patch.object(r, 'train'):
            r.main(['--train'])
            # No evaluate module exists; main() must complete without raising ImportError.
            self.assertTrue((Path(tmp)/'main/run_claim.json').is_file())


if __name__ == '__main__':
    unittest.main()
