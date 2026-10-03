"""Synthetic CPU models; no pretrained model, CUDA or real training (X-VLA B3 robosuite)."""
from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_contract import frame, metadata


class SynConfig:
    device = 'cuda'
    dtype = 'float32'
    chunk_size = 30

    def __init__(self):
        from types import SimpleNamespace
        self.action_feature = SimpleNamespace(shape=(8,))

    def save_pretrained(self, path):
        path.mkdir(exist_ok=True)
        (path/'config.json').write_text(json.dumps({'dtype': 'float32'}))

    def get_optimizer_preset(self):
        class Preset:
            def build(self, params):
                return torch.optim.AdamW([{'params': list(params.values()), 'name': 'other', 'lr': 1e-4,
                                           'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 1e-4}])
        return Preset()


class SynTiny(torch.nn.Module):
    """One weight; the loss draws from the global RNG, so a restored run only matches if the RNG is."""

    def __init__(self):
        super().__init__()
        self.config = SynConfig()
        self.action_out_proj = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.action_out_proj.weight.fill_(1.)

    def forward(self, batch):
        return (self.action_out_proj.weight*torch.randn(1)).square().mean(), {}

    def predict_action_chunk(self, batch):
        return self.action_out_proj.weight.expand(1, 30, 8)

    def save_pretrained(self, path):
        from safetensors.torch import save_model
        save_model(self, str(path/'model.safetensors'))
        self.config.save_pretrained(path)

    def get_optim_params(self):
        return dict(self.named_parameters())

    def to(self, device):
        return self


class SynProcessor:
    def __init__(self, name):
        self.name = name

    def __call__(self, value):
        return value

    def save_pretrained(self, path):
        (path/self.name).write_text('{}')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'runner.py').is_file(),
                        'X-VLA B3 robosuite runner missing')
        self.r = importlib.import_module('experiments.xvla_b3_robosuite.runner')
        self.c = self.r.c

    def test_runner_horizon_is_taken_from_the_contract_not_hardcoded(self):
        """Regression class from FAILURES #15: a runner hardcoding its own horizon silently
        truncated a longer run and only failed afterwards, on `verify_checkpoints`."""
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertNotIn('else 5000', source, 'runner must not hardcode the training horizon')
        self.assertIn("c.recipe(False)['max_optimizer_updates']", source)
        self.assertEqual(self.r.c.recipe(False)['max_optimizer_updates'],
                         self.r.c.checkpoint_updates(False)[-1],
                         'final checkpoint must coincide with the training horizon')
        self.assertEqual(self.r.c.recipe(False)['schedule']['total'],
                         self.r.c.recipe(False)['max_optimizer_updates'],
                         'LR schedule horizon must match the training horizon')

    def test_cli_no_default_gpu_resume_override_or_abbreviation(self):
        # --resume continues only --train: never alone, never with a gate, never abbreviated.
        for argv in ([], ['--resume'], ['--train', '--preflight'], ['--batch-size', '4'],
                     ['--tr'], ['--check', '--preflight'], ['--evaluate'], ['--train', '--steps', '1'],
                     ['--check', '--resume'], ['--preflight', '--resume'], ['--resume-proof', '--resume'],
                     ['--train', '--resume-proof'], ['--train', '--res'], ['--resume-p']):
            with self.assertRaises(SystemExit):
                self.r.parse_args(argv)
        self.assertTrue(self.r.parse_args(['--check']).check)
        self.assertTrue(self.r.parse_args(['--preflight']).preflight)
        self.assertTrue(self.r.parse_args(['--train']).train)
        self.assertFalse(self.r.parse_args(['--train']).resume)
        self.assertTrue(self.r.parse_args(['--train', '--resume']).resume)
        self.assertTrue(self.r.parse_args(['--resume-proof']).resume_proof)

    def test_resources_fail_below_preflight_28_and_train_60_disk_or_28_gpu(self):
        from types import SimpleNamespace
        # Preflight / resume proof: 28 GiB disk. Train: 60 GiB disk (6 checkpoints + 3 snapshots).
        # Both need 28 GiB of free GPU memory.
        for disk, gpu, preflight, passes in ((27, 40, True, False), (28, 40, True, True),
                                             (28, 27, True, False),
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

    def test_training_disk_guard_covers_checkpoints_and_the_snapshot_working_set(self):
        """6 checkpoints (3.3 GB each, 9.9 GB final) + KEEP_SNAPSHOTS kept + 1 being written (~10.5 GB)."""
        from types import SimpleNamespace
        updates = self.r.c.checkpoint_updates(False)
        self.assertEqual(len(updates), 6)
        needed_gb = 3.3*(len(updates)-1) + 9.9 + (self.r.c.KEEP_SNAPSHOTS+1)*10.5
        self.assertLess(needed_gb*1e9, 60*1024**3, 'the 60 GiB guard must cover the working set')
        with patch.object(self.r.shutil, 'disk_usage',
                          return_value=SimpleNamespace(free=int(needed_gb*1e9))), \
                patch.object(torch.cuda, 'mem_get_info', return_value=(40*1024**3, 48*1024**3)):
            with self.assertRaises(RuntimeError):  # free space equal to the bare need is not enough
                self.r.require_resources(Path('/tmp'), False)

    def test_processor_check_uses_real_processor_no_model_construction(self):
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        rows = {i: frame(0) for i in (0, 498, 499, 249_499)}
        evidence = self.r.processor_check(rows, [0, 498, 499, 249_499], stats)
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
                      'binding': binding, 'training_rows': c.TRAINING_ROWS,
                      'complete_training_chunks': c.COMPLETE_STARTS,
                      'cpu_admission_sha256': 'cpu', 'final_data_binding_passed': True,
                      'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None,
                      'rollouts': 0, 'checkpoints': markers, 'checkpoint_files': files,
                      'training_stats_sha256': c.sha256(root/'training_stats.json')}
            c.validate_scratch(report, binding, root, stats, 'cpu')
            for change in ({'completed_updates': 1}, {'cpu_admission_sha256': 'stale'},
                           {'binding': {}}, {'adam_reload_fp32_exact': False},
                           {'training_stats_sha256': 'changed'}, {'rollouts': 1},
                           {'checkpoint_files': {}}, {'training_rows': 84_480}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_scratch({**report, **change}, binding, root, stats, 'cpu')
            # Legacy naming regression must fail.
            (root/'checkpoints/000001').mkdir()
            with self.assertRaises(ValueError):
                c.validate_scratch(report, binding, root, stats, 'cpu')

    def test_final_report_and_all_step_inventories_gate(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = {'action': {'mean': [0.]*8}}
            c.write_json(root/'training_stats.json', stats)
            updates = c.checkpoint_updates(False)
            markers, files = self._synthetic_checkpoint(root, updates, updates[-1], False)
            report = dict(checkpoints=markers, checkpoint_files=files)
            c.verify_checkpoints(report, root, False)
            # Missing marker → rejection.
            bad = dict(checkpoints=markers[:-1], checkpoint_files=files)
            with self.assertRaises(ValueError):
                c.verify_checkpoints(bad, root, False)
            # Legacy 6-digit dir alongside the final step_ dir → rejection.
            (root/f'checkpoints/{updates[-1]:06d}').mkdir()
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

    def test_check_recomputes_the_parent_fingerprint_before_anything_else(self):
        r, c = self.r, self.c
        with tempfile.TemporaryDirectory() as tmp:
            admission = Path(tmp)/'admission'
            with patch.object(c, 'ADMISSION', admission), \
                    patch.object(c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(c, 'parent_fingerprint', return_value='0'*64), \
                    patch.object(c, 'current_binding', side_effect=AssertionError('binding before fingerprint')), \
                    patch.object(c, 'load_data', side_effect=AssertionError('data before fingerprint')):
                with self.assertRaisesRegex(ValueError, 'B3 parent fingerprint differs from the seal'):
                    r.check()
                report = json.loads((admission/'result.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertNotEqual(report.get('status'), 'passed')
                self.assertTrue((admission/'failure.json').is_file())

    def test_check_does_not_call_a_simulator_audit(self):
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertNotIn('require_broad_workspace_pass', source)
        self.assertIn('c.parent_fingerprint()', source)
        self.assertIn('parent_fingerprint_recomputed', (Path(__file__).parents[1]/'contract.py').read_text())

    def test_dispatch_does_not_launch_evaluation(self):
        r, c = self.r, self.c
        # No evaluate submodule in this package; main() must never look for one.
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
    # --- resume ---------------------------------------------------------------------------------
    # A 4-update synthetic run: checkpoints at 1, 3 and 4 (4 is final), snapshots as given. Checkpoint 3
    # lies past a snapshot at 2, and update 2 is not a checkpoint update, so the RNG at snapshot 2 is
    # mid-stream (a checkpoint reseeds it) and the restore is actually exercised.

    def _synthetic_run(self, root, out, resume=0, crash_at=None, snapshots=(2,), binding=None):
        from safetensors.torch import load_model
        c, r = self.c, self.r
        cpu = root/'cpu.json'
        cpu.write_text('cpu')
        for name in ('scratch', 'proof'):
            (root/name).mkdir(exist_ok=True)
            (root/name/'result.json').write_text('{}')
        (root/'scratch/training_stats.json').write_text(json.dumps({'action': {'mean': [0.]*8}}))
        binding = binding or {'bound': 'data'}
        tiny, config = SynTiny(), SynConfig()
        ds = type('Dataset', (), {
            'meta': type('Meta', (), {'episodes': metadata()})(), 'sample': frame(),
            '__getitem__': lambda self, index: {**self.sample, 'episode_index': torch.tensor(index//528)}})()
        starts = c.indices(metadata())[1]
        real_recipe = c.recipe

        def reload(checkpoint, expected):
            new = SynTiny()
            load_model(new, str(checkpoint/'model.safetensors'), strict=True)
            if not torch.equal(new.predict_action_chunk(None).detach(), expected['prediction']):
                raise ValueError('reload mismatch')
            return 0.

        def batches(starts):
            number = 0
            while True:
                number += 1
                yield [starts[number]]*4

        def rate(optimizer, update):
            if update == crash_at:
                raise RuntimeError('synthetic crash')
            for group in optimizer.param_groups:
                group['lr'] = 1e-4*update

        with ExitStack() as stack:
            for item in [
                    patch.object(c, 'SCRATCH', root/'scratch'), patch.object(c, 'PROOF', root/'proof'),
                    patch.object(c, 'CPU_ADMISSION', cpu),
                    patch.object(c, 'recipe', side_effect=lambda p: {**real_recipe(p), 'max_optimizer_updates': 4}),
                    patch.object(c, 'checkpoint_updates', return_value=[1, 3, 4]),
                    patch.object(c, 'snapshot_updates', return_value=list(snapshots)),
                    patch.object(c, 'prepare_data', return_value=(ds, starts, {'action': {'mean': torch.zeros(8)}},
                                                                  binding)),
                    patch.object(c, 'admission', return_value=binding),
                    patch.object(c, 'validate_scratch'), patch.object(c, 'validate_resume_proof'),
                    patch.object(c, 'make_config', return_value=config),
                    patch.object(r, 'load_fresh_policy', return_value=(tiny, tiny.get_optim_params())),
                    patch.object(r, 'EXPECTED_PARAM_TENSORS', 1),
                    patch.object(r, 'require_resources', return_value={}), patch.object(r, 'require_disk'),
                    patch.object(c, 'verify_checkpoints'),
                    patch.object(c, 'check_model_inputs', return_value={'checked': True}),
                    patch.object(c, 'set_learning_rate', side_effect=rate),
                    patch.object(c, 'collate_rows',
                                 side_effect=lambda rows: {**frame(), 'action': torch.zeros(4, 30, 8)}),
                    patch.object(c, 'backward_microbatch', side_effect=lambda loss: (loss/2).backward()),
                    patch.object(c, 'observation', side_effect=lambda x: dict(x)),
                    patch.object(c, 'sample_batches', side_effect=batches),
                    patch.object(c, 'inventory',
                                 side_effect=lambda d: {p.name: c.sha256(p) for p in Path(d).iterdir()}),
                    patch.object(c.p, 'verify_precision'), patch.object(c.p, 'memory', return_value={}),
                    patch.object(c.p, 'load_weights', side_effect=lambda pol, path: pol),
                    patch.object(c.p, 'process_row', side_effect=lambda pre, obs: obs),
                    patch.object(r, '_reload_and_verify', side_effect=reload),
                    patch.object(torch.cuda, 'is_available', return_value=True),
                    patch.object(torch.cuda, 'is_bf16_supported', return_value=True),
                    patch.object(torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'synchronize'),
                    patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
                    patch.object(torch.cuda, 'get_device_name', return_value='fake'),
                    patch.object(torch.cuda, 'get_device_properties', return_value=object()),
                    patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
                    patch.object(torch.nn.utils, 'clip_grad_norm_',
                                 side_effect=lambda ps, m, error_if_nonfinite=True: torch.tensor(.5)),
                    patch('lerobot.policies.xvla.processor_xvla.make_xvla_pre_post_processors',
                          return_value=(SynProcessor('policy_preprocessor.json'),
                                        SynProcessor('policy_postprocessor.json'))),
                    patch.object(torch, 'use_deterministic_algorithms')]:
                stack.enter_context(item)
            report = {**real_recipe(False), 'binding': None}
            if resume:
                report = {}
            r.train(out, False, report, resume=resume)
        return tiny.action_out_proj.weight.detach().clone(), report

    @staticmethod
    def _metric_rows(directory):
        keys = ('update', 'loss', 'grad_norm', 'dataset_rows', 'learning_rates')
        return [{k: json.loads(line)[k] for k in keys}
                for line in (directory/'metrics.jsonl').read_text().splitlines()]

    def _crash_then_resume(self, root, snapshots=(2,), before_resume=None):
        full, part = root/'full', root/'part'
        full.mkdir()
        part.mkdir()
        weight, _ = self._synthetic_run(root, full, snapshots=snapshots)
        with self.assertRaisesRegex(RuntimeError, 'synthetic crash'):
            self._synthetic_run(root, part, crash_at=4, snapshots=snapshots)
        if before_resume:
            before_resume(part)
        torch.manual_seed(12345)  # a fresh process does not share the dead one's RNG stream
        random.seed(1)
        np.random.seed(1)
        resumed, report = self._synthetic_run(root, part, resume=1, snapshots=snapshots)
        return full, part, weight, resumed, report

    def test_a_resumed_run_reproduces_the_uninterrupted_run_exactly(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full, part, weight, resumed, report = self._crash_then_resume(root)
            self.assertEqual(len(self._metric_rows(full)), 4)
            self.assertEqual(self._metric_rows(part), self._metric_rows(full))
            self.assertTrue(torch.equal(weight, resumed))
            for update in (1, 3, 4):
                name = c.checkpoint_directory(update)
                self.assertEqual(c.sha256(part/'checkpoints'/name/'model.safetensors'),
                                 c.sha256(full/'checkpoints'/name/'model.safetensors'), name)
            self.assertTrue(torch.equal(*(torch.load(d/'checkpoints/step_000004/training_state.pt',
                                                     weights_only=True)['state'][0]['exp_avg']
                                          for d in (part, full))))
            self.assertEqual([m['update'] for m in report['checkpoints']], [1, 3, 4])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual((report['resumes'][0]['number'], report['resumes'][0]['from_update']), (1, 2))
            self.assertEqual(report['completed_updates'], 4)

    def test_resume_quarantines_what_the_snapshot_does_not_vouch_for_and_deletes_nothing(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def leftovers(part):
                (part/'resume/tmp_000003').mkdir()  # a snapshot write the crash interrupted
                (part/'resume/tmp_000003/model.pt').write_bytes(b'half')
                (part/'failure.json').write_text('{"status": "failed"}')

            full, part, _, _, report = self._crash_then_resume(root, before_resume=leftovers)
            aside = part/'quarantine/resume_01'
            self.assertEqual(sorted(p.name for p in (aside/'checkpoints').iterdir()), ['step_000003'])
            self.assertEqual(sorted(p.name for p in (aside/'resume').iterdir()), ['tmp_000003'])
            self.assertTrue((aside/'failure.json').is_file())
            self.assertTrue((aside/'result.json').is_file())
            self.assertEqual(len((aside/'metrics.jsonl').read_text().splitlines()), 3,
                             'the three rows logged before the crash are kept, aside')
            # The checkpoint the crash left behind was regenerated byte for byte.
            self.assertEqual(c.sha256(aside/'checkpoints/step_000003/model.safetensors'),
                             c.sha256(part/'checkpoints/step_000003/model.safetensors'))
            self.assertEqual(sorted(report['resumes'][0]['quarantined']),
                             sorted(['checkpoints/step_000003', 'resume/tmp_000003', 'failure.json',
                                     'result.json', 'metrics.jsonl']))

    def test_a_damaged_newest_snapshot_falls_back_to_the_previous_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def damage(part):
                self.assertEqual(sorted(p.name for p in (part/'resume').iterdir()),
                                 ['snap_000002', 'snap_000003'], 'only the newest two are kept')
                with (part/'resume/snap_000003/model.pt').open('ab') as stream:
                    stream.write(b'x')

            full, part, weight, resumed, report = self._crash_then_resume(root, snapshots=(1, 2, 3),
                                                                          before_resume=damage)
            self.assertEqual(report['resumes'][0]['from_update'], 2)
            self.assertIn('snap_000003', report['resumes'][0]['skipped_snapshots'][0])
            self.assertEqual(self._metric_rows(part), self._metric_rows(full))
            self.assertTrue(torch.equal(weight, resumed))
            self.assertTrue((part/'quarantine/resume_01/resume/snap_000003').is_dir())

    def test_resume_refuses_when_every_snapshot_is_damaged_or_the_binding_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root/'part'
            part.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'synthetic crash'):
                self._synthetic_run(root, part, crash_at=4, snapshots=(2, 3))
            with self.assertRaisesRegex(ValueError, 'refusing to resume'):
                self._synthetic_run(root, part, resume=1, snapshots=(2, 3), binding={'bound': 'other'})
            self.assertTrue((part/'metrics.jsonl').is_file(), 'a refused resume must not move anything')
            for snap in (part/'resume').iterdir():
                (snap/'optimizer.pt').write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError, 'No verifiable snapshot'):
                self._synthetic_run(root, part, resume=1, snapshots=(2, 3))
            self.assertTrue((part/'metrics.jsonl').is_file())

    def test_rng_state_roundtrips_through_the_snapshot_encoding(self):
        r = self.r
        random.seed(5)
        np.random.seed(6)
        torch.manual_seed(7)
        state = r.capture_rng()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        python = [state['python'][0], list(state['python'][1]), state['python'][2]]
        numpy = json.loads(json.dumps([state['numpy'][0], state['numpy'][1].tolist(), *state['numpy'][2:]]))
        random.random(), np.random.rand(), torch.rand(3)
        r.restore_rng(json.loads(json.dumps(python)), numpy, {'torch': state['torch']})
        self.assertEqual((random.random(), float(np.random.rand()), float(torch.rand(1))), expected)

    def test_failures_never_overwrite_an_earlier_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for number, text in enumerate(('first', 'second', 'third'), start=1):
                self.r.failure(out, {'stage': 'x'}, ValueError(text))
            names = sorted(p.name for p in out.glob('failure*.json'))
            self.assertEqual(names, ['failure.json', 'failure_2.json', 'failure_3.json'])
            self.assertEqual(json.loads((out/'failure.json').read_text())['error'], 'first')
            self.assertEqual(json.loads((out/'failure_3.json').read_text())['error'], 'third')

    def test_resume_proof_validation_needs_bit_exact_equality_with_the_scratch_run(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_json(root/'training_stats.json', {'action': {'mean': [0.]*8}})
            markers, files = self._synthetic_checkpoint(root, [1, 2], 2, True)
            binding = {'data': 'sealed'}
            scratch = {'checkpoints': markers}
            report = {**c.recipe(True), 'status': 'complete', 'completed_updates': 2, 'binding': binding,
                      'final_data_binding_passed': True, 'adam_reload_fp32_exact': True, 'rollouts': 0,
                      'checkpoints': markers, 'checkpoint_files': files,
                      'training_stats_sha256': c.sha256(root/'training_stats.json'),
                      'resume_proof': dict(from_update=1, compared_update=2, metrics_equal=True,
                                           model_sha256_equal=True, training_state_sha256_equal=True,
                                           scratch_model_sha256=markers[-1]['model_sha256'])}
            c.validate_resume_proof(report, binding, scratch, root)
            for key in ('metrics_equal', 'model_sha256_equal', 'training_state_sha256_equal'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.validate_resume_proof({**report, 'resume_proof': {**report['resume_proof'], key: False}},
                                            binding, scratch, root)
            for change in ({'binding': {}}, {'completed_updates': 1}, {'resume_proof': None},
                           {'adam_reload_fp32_exact': False},
                           {'resume_proof': {**report['resume_proof'], 'from_update': 0}},
                           {'resume_proof': {**report['resume_proof'], 'scratch_model_sha256': 'other'}}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_resume_proof({**report, **change}, binding, scratch, root)

    def test_compare_with_reference_detects_any_divergence(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp)/'a', Path(tmp)/'b']
            for root in roots:
                (root/'checkpoints/step_000002').mkdir(parents=True)
                (root/'checkpoints/step_000002/model.safetensors').write_bytes(b'm')
                (root/'checkpoints/step_000002/training_state.pt').write_bytes(b'o')
                rows = [dict(update=u, loss=.5, grad_norm=.1, dataset_rows=[[u]], learning_rates={'o': 1e-4},
                             seconds=float(u+len(root.name))) for u in (1, 2)]
                (root/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            same = self.r.compare_with_reference(roots[0], roots[1], 2)
            self.assertEqual((same['metrics_equal'], same['model_sha256_equal'],
                              same['training_state_sha256_equal']), (True, True, True),
                             'wall-clock fields must not count')
            (roots[0]/'checkpoints/step_000002/model.safetensors').write_bytes(b'M')
            (roots[0]/'checkpoints/step_000002/training_state.pt').write_bytes(b'O')
            rows = [json.loads(x) for x in (roots[0]/'metrics.jsonl').read_text().splitlines()]
            rows[1]['loss'] = .75
            (roots[0]/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            different = self.r.compare_with_reference(roots[0], roots[1], 2)
            self.assertEqual((different['metrics_equal'], different['model_sha256_equal'],
                              different['training_state_sha256_equal']), (False, False, False))

    def test_main_resume_uses_the_original_mount_and_fresh_runs_still_refuse_a_used_one(self):
        r, c = self.r, self.c
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUTPUT', Path(tmp)/'main'), \
                patch.object(c, 'is_mountpoint', return_value=True), \
                patch.object(r, 'train') as train:
            out = Path(tmp)/'main'
            out.mkdir()
            with self.assertRaises(FileExistsError):
                r.main(['--train', '--resume'])  # nothing was ever claimed here
            r.main(['--train'])
            c.write_json(out/'result.json', {'status': 'training'})
            with self.assertRaises(FileExistsError):
                r.main(['--train'])  # a used output is never silently reused
            r.main(['--train', '--resume'])
            self.assertEqual(train.call_args.kwargs['resume'], 1)
            self.assertEqual(json.loads((out/'result_before_resume_01.json').read_text()),
                             {'status': 'training'})
            r.main(['--train', '--resume'])
            self.assertEqual(train.call_args.kwargs['resume'], 2)


if __name__ == '__main__':
    unittest.main()
