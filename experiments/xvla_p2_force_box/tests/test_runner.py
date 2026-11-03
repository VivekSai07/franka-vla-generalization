"""Synthetic CPU models; no pretrained model, CUDA or real training (X-VLA P2 + force/torque)."""
from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_boxes_join import FakeBase, mounted
from test_contract import frame, metadata
from test_inputs import row as joined_row


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
                        'X-VLA P2 force runner missing')
        self.r = importlib.import_module('experiments.xvla_p2_force_box.runner')
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

    def test_resources_fail_below_preflight_28_and_train_65_disk_or_28_gpu(self):
        from types import SimpleNamespace
        for disk, gpu, preflight, passes in ((27, 40, True, False), (28, 40, True, True),
                                             (28, 27, True, False),
                                             (64, 40, False, False), (65, 40, False, True),
                                             (65, 27, False, False)):
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
        """6 intermediate checkpoints (3.3 GB each) + final 9.9 GB + KEEP_SNAPSHOTS kept + 1 being written."""
        from types import SimpleNamespace
        updates = self.r.c.checkpoint_updates(False)
        self.assertEqual(len(updates), 7)
        needed_gb = 3.3*(len(updates)-1) + 9.9 + (self.r.c.KEEP_SNAPSHOTS+1)*10.5
        self.assertLess(needed_gb*1e9, 65*1024**3, 'the 65 GiB guard must cover the working set')
        with patch.object(self.r.shutil, 'disk_usage',
                          return_value=SimpleNamespace(free=int(needed_gb*1e9))), \
                patch.object(torch.cuda, 'mem_get_info', return_value=(40*1024**3, 48*1024**3)):
            with self.assertRaises(RuntimeError):  # free space equal to the bare need is not enough
                self.r.require_resources(Path('/tmp'), False)

    def test_processor_check_uses_real_processor_no_model_construction(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        selected = [0, 421, 422, 203_039]
        rows = {i: joined_row(self.c, i) for i in selected}
        evidence = self.r.processor_check(rows, selected, stats)
        self.assertEqual(evidence['image_shape'], [4, 3, 3, 224, 224])
        self.assertEqual((evidence['real_state_dim'], evidence['extra_state_dim']), (17, 8))
        self.assertTrue(evidence['physical_action_roundtrip'])
        self.assertEqual(evidence['overlay']['thickness'], 4)
        self.assertEqual(len(evidence['overlay']['band_pixels']), 8)  # 4 samples x 2 cameras
        self.assertTrue(all(pixels > 0 for pixels in evidence['overlay']['band_pixels']))
        self.assertFalse(evidence['whole_policy_constructed'])
        self.assertFalse(evidence['backward_executed'])

    def test_the_runner_hands_the_dataset_rows_to_the_model_input_check(self):
        """The check recomputes the expected extras and boxes from these row indices and checks the frames before the processors;
        both call sites must pass them."""
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertEqual(source.count('c.check_model_inputs('), 2)
        self.assertIn("c.check_model_inputs(policy, batch, raw['observation.state'], selected, {key: raw[key] for key in c.CAMERAS})",
                      source)
        self.assertIn("c.check_model_inputs(policy, checked, raw['observation.state'], checked_rows,\n"
                      "                                                       {key: raw[key] for key in c.CAMERAS})", source)

    def test_absolute_reload_marker_reject_relative_only_and_nonfinite(self):
        for update in (1, 2, 126900):
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
            saved = torch.load(path, weights_only=False)
            first = next(iter(saved['state'].values()))
            first['exp_avg'] = first['exp_avg'].bfloat16()
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                self.r.adam_roundtrip(opt, path, 2)

    def test_load_fresh_policy_requires_expected_parameter_budget(self):
        c = self.c
        config = object()
        with patch('lerobot.policies.xvla.modeling_xvla.XVLAPolicy', lambda cfg: object()), \
                patch.object(c.p, 'load_weights') as lw:
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
            bad = dict(checkpoints=markers[:-1], checkpoint_files=files)
            with self.assertRaises(ValueError):
                c.verify_checkpoints(bad, root, False)
            (root/f'checkpoints/{updates[-1]:06d}').mkdir()
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, root, False)

    def test_train_two_updates_with_synthetic_policy_writes_step_prefixed_checkpoints(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out, report = self._synthetic_train(Path(tmp))
            self.assertEqual(report['completed_updates'], 2)
            self.assertEqual([m['update'] for m in report['checkpoints']], [1, 2])
            for update in (1, 2):
                key = c.checkpoint_directory(update)
                self.assertTrue((out/'checkpoints'/key).is_dir())
                self.assertFalse((out/f'checkpoints/{update:06d}').exists())
            self.assertFalse((out/'checkpoints/step_000001/training_state.pt').exists())
            self.assertTrue((out/'checkpoints/step_000002/training_state.pt').exists())
            self.assertTrue((out/'checkpoints/step_000002/adam_reload.json').is_file())

    def test_a_past_deadline_pauses_cleanly_at_the_first_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, report = self._synthetic_train(Path(tmp), stop_at=0.)
            self.assertEqual(report['status'], 'paused')
            self.assertEqual(report['paused_at_update'], 1)
            self.assertEqual(report['completed_updates'], 1)
            self.assertTrue((out/'resume/snap_000001/snapshot.json').is_file())
            self.assertFalse((out/'checkpoints/step_000002').exists())

    def _synthetic_train(self, root, **train_kwargs):
        from types import SimpleNamespace
        from safetensors.torch import load_model
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
                from safetensors.torch import save_model
                save_model(self, str(path/'model.safetensors'))
                self.config.save_pretrained(path)
            def get_optim_params(self):
                return dict(self.named_parameters())
            def to(self, device):
                return self

        pre, post = SynProcessor('policy_preprocessor.json'), SynProcessor('policy_postprocessor.json')
        stats = {'action': {'mean': torch.zeros(8)}}

        class Dataset:
            meta = SimpleNamespace(episodes=metadata())
            sample = frame()
            def __getitem__(self, index):
                return {**self.sample, 'episode_index': torch.tensor(index//452)}

        ds = Dataset()
        raw_starts = c.indices(metadata())[1]

        def _reload(checkpoint, expected):
            new = Tiny()
            load_model(new, str(checkpoint/'model.safetensors'), strict=True)
            new.eval()
            with torch.inference_mode():
                actual = new.predict_action_chunk(None).cpu()
            if not torch.equal(actual, expected['prediction']):
                raise ValueError('reload mismatch')
            return 0.

        with ExitStack() as stack:
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
            r.train(out, True, report, **train_kwargs)
            return out, report

    # --- the faster loop's plumbing ---------------------------------------------------------

    def _verify_both(self, params, optimizer=None, require_gradients=False):
        outcomes = []
        for fn in (self.c.p.verify_precision, self.r.verify_precision_fast):
            try:
                fn(params, optimizer, require_gradients)
                outcomes.append('ok')
            except ValueError as exc:
                outcomes.append(str(exc))
        return outcomes

    def test_fused_checks_accept_and_reject_exactly_what_the_original_does(self):
        def fresh():
            model = torch.nn.Linear(3, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            model(torch.ones(4, 3)).square().mean().backward()
            opt.step()
            return dict(model.named_parameters()), opt
        cases = {}
        params, opt = fresh()
        cases['clean'] = (params, opt, True)
        params, opt = fresh()
        with torch.no_grad():
            params['weight'][0, 0] = float('nan')
        cases['nan_param'] = (params, opt, True)
        params, opt = fresh()
        params['bias'].grad[0] = float('inf')
        cases['inf_grad'] = (params, opt, True)
        params, opt = fresh()
        params['weight'].grad = None
        cases['missing_grad'] = (params, opt, True)
        params, opt = fresh()
        opt.state[params['bias']]['exp_avg_sq'][0] = float('-inf')
        cases['inf_adam'] = (params, opt, True)
        params, opt = fresh()
        del opt.state[params['bias']]['exp_avg']
        cases['missing_adam'] = (params, opt, True)
        params, opt = fresh()
        params['bias'].requires_grad_(False)
        cases['frozen'] = (params, opt, False)
        params, opt = fresh()
        params['weight'].data = params['weight'].data.double()
        cases['fp64'] = (params, None, False)
        params, opt = fresh()
        with torch.no_grad():
            params['weight'][0, 0] = 3e38
        cases['huge_but_finite'] = (params, opt, True)
        for name, (params, opt, grads) in cases.items():
            for optimizer in (None, opt):
                with self.subTest(case=name, optimizer=optimizer is not None):
                    original, fast = self._verify_both(params, optimizer, grads)
                    self.assertEqual(original, fast)
        self.assertEqual(self._verify_both(*cases['clean'][:2], True), ['ok', 'ok'])
        self.assertNotEqual(self._verify_both(*cases['nan_param'][:2], True)[1], 'ok')

    def test_prefetcher_yields_exactly_the_serial_sequence_and_reraises(self):
        r, c = self.r, self.c
        with patch.object(c, 'collate_rows', side_effect=lambda rows: list(rows)):
            serial = r.serial_fetch(list(range(100)), iter([[i, i+1, i+2, i+3] for i in range(0, 80, 4)]))
            expected = [next(serial) for _ in range(20)]
            fetch = r.Prefetcher(r.LockedDataset(list(range(100))),
                                 iter([[i, i+1, i+2, i+3] for i in range(0, 80, 4)]))
            try:
                self.assertEqual([next(fetch) for _ in range(20)], expected)
                with self.assertRaises(StopIteration):
                    next(fetch)
            finally:
                fetch.close()

    def test_should_pause_only_when_the_next_snapshot_would_miss_the_deadline(self):
        r = self.r
        self.assertFalse(r.should_pause(1000., None, .5, 2600))
        self.assertFalse(r.should_pause(1000., 1000.+.5*2600+r.SNAPSHOT_WRITE_S+1, .5, 2600))
        self.assertTrue(r.should_pause(1000., 1000.+.5*2600+r.SNAPSHOT_WRITE_S-1, .5, 2600))
        self.assertTrue(r.should_pause(1000., 0., .5, 2600))

    def test_telemetry_parse_and_heat_decision(self):
        r = self.r
        cool = r.parse_telemetry('62, 210.5, 300.00, Not Active, Not Active, 97\n')
        self.assertEqual(cool, dict(temperature_c=62, power_w=210.5, power_limit_w=300., thermal_slowdown=False,
                                    utilization=97))
        self.assertFalse(r.too_hot(cool))
        self.assertTrue(r.too_hot({**cool, 'temperature_c': r.HOT_C}))
        self.assertTrue(r.too_hot({**cool, 'thermal_slowdown': True}))
        self.assertFalse(r.too_hot(None))
        self.assertTrue(r.parse_telemetry('84, 299, 300, Active, Not Active, 99')['thermal_slowdown'])

    def test_cool_down_waits_for_cool_and_gives_up_after_the_limit(self):
        r = self.r
        clock = {'t': 0.}
        sleep = lambda s: clock.__setitem__('t', clock['t']+s)
        temps = iter([85, 82, 76, 75])
        read = lambda: dict(temperature_c=next(temps), thermal_slowdown=False)
        self.assertEqual(r.cool_down(read, sleep, lambda: clock['t']), 90.)
        clock['t'] = 0.
        hot = lambda: dict(temperature_c=90, thermal_slowdown=False)
        self.assertIsNone(r.cool_down(hot, sleep, lambda: clock['t']))
        self.assertEqual(r.cool_down(lambda: None, sleep, lambda: clock['t']), 0.)

    def test_cli_equivalence_and_stop_at(self):
        r = self.r
        self.assertTrue(r.parse_args(['--equivalence']).equivalence)
        self.assertEqual(r.parse_args(['--train', '--stop-at', '1760000000']).stop_at, 1760000000.)
        self.assertEqual(r.parse_args(['--train', '--resume', '--stop-at', '1']).stop_at, 1.)
        for argv in (['--preflight', '--stop-at', '1'], ['--equivalence', '--resume'], ['--equivalence', '--train']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                r.parse_args(argv)

    def test_train_requires_the_equivalence_proof_and_it_must_be_exact(self):
        c = self.c
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertIn("c.validate_equivalence(json.loads((c.EQUIVALENCE/'result.json').read_text()), binding)", source)
        good = dict(status='complete', binding={'b': 1}, updates=30, metrics_equal=True,
                    model_state_equal=True, optimizer_state_equal=True)
        c.validate_equivalence(good, {'b': 1})
        for change in ({'metrics_equal': False}, {'model_state_equal': False}, {'optimizer_state_equal': False},
                       {'updates': 29}, {'binding': {'b': 2}}, {'status': 'failed'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_equivalence({**good, **change}, {'b': 1})

    def test_the_update_math_lives_in_one_function_used_by_both_loops(self):
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertEqual(source.count('optimizer.step()'), 1)
        self.assertEqual(source.count('clip_grad_norm_('), 1)
        self.assertIn('one_update(update, policy, params, optimizer, pre, fetch, verify_precision_fast)', source)
        self.assertIn('rows.append(one_update(update, policy, params, optimizer, pre, fetch, verify))', source)

    def test_check_recomputes_the_parent_fingerprint_before_anything_else(self):
        r, c = self.r, self.c
        with tempfile.TemporaryDirectory() as tmp:
            admission = Path(tmp)/'admission'
            with patch.object(c, 'ADMISSION', admission), \
                    patch.object(c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(c, 'parent_fingerprint', return_value='0'*64), \
                    patch.object(c, 'current_binding', side_effect=AssertionError('binding before fingerprint')), \
                    patch.object(c, 'load_data', side_effect=AssertionError('data before fingerprint')):
                with self.assertRaisesRegex(ValueError, 'Props P2 parent fingerprint differs from the seal'):
                    r.check()
                report = json.loads((admission/'result.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertNotEqual(report.get('status'), 'passed')
                self.assertTrue((admission/'failure.json').is_file())

    def test_check_verifies_the_sidecar_before_the_binding_and_the_data(self):
        r, c = self.r, self.c
        with tempfile.TemporaryDirectory() as tmp:
            admission = Path(tmp)/'admission'
            with patch.object(c, 'ADMISSION', admission), \
                    patch.object(c, 'CPU_ADMISSION', admission/'result.json'), \
                    patch.object(c, 'parent_fingerprint', return_value=c.SEAL), \
                    patch.object(c.forces, 'verify_sidecar', side_effect=ValueError('sidecar seal does not verify')), \
                    patch.object(c, 'current_binding', side_effect=AssertionError('binding before the sidecar')), \
                    patch.object(c, 'load_data', side_effect=AssertionError('data before the sidecar')):
                with self.assertRaisesRegex(ValueError, 'sidecar seal does not verify'):
                    r.check()
                self.assertEqual(json.loads((admission/'result.json').read_text())['status'], 'failed')

    def test_check_does_not_call_a_simulator_audit(self):
        source = (Path(__file__).parents[1]/'runner.py').read_text()
        self.assertNotIn('require_broad_workspace_pass', source)
        self.assertIn('c.parent_fingerprint()', source)
        self.assertIn('parent_fingerprint_recomputed', (Path(__file__).parents[1]/'contract.py').read_text())

    def test_dispatch_does_not_launch_evaluation(self):
        r, c = self.r, self.c
        self.assertFalse((Path(__file__).parents[1]/'evaluate.py').exists())
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'SCRATCH', Path(tmp)/'scratch'), \
                patch.object(r, 'train', side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):
                r.main(['--preflight'])
            self.assertEqual(json.loads((c.SCRATCH/'failure.json').read_text())['status'], 'failed')
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUTPUT', Path(tmp)/'main'), \
                patch.object(r, 'train'):
            r.main(['--train'])
            self.assertTrue((Path(tmp)/'main/run_claim.json').is_file())


class ColourAndSamplesTests(unittest.TestCase):
    """`--check`'s two additions: the colour measurement that can stop the run, and the PNG sheets the user approves."""

    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_p2_force_box.runner')
        self.c = self.r.c

    class Undrawn:
        """A joined-dataset stand-in whose wrapped dataset yields grey frames, some with a magenta patch."""

        def __init__(self, magenta_pixels=0):
            self.magenta_pixels = magenta_pixels
            self.ds = self

        def __getitem__(self, i):
            image = torch.full((3, 512, 512), 0.5)
            if self.magenta_pixels:
                image[:, 0, :self.magenta_pixels] = torch.tensor([1.0, 0.0, 1.0]).unsqueeze(1)
            return {key: image.clone() for key in ('observation.images.image', 'observation.images.image2')}

    def test_a_scene_without_the_colour_passes_with_a_zero_fraction(self):
        report = self.r.colour_check(self.Undrawn())
        self.assertTrue(report['passed'])
        self.assertEqual((report['near_magenta_pixels'], report['frames']), (0, 288))
        self.assertEqual(report['pixels'], 288*512*512)

    def test_a_scene_with_the_colour_everywhere_fails(self):
        report = self.r.colour_check(self.Undrawn(magenta_pixels=512))  # one full row per frame: 512 of 262,144 = 0.195%
        self.assertFalse(report['passed'])
        self.assertEqual(report['near_magenta_pixels'], 288*512)

    def test_the_measure_reads_every_tenth_training_episode_at_three_frames_in_both_cameras(self):
        seen = []

        class Spy(self.Undrawn):
            def __getitem__(self, i):
                seen.append(i)
                return super().__getitem__(i)

        self.r.colour_check(Spy())
        self.assertEqual(seen, [e*452+f for e in range(0, 480, 10) for f in (0, 225, 451)])

    def test_a_frame_that_is_not_float32_chw_512_is_refused(self):
        class Bad(self.Undrawn):
            def __getitem__(self, i):
                return {key: torch.zeros(3, 256, 256) for key in ('observation.images.image', 'observation.images.image2')}

        with self.assertRaises(ValueError):
            self.r.colour_check(Bad())

    def check_with(self, colour, tmp):
        r, c = self.r, self.c
        admission = Path(tmp)/'admission'
        calls = []
        binding = {'validation_sha256': c.digest_json({})}
        patches = [patch.object(c, 'ADMISSION', admission), patch.object(c, 'CPU_ADMISSION', admission/'result.json'),
                   patch.object(c, 'parent_fingerprint', return_value=c.SEAL),
                   patch.object(c.forces, 'verify_sidecar', return_value=c.forces.SIDECAR_RECORD),
                   patch.object(c, 'current_binding', return_value=binding),
                   patch.object(c, 'semantic_gate', return_value={}),
                   patch.object(c, 'load_data', return_value=(object(), list(range(500)), {'action': {'mean': torch.zeros(8)}})),
                   patch.object(r, 'processor_check', return_value={'batch_size': 4}),
                   patch.object(r, 'colour_check', return_value=colour),
                   patch.object(r, 'overlay_samples', side_effect=lambda ds, directory: calls.append(directory))]
        return admission, calls, patches

    def test_check_stops_on_a_failed_colour_check_writes_the_verdict_and_draws_no_samples(self):
        c = self.c
        failed = c.colour_report(near=10**6, pixels=288*512*512, frames=288)
        with tempfile.TemporaryDirectory() as tmp:
            admission, calls, patches = self.check_with(failed, tmp)
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                with self.assertRaisesRegex(ValueError, 'user must pick another colour'):
                    self.r.check()
            self.assertFalse(json.loads((admission/'colour_check.json').read_text())['passed'])
            self.assertEqual(json.loads((admission/'result.json').read_text())['status'], 'failed')
            self.assertEqual(calls, [])

    def test_check_passes_with_a_clean_colour_check_writes_it_and_the_samples_and_binds_the_verdict(self):
        c = self.c
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        good = c.colour_report(near=0, pixels=288*512*512, frames=288)
        with tempfile.TemporaryDirectory() as tmp:
            admission, calls, patches = self.check_with(good, tmp)
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                self.r.check()
            self.assertEqual(json.loads((admission/'colour_check.json').read_text()), good)
            self.assertEqual(calls, [admission/'overlay_samples'])
            report = json.loads((admission/'result.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['colour_check'], good)

    def test_the_sample_sheets_are_one_png_per_object_front_camera_on_top_and_drawn_by_the_training_loader(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        from PIL import Image
        import numpy as np
        c = self.c
        ds = c.JoinedDataset(FakeBase())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'overlay_samples'
            index = self.r.overlay_samples(ds, directory)
            self.assertEqual(len(index), 6)
            self.assertEqual(sorted(item['object'] for item in index), ['bottle', 'bread', 'can', 'cereal', 'lemon', 'milk'])
            self.assertTrue(all(item['episode'] in c.TRAIN for item in index))
            self.assertEqual(json.loads((directory/'index.json').read_text()), index)
            objects, segments = c.boxes.labels()
            for item in index:
                self.assertEqual(item['episode'], min(i for i in c.TRAIN if objects[i] == item['object']))
                sheet = np.asarray(Image.open(directory/item['file']))
                self.assertEqual(sheet.shape, (1024, 1536, 3))
                frames = [column['frame'] for column in item['columns']]
                self.assertEqual(frames[0], 0)
                self.assertEqual(frames[1], c.boxes.first_frame(segments[item['episode']], 1))
                self.assertEqual(frames[2], c.boxes.first_frame(segments[item['episode']], 2))
                for column, frame_index in enumerate(frames):
                    row = ds[item['episode']*452+frame_index]
                    for line, key in enumerate(c.CAMERAS):  # rows: front camera, then gripper camera
                        panel = sheet[line*512:(line+1)*512, column*512:(column+1)*512]
                        expected = (row[key].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
                        self.assertTrue(np.array_equal(panel, expected), (item['file'], line, column))
                        self.assertTrue(((panel == [255, 0, 255]).all(axis=-1)).any())

    def test_the_samples_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'overlay_samples'
            directory.mkdir()
            with self.assertRaises(FileExistsError):
                self.r.overlay_samples(self.Undrawn(), directory)


if __name__ == '__main__':
    unittest.main()
