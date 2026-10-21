"""Synthetic CPU models; no pretrained model, CUDA or real training (X-VLA props P3)."""
from contextlib import ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
                        'X-VLA props P3 runner missing')
        self.r = importlib.import_module('experiments.xvla_props_p3.runner')
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
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        rows = {i: frame(0) for i in (0, 421, 422, 203_039)}
        evidence = self.r.processor_check(rows, [0, 421, 422, 203_039], stats)
        self.assertEqual(evidence['image_shape'], [4, 3, 3, 224, 224])
        self.assertTrue(evidence['physical_action_roundtrip'])
        self.assertFalse(evidence['whole_policy_constructed'])
        self.assertFalse(evidence['backward_executed'])

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
                with self.assertRaisesRegex(ValueError, 'Props P3 parent fingerprint differs from the seal'):
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


if __name__ == '__main__':
    unittest.main()
