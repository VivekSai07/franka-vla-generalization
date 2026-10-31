"""Synthetic CPU models; no pretrained model, CUDA or real training (X-VLA P2 continuation pilots)."""
import ast
import importlib
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_contract import FRAMES, PLANNED, VARIANTS, bundle_mounted, mounted, joined_row

PACKAGE = Path(__file__).resolve().parents[1]
M2_RUNNER = PACKAGE.parent/'xvla_p2_force/runner.py'
# Functions that are M2's plumbing, copied unchanged: they must stay byte-identical in meaning (AST) to M2's.
COPIED = ('verify_precision_fast', 'LockedDataset', 'serial_fetch', 'Prefetcher', 'one_update', 'gpu_telemetry',
          'parse_telemetry', 'too_hot', 'cool_down', 'state_digest', 'require_disk', 'failure', 'statistics_json',
          '_reload_and_verify')


def runner():
    return importlib.import_module('experiments.xvla_p2_pilot.runner')


def definitions(path):
    tree = ast.parse(Path(path).read_text())
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class SynConfig:
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
                common = dict(lr=1e-4, betas=(.9, .95), eps=1e-8)
                return torch.optim.AdamW([
                    dict(params=[params['a']], name='vlm', weight_decay=1e-5, **common),
                    dict(params=[params['b']], name='soft_prompts', weight_decay=1e-4, **common),
                    dict(params=[params['proj.weight']], name='other', weight_decay=1e-4, **common)])
        return Preset()


class SynTiny(torch.nn.Module):
    """A linear read of the 19-value state: the extras change the loss and the gradient, so leaking one is visible."""

    def __init__(self):
        super().__init__()
        self.config = SynConfig()
        self.a, self.b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
        self.proj = torch.nn.Linear(19, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.linspace(.1, 1., 19).unsqueeze(0))

    def forward(self, batch):
        return self.proj(batch['observation.state']).square().mean()+0*(self.a+self.b).sum(), {}

    def predict_action_chunk(self, batch):
        return self.proj.weight.sum().expand(1, 30, 8)

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


class SynDataset:
    """Raw rows (9-value state) at their sealed positions; the pilot's `VariantDataset` joins the extras."""
    meta = SimpleNamespace(episodes=None)

    def __len__(self):
        return 540*FRAMES

    def __getitem__(self, i):
        episode, position = divmod(int(i), FRAMES)
        generator = torch.Generator().manual_seed(int(i))
        return {'episode_index': torch.tensor(episode), 'frame_index': torch.tensor(position),
                'observation.state': torch.randn(9, generator=generator),
                'task': PLANNED[episode],
                'observation.images.image': torch.zeros(3, 512, 512),
                'observation.images.image2': torch.zeros(3, 512, 512),
                'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}


class CliTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()

    def test_a_mode_a_variant_and_an_output_are_required_and_nothing_is_abbreviated_or_resumable(self):
        for argv in ([], ['--train'], ['--train', '--variant', 'force'], ['--variant', 'force', '--out', '/x'],
                     ['--train', '--check', '--variant', 'force', '--out', '/x'],
                     ['--tr', '--variant', 'force', '--out', '/x'], ['--train', '--variant', 'force', '--out', '/x', '--resume'],
                     ['--train', '--variant', 'force', '--out', '/x', '--stop-at', '1'],
                     ['--train', '--variant', 'force', '--out', '/x', '--preflight'],
                     ['--train', '--variant', 'force', '--out', '/x', '--batch-size', '4'],
                     ['--check', '--variant', 'force', '--out', '/x', '--updates', '10'],
                     ['--zero-equivalence', '--variant', 'force', '--out', '/x', '--updates', '10'],
                     ['--train', '--variant', 'force', '--out', '/x', '--updates', '0'],
                     ['--train', '--variant', 'force', '--out', '/x', '--updates', '10001'],
                     ['--train', '--variant', 'force', '--out', '/x', '--updates', 'many']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.r.parse_args(argv)
        a = self.r.parse_args(['--train', '--variant', 'force', '--out', '/x'])
        self.assertEqual((a.train, a.check, a.zero_equivalence, a.variant, a.out, a.updates),
                         (True, False, False, 'force', '/x', self.r.c.UPDATES))
        self.assertEqual(self.r.parse_args(['--train', '--variant', 'box', '--out', '/x', '--updates', '300']).updates, 300)
        self.assertTrue(self.r.parse_args(['--check', '--variant', 'box', '--out', '/x']).check)
        self.assertTrue(self.r.parse_args(['--zero-equivalence', '--variant', 'box', '--out', '/x']).zero_equivalence)

    def test_a_variant_that_is_not_a_known_file_stops_before_any_work(self):
        with self.assertRaises(ValueError):
            self.r.main(['--train', '--variant', 'nonesuch', '--out', '/nonexistent/out'])

    def test_the_horizon_comes_from_the_contract_not_a_literal(self):
        source = (PACKAGE/'runner.py').read_text()
        for literal in ('10000', '10_000', 'else 5000', '5000'):
            self.assertNotIn(literal, source, 'the runner must not hardcode the training horizon')
        self.assertEqual(self.r.parse_args(['--train', '--variant', 'box', '--out', '/x']).updates, self.r.c.recipe()
                         ['max_optimizer_updates'])


class CopiedPlumbingTests(unittest.TestCase):
    def test_the_plumbing_copied_from_m2_is_unchanged(self):
        ours, theirs = definitions(PACKAGE/'runner.py'), definitions(M2_RUNNER)
        for name in COPIED:
            self.assertIn(name, ours, name)
            self.assertEqual(ours[name], theirs[name], f'{name} differs from M2\'s')

    def test_the_comparison_reaches_m2s_runner(self):
        self.assertIn('Prefetcher', definitions(M2_RUNNER))
        self.assertGreater(len(definitions(M2_RUNNER)), 20)

    def test_the_update_math_lives_in_one_function_used_by_every_loop(self):
        source = (PACKAGE/'runner.py').read_text()
        self.assertEqual(source.count('one_update('), 3)  # its definition, the training loop, the equivalence loop
        for forbidden in ('clip_grad_norm_(', 'optimizer.step()'):
            self.assertEqual(source.count(forbidden), 1, forbidden)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()

    def test_resources_fail_below_15_gib_disk_or_28_gib_gpu(self):
        for disk, gpu, passes in ((14, 40, False), (15, 40, True), (15, 27, False), (100, 28, True)):
            with patch.object(self.r.shutil, 'disk_usage', return_value=SimpleNamespace(free=disk*1024**3)), \
                    patch.object(torch.cuda, 'mem_get_info', return_value=(gpu*1024**3, 48*1024**3)):
                if passes:
                    self.r.require_resources(Path('/tmp'))
                else:
                    with self.assertRaises(RuntimeError):
                        self.r.require_resources(Path('/tmp'))

    def test_a_checkpoint_needs_little_disk_because_nothing_else_is_kept(self):
        # One 3.3 GB checkpoint, no snapshots, no optimizer state: the guard is a margin over that, not a working set.
        self.assertGreaterEqual(15*1024**3, 3.5e9*3)


class ProcessorCheckTests(unittest.TestCase):
    def test_the_real_processor_and_model_input_check_pass_for_every_variant_with_no_model_construction(self):
        r = runner()
        if not (mounted() and bundle_mounted()):
            self.skipTest('S0 sidecar or base bundle not mounted')
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        selected = [0, 421, 422, 203_039]
        for name in VARIANTS:
            builder = r.c.BUILDERS[name]
            rows = {i: joined_row(r.c, builder, i) for i in selected}
            evidence = r.processor_check(rows, selected, stats, builder)
            self.assertEqual(evidence['image_shape'], [4, 3, 3, 224, 224])
            self.assertEqual((evidence['state_width'], evidence['extra_width']), (19, builder.width))
            self.assertTrue(evidence['physical_action_roundtrip'])
            self.assertFalse(evidence['whole_policy_constructed'])
            self.assertFalse(evidence['backward_executed'])

    def test_the_runner_hands_the_dataset_rows_and_the_builder_to_the_model_input_check(self):
        source = (PACKAGE/'runner.py').read_text()
        self.assertEqual(source.count('c.check_model_inputs('), 2)
        self.assertIn("c.check_model_inputs(policy, batch, raw['observation.state'], selected, builder, zero)", source)
        self.assertIn("c.check_model_inputs(policy, checked, raw['observation.state'], checked_rows, builder, zero)", source)


class StartTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()
        self.c = self.r.c

    def test_the_start_policy_is_loaded_from_the_pinned_bundle_weights_and_needs_the_exact_parameter_budget(self):
        c = self.c

        class Good:
            def __init__(self):
                self._np = {f'p{i}': torch.nn.Parameter(torch.zeros(879482456//901+(1 if i < 879482456 % 901 else 0)))
                            for i in range(901)}

            def named_parameters(self):
                return list(self._np.items())

            def get_optim_params(self):
                return dict(self._np)

        class Bad(Good):
            def __init__(self):
                self._np = {f'p{i}': torch.nn.Parameter(torch.zeros(1)) for i in range(900)}

        with patch('lerobot.policies.xvla.modeling_xvla.XVLAPolicy', lambda cfg: object()), \
                patch.object(c.p, 'load_weights') as load:
            good = Good()
            load.return_value = good
            policy, params = self.r.load_start_policy(object())
            self.assertIs(policy, good)
            self.assertEqual(len(params), 901)
            self.assertEqual(load.call_args.args[1], c.START/'model.safetensors')
            load.return_value = Bad()
            with self.assertRaises(ValueError):
                self.r.load_start_policy(object())

    def test_a_fresh_optimizer_has_no_state_and_starts_at_the_first_warm_up_rate(self):
        model = torch.nn.Linear(2, 1)
        groups = [{'params': [p], 'name': name, 'lr': 1., 'betas': (.9, .95), 'eps': 1e-8,
                   'weight_decay': 1e-4*(.1 if name == 'vlm' else 1.)}
                  for p, name in zip([model.weight, model.bias, torch.nn.Parameter(torch.zeros(1))],
                                     ('vlm', 'soft_prompts', 'other'))]
        config = SimpleNamespace(get_optimizer_preset=lambda: SimpleNamespace(
            build=lambda params: torch.optim.AdamW(groups)))
        params = {'a': model.weight, 'b': model.bias, 'c': groups[2]['params'][0]}
        with patch.object(self.r, 'EXPECTED_PARAM_TENSORS', 3):
            optimizer = self.r.fresh_optimizer(config, params)
        self.assertEqual(len(optimizer.state), 0)
        self.assertAlmostEqual(optimizer.param_groups[2]['lr'], 1e-5/200, places=15)
        # An optimizer that already stepped is not a fresh one.
        model(torch.ones(1, 2)).sum().backward()
        warm = torch.optim.AdamW(groups)
        warm.step()
        config = SimpleNamespace(get_optimizer_preset=lambda: SimpleNamespace(build=lambda params: warm))
        with patch.object(self.r, 'EXPECTED_PARAM_TENSORS', 3), self.assertRaises(ValueError):
            self.r.fresh_optimizer(config, params)

    def test_a_missing_optimizer_parameter_is_refused(self):
        model = torch.nn.Linear(2, 1)
        groups = [{'params': [model.weight], 'name': name, 'lr': 1., 'betas': (.9, .95), 'eps': 1e-8,
                   'weight_decay': 1e-4*(.1 if name == 'vlm' else 1.)} for name in ('vlm', 'soft_prompts', 'other')]
        groups[1]['params'] = [torch.nn.Parameter(torch.zeros(1))]
        groups[2]['params'] = [torch.nn.Parameter(torch.zeros(1))]
        config = SimpleNamespace(get_optimizer_preset=lambda: SimpleNamespace(build=lambda params: torch.optim.AdamW(groups)))
        with patch.object(self.r, 'EXPECTED_PARAM_TENSORS', 4), self.assertRaises(ValueError):
            self.r.fresh_optimizer(config, {'a': model.weight, 'b': model.bias})


BINDING = {'bound': 'data', 'start': {'update': 101_520}}


def synthetic_stack(r, root, variant_name='force'):
    """Patches that let `train` and `zero_equivalence` run on the CPU with a one-layer policy and the real contract."""
    c = r.c
    from safetensors.torch import load_model
    pre, post = SynProcessor('policy_preprocessor.json'), SynProcessor('policy_postprocessor.json')
    stats = {'action': {'mean': torch.zeros(8)}}
    raw = SynDataset()
    starts = [i*FRAMES+j for i in c.TRAIN for j in range(FRAMES-c.CHUNK_SIZE+1)]
    created = []

    def admit(variant, zero=False):
        builder = c.BUILDERS[variant['spec']['state_builder']]
        ds = c.VariantDataset(raw, builder, zero)
        return dict(ds=ds, starts=starts, stats=stats, binding=BINDING, builder=builder,
                    evidence={'checked': True}, semantic={'ok': True}, fingerprint=c.SEAL, sidecar_fingerprint='s')

    def tiny():
        model = SynTiny()
        created.append(model)
        return model, model.get_optim_params()

    def reload(checkpoint, expected):
        new = SynTiny()
        load_model(new, str(checkpoint/'model.safetensors'), strict=True)
        with torch.inference_mode():
            actual = new.predict_action_chunk(None).cpu()
        if not torch.equal(actual, expected['prediction']):
            raise ValueError('reload mismatch')
        return 0.

    stack = ExitStack()
    for item in (
            patch.object(r, 'admit', side_effect=admit),
            patch.object(r, 'load_start_policy', side_effect=lambda config: tiny()),
            patch.object(r, 'EXPECTED_PARAM_TENSORS', 3),
            patch.object(r, 'require_resources', return_value={}),
            patch.object(r, 'require_disk'),
            patch.object(r, '_reload_and_verify', side_effect=reload),
            patch.object(r, 'processor_check', return_value={'checked': True}),
            patch.object(c, 'make_config', return_value=SynConfig()),
            patch.object(c, 'current_binding', return_value=BINDING),
            patch.object(c, 'check_model_inputs', return_value={'checked': True}),
            patch.object(c, 'inventory', side_effect=lambda directory: {p.name: 'x' for p in Path(directory).iterdir()}),
            patch.object(c, 'verify_checkpoint'),
            patch.object(c.p, 'verify_precision'),
            patch.object(c.p, 'memory', return_value={}),
            patch.object(c.p, 'process_row', side_effect=lambda pre_, obs: obs),
            patch.object(torch.cuda, 'is_available', return_value=True),
            patch.object(torch.cuda, 'is_bf16_supported', return_value=True),
            patch.object(torch.cuda, 'is_initialized', return_value=False),
            patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
            patch.object(torch.cuda, 'reset_peak_memory_stats'),
            patch.object(torch.cuda, 'synchronize'),
            patch.object(torch.cuda, 'empty_cache'),
            patch.object(torch.cuda, 'get_device_name', return_value='fake'),
            patch.object(torch.cuda, 'get_device_properties', return_value=object()),
            patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
            patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=lambda ps, m, error_if_nonfinite=True: torch.tensor(.5)),
            patch.object(r, 'gpu_telemetry', return_value=None),
            patch('lerobot.policies.xvla.processor_xvla.make_xvla_pre_post_processors', return_value=(pre, post)),
            patch.object(torch, 'use_deterministic_algorithms')):
        stack.enter_context(item)
    return stack, created


@unittest.skipUnless(mounted(), 'S0 sidecar not mounted')
class SyntheticTrainTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()
        self.c = self.r.c

    def run_train(self, root, name='force', updates=3):
        r, c = self.r, self.c
        variant = c.load_variant(name)
        out = Path(root)/f'run_{name}'
        out.mkdir()
        stack, created = synthetic_stack(r, root, name)
        with stack:
            report = {**c.recipe(updates), 'status': 'preparing', 'variant': c.variant_record(variant)}
            r.train(out, variant, report, updates=updates)
        return out, report, created

    def test_a_short_run_writes_one_final_checkpoint_with_a_marker_and_no_optimizer_state(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out, report, _ = self.run_train(tmp)
            key = c.checkpoint_directory(3)
            self.assertEqual(sorted(p.name for p in (out/'checkpoints').iterdir()), [key])
            self.assertEqual(sorted(p.name for p in (out/'checkpoints'/key).iterdir()),
                             ['checkpoint_result.json', 'config.json', 'model.safetensors',
                              'policy_postprocessor.json', 'policy_preprocessor.json'])
            marker = json.loads((out/'checkpoints'/key/'checkpoint_result.json').read_text())
            variant = c.load_variant('force')
            self.assertEqual((marker['update'], marker['optimizer_saved'], marker['variant'], marker['variant_sha256'],
                              marker['start_update'], marker['start_model_sha256']),
                             (3, False, 'force', variant['sha256'], 101_520, c.START_MODEL_SHA256))
            self.assertEqual((report['status'], report['completed_updates'], report['rollouts']), ('complete', 3, 0))
            self.assertEqual(report['variant']['sha256'], variant['sha256'])
            self.assertEqual(report['variant']['builder']['transform_source'],
                             c.builder_record(c.BUILDERS['force'])['transform_source'])
            self.assertEqual((out/'variant.json').read_bytes(), (c.VARIANTS/'force.json').read_bytes())
            self.assertEqual(c.sha256(out/'variant.json'), variant['sha256'])

    def test_the_logged_rates_follow_the_warm_up_and_the_sampler_is_the_shared_one(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out, _, _ = self.run_train(tmp, updates=3)
            rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual([r['update'] for r in rows], [1, 2, 3])
        for row in rows:
            self.assertAlmostEqual(row['learning_rates']['other'], c.learning_rate(row['update']), places=15)
            self.assertEqual(len(row['dataset_rows']), 2)
            self.assertTrue(all(len(batch) == 4 for batch in row['dataset_rows']))
        starts = [i*FRAMES+j for i in c.TRAIN for j in range(FRAMES-c.CHUNK_SIZE+1)]
        stream = c.sample_batches(starts)
        expected = [[next(stream), next(stream)] for _ in range(3)]
        self.assertEqual([row['dataset_rows'] for row in rows], expected)

    def test_every_variant_sees_the_same_rows_in_the_same_order(self):
        orders = {}
        for name in VARIANTS:
            with tempfile.TemporaryDirectory() as tmp:
                out, _, _ = self.run_train(tmp, name, 2)
                orders[name] = [json.loads(line)['dataset_rows'] for line in (out/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(orders['control'], orders['force'])
        self.assertEqual(orders['control'], orders['box'])

    def test_the_continuation_starts_from_the_loaded_weights_with_a_fresh_optimizer_and_differs_by_variant_only(self):
        first = []
        for name in ('control', 'force'):
            with tempfile.TemporaryDirectory() as tmp:
                out, _, created = self.run_train(tmp, name, 2)
                first.append(json.loads((out/'metrics.jsonl').read_text().splitlines()[0]))
                self.assertEqual(len(created), 1)  # one policy, loaded from the start checkpoint
        self.assertNotEqual(first[0]['loss'], first[1]['loss'])  # the extras are in the model's input
        self.assertEqual(first[0]['learning_rates']['other'], first[1]['learning_rates']['other'])

    def test_a_binding_that_changed_during_the_run_leaves_no_checkpoint(self):
        r, c = self.r, self.c
        variant = c.load_variant('control')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            out.mkdir()
            stack, _ = synthetic_stack(r, tmp)
            with stack, patch.object(c, 'current_binding', return_value={'bound': 'changed'}), self.assertRaises(ValueError):
                r.train(out, variant, {**c.recipe(2), 'variant': c.variant_record(variant)}, updates=2)
            self.assertFalse((out/'checkpoints').exists())

    def test_a_failed_run_leaves_its_failure_record_and_no_checkpoint(self):
        r, c = self.r, self.c
        variant = c.load_variant('control')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            out.mkdir()
            stack, _ = synthetic_stack(r, tmp)
            with stack, patch.object(c, 'collate_rows', side_effect=ValueError('bad row')), self.assertRaises(ValueError):
                report = {**c.recipe(2), 'variant': c.variant_record(variant)}
                try:
                    r.train(out, variant, report, updates=2)
                except BaseException as exc:
                    r.failure(out, report, exc)
                    raise
            self.assertTrue((out/'failure.json').is_file())
            self.assertEqual(json.loads((out/'failure.json').read_text())['status'], 'failed')
            self.assertFalse((out/'checkpoints').exists() and any((out/'checkpoints').iterdir()))


@unittest.skipUnless(mounted(), 'S0 sidecar not mounted')
class ZeroEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()
        self.c = self.r.c

    def run_gate(self, name, root, patches=()):
        """The gate's output directory; the gate's own exception propagates."""
        r = self.r
        out = Path(root)/f'gate_{name}'
        out.mkdir()
        stack, _ = synthetic_stack(r, root, name)
        with stack, ExitStack() as extra:
            for item in patches:
                extra.enter_context(item)
            with patch.object(r, 'ZERO_EQUIVALENCE_UPDATES', 4):
                r.zero_equivalence(name, out)
        return out

    def test_extras_forced_to_zero_are_byte_equal_to_the_control_for_every_variant(self):
        for name in VARIANTS:
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as tmp:
                report = json.loads((self.run_gate(name, tmp)/'result.json').read_text())
                self.assertEqual((report['status'], report['updates'], report['variant']['name']), ('complete', 4, name))
                self.assertTrue(report['metrics_equal'] and report['model_state_equal'] and report['optimizer_state_equal'])
                self.assertTrue(report['extras_forced_zero'])

    def test_the_gate_fails_when_a_forced_zero_variant_still_leaks_an_extra(self):
        c = self.c
        original = c.extras_row

        def leaky(builder, episode, frame, zero=False):
            row = original(builder, episode, frame, zero)
            if builder.name == 'force' and zero:
                row[0] = 1.
            return row

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.run_gate('force', tmp, [patch.object(c, 'extras_row', leaky)])
            report = json.loads((Path(tmp)/'gate_force/result.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertTrue((Path(tmp)/'gate_force/failure.json').is_file())

    def test_the_gate_compares_metrics_weights_and_adam_moments(self):
        r = self.r
        a = dict(rows=[{'update': 1, 'loss': 1.}], model='m', optimizer='o')
        self.assertEqual(r.compare_runs(a, dict(a)), dict(metrics_equal=True, model_state_equal=True, optimizer_state_equal=True))
        for change, key in (({'rows': [{'update': 1, 'loss': 1.0000001}]}, 'metrics_equal'), ({'model': 'x'}, 'model_state_equal'),
                            ({'optimizer': 'x'}, 'optimizer_state_equal')):
            result = r.compare_runs(a, {**a, **change})
            self.assertFalse(result[key])
            self.assertEqual(sum(result.values()), 2)

    def test_the_comparison_is_not_vacuous(self):
        # The same variant with its extras live gives other rows than the control; forced to zero it gives the same bytes.
        r, c = self.r, self.c
        starts = [i*FRAMES+j for i in c.TRAIN for j in range(FRAMES-c.CHUNK_SIZE+1)]
        stats = {'action': {'mean': torch.zeros(8)}}
        stack, _ = synthetic_stack(r, '/tmp')
        with stack:
            runs = {label: r.run_updates(c.VariantDataset(SynDataset(), c.BUILDERS[name], zero), starts, stats, 3)
                    for label, name, zero in (('control', 'control', False), ('live', 'force', False), ('forced', 'force', True))}
        self.assertEqual(r.compare_runs(runs['control'], runs['forced']),
                         dict(metrics_equal=True, model_state_equal=True, optimizer_state_equal=True))
        live = r.compare_runs(runs['control'], runs['live'])
        self.assertFalse(live['metrics_equal'])
        self.assertFalse(live['model_state_equal'])
        self.assertFalse(live['optimizer_state_equal'])


class MainTests(unittest.TestCase):
    def test_the_runner_has_no_resume_snapshot_or_evaluation_code(self):
        tree = ast.parse((PACKAGE/'runner.py').read_text())
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for needle in ('resume', 'snapshot', 'rollout', 'evaluate', 'quarantine'):
            self.assertEqual([n for n in names if needle in n.lower()], [], needle)
        options = {arg for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument'
                   for arg in [a.value for a in node.args if isinstance(a, ast.Constant)]}
        self.assertEqual(options, {'--variant', '--out', '--updates'})  # the three modes are added from one tuple


if __name__ == '__main__':
    unittest.main()
