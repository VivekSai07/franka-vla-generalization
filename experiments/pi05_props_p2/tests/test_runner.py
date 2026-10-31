"""Runner structure: modes, horizon, disk guard, snapshots, thermal guard, resume bookkeeping and the data stream."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from experiments.pi05_props_p2 import contract as c
from experiments.pi05_props_p2 import runner as r
from experiments.props_p2_data import plan as p2_plan

DECISION = dict(strategy='bf16_cpu_master', microbatch=8, snapshot_every=2600, checkpoint_bytes=7_233_582_936,
                snapshot_bytes=43_000_000_000, final_state_bytes=29_000_000_000, gpu_peak_bytes=30*1024**3)
PLANNED = tuple(e['instruction'] for e in p2_plan.make_plan()['episodes'])
FRAMES = 452


class ModeTests(unittest.TestCase):
    def test_exactly_one_mode(self):
        for argv in ([], ['--check', '--train'], ['--preflight', '--resume-proof'], ['--probe-memory', '--check']):
            with self.subTest(argv), self.assertRaises(SystemExit):
                r.parse_args(argv)
        for mode in ('probe-memory', 'check', 'preflight', 'resume-proof', 'train'):
            self.assertTrue(getattr(r.parse_args(['--'+mode]), mode.replace('-', '_')))

    def test_resume_and_stop_at_belong_to_train_alone(self):
        for argv in (['--preflight', '--resume'], ['--check', '--resume'], ['--resume-proof', '--stop-at', '1'],
                     ['--probe-memory', '--resume'], ['--preflight', '--stop-at', '5']):
            with self.subTest(argv), self.assertRaises(SystemExit):
                r.parse_args(argv)
        args = r.parse_args(['--train', '--resume', '--stop-at', '12.5'])
        self.assertTrue(args.resume)
        self.assertEqual(args.stop_at, 12.5)

    def test_probe_options_belong_to_the_probe_alone(self):
        for argv in (['--train', '--configs', 'bf16_all:8'], ['--check', '--updates', '5'],
                     ['--preflight', '--cpu-threads', '8']):
            with self.subTest(argv), self.assertRaises(SystemExit):
                r.parse_args(argv)
        args = r.parse_args(['--probe-memory', '--configs', 'bf16_all:4', '--updates', '8', '--cpu-threads', '16'])
        self.assertEqual((args.configs, args.updates, args.cpu_threads), ('bf16_all:4', 8, 16))

    def test_probe_configs(self):
        self.assertEqual(r.parse_configs('bf16_all:8,expert_only:1'), [('bf16_all', 8), ('expert_only', 1)])
        for bad in ('bf16_all', 'bf16_all:3', 'bf16_all:', 'fp32:8', 'bf16_all:8,bf16_all:8', 'bf16_all:16'):
            with self.subTest(bad), self.assertRaises(ValueError):
                r.parse_configs(bad)

    def test_every_other_mode_refuses_while_the_decision_is_open(self):
        self.assertIsNone(c.DECISION)
        for argv in (['--check'], ['--preflight'], ['--resume-proof'], ['--train']):
            with self.subTest(argv), self.assertRaisesRegex(RuntimeError, 'not decided'):
                r.main(argv)

    def test_the_probe_does_not_need_a_decision_and_never_runs_after_check(self):
        with patch.object(c, 'ADMISSION', Path(tempfile.mkdtemp())), self.assertRaisesRegex(RuntimeError, 'after --check'):
            r.probe_memory(r.parse_args(['--probe-memory']))
        with patch.object(c, 'ADMISSION', Path('/nonexistent/x')), patch.object(r.torch.cuda, 'is_available',
                                                                                return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA BF16 required'):
                r.probe_memory(r.parse_args(['--probe-memory']))


class HorizonAndGuardTests(unittest.TestCase):
    def test_the_runner_takes_its_horizon_from_the_contract_and_types_none(self):
        source = Path(r.__file__).read_text()
        self.assertIn("maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']", source)
        for typed in ('120900', '120_900', '48360', '108810'):
            self.assertNotIn(typed, source)
        self.assertEqual(c.checkpoint_updates(False)[-1], c.recipe(False)['max_optimizer_updates'])

    def test_the_runner_never_calls_the_installed_loader(self):
        source = '\n'.join(Path(r.__file__).with_name(n).read_text() for n in ('runner.py', 'strategy.py', 'weights.py'))
        for banned in ('Policy.from_pretrained', 'load_model_as_safetensor', '_load_as_safetensor'):
            self.assertNotIn(banned, source)

    def test_decision_validation(self):
        self.assertEqual(c.validate_decision(DECISION), DECISION)
        for bad in ({**DECISION, 'strategy': 'fp32'}, {**DECISION, 'microbatch': 3}, {**DECISION, 'snapshot_every': 0},
                    {**DECISION, 'checkpoint_bytes': 1.5}, {k: v for k, v in DECISION.items() if k != 'gpu_peak_bytes'},
                    {**DECISION, 'extra': 1}, None):
            with self.subTest(bad), self.assertRaises(ValueError):
                c.validate_decision(bad)

    def test_disk_guard_is_the_formula_over_the_measured_sizes(self):
        d = DECISION
        expected = (7*d['checkpoint_bytes'] + d['final_state_bytes'] + 2*d['snapshot_bytes'] + 2*1024**3)
        self.assertEqual(c.disk_requirement_bytes(False, d), expected)
        self.assertEqual(c.disk_requirement_bytes(True, d), 2*d['checkpoint_bytes'] + d['final_state_bytes']
                         + 2*d['snapshot_bytes'] + 2*1024**3)
        with patch.object(c, 'DECISION', d):
            self.assertEqual(c.gpu_requirement_bytes(), d['gpu_peak_bytes']+2*1024**3)
            self.assertEqual(c.accumulation(), 1)

    def test_resources_check_uses_the_contracts_guards(self):
        with patch.object(c, 'DECISION', DECISION), tempfile.TemporaryDirectory() as tmp:
            with patch.object(r.shutil, 'disk_usage', return_value=type('U', (), {'free': 10})()):
                with self.assertRaisesRegex(RuntimeError, 'GiB free'):
                    r.require_resources(Path(tmp), False)
            with patch.object(r.shutil, 'disk_usage', return_value=type('U', (), {'free': 10**13})()), \
                    patch.object(r.torch.cuda, 'mem_get_info', return_value=(1024**3, 48*1024**3)):
                with self.assertRaisesRegex(RuntimeError, 'GPU free'):
                    r.require_resources(Path(tmp), False)

    def test_snapshots_kept_are_one_plus_the_one_being_written(self):
        self.assertEqual((c.KEEP_SNAPSHOTS, c.SNAPSHOT_SLOTS), (1, 2))

    def test_snapshot_updates_come_from_the_decision(self):
        d = {**DECISION, 'snapshot_every': 30_225}
        self.assertEqual(c.snapshot_updates(False, d), [30_225, 60_450, 90_675])
        self.assertEqual(c.snapshot_updates(True, d), [1])


class FakeStrategy:
    def __init__(self):
        self.state = {'w': torch.arange(4.)}

    def model_state(self):
        return self.state

    def optimizer_state(self):
        return {'state': {0: {'step': torch.tensor(3.)}}, 'param_groups': [{'lr': 1e-3, 'params': [0]}]}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.binding = {'seal': 'x'}

    def save(self, update, report=None):
        report = report or {'completed_updates': update, 'binding': self.binding, 'checkpoints': []}
        return r.save_snapshot(FakeStrategy(), report, self.out, update, self.binding)

    def test_only_the_newest_snapshot_survives_and_never_more_than_two_exist_while_writing(self):
        seen = []
        real = r.write_snapshot_files

        def spy(strategy, report, tmp):
            seen.append(sorted(p.name for p in (self.out/'resume').iterdir()))
            return real(strategy, report, tmp)

        with patch.object(r, 'write_snapshot_files', spy):
            for update in (1, 2, 3):
                self.save(update)
                self.assertEqual([p.name for p in (self.out/'resume').glob('snap_*')], [f'snap_{update:06d}'])
        self.assertEqual(seen, [['tmp_000001'], ['snap_000001', 'tmp_000002'], ['snap_000002', 'tmp_000003']])
        self.assertTrue(all(len(names) <= c.SNAPSHOT_SLOTS for names in seen))

    def test_a_snapshot_loads_and_a_damaged_one_does_not(self):
        self.save(5)
        directory, meta, problems = r.load_snapshot(self.out, self.binding)
        self.assertEqual((directory.name, meta['update'], problems), ('snap_000005', 5, []))
        with (directory/'model.pt').open('ab') as stream:
            stream.write(b'x')
        with self.assertRaisesRegex(RuntimeError, 'No verifiable snapshot'):
            r.load_snapshot(self.out, self.binding)

    def test_a_snapshot_of_other_sources_is_refused(self):
        self.save(5)
        with self.assertRaisesRegex(ValueError, 'refusing to resume'):
            r.load_snapshot(self.out, {'seal': 'other'})

    def test_quarantine_moves_aside_and_trims_the_metrics_to_the_snapshot(self):
        self.save(2)
        (self.out/'resume'/'tmp_000009').mkdir()
        (self.out/'resume'/'snap_000004').mkdir()
        for name in ('step_000001', 'step_000003'):
            (self.out/'checkpoints'/name).mkdir(parents=True)
        (self.out/'result.json').write_text('{}')
        (self.out/'metrics.jsonl').write_text(''.join(json.dumps({'update': i})+'\n' for i in range(1, 6)))
        moved = r.quarantine(self.out, 1, {'checkpoints': [{'update': 1}]}, 2)
        self.assertEqual(sorted(moved), ['checkpoints/step_000003', 'metrics.jsonl', 'result.json',
                                         'resume/snap_000004', 'resume/tmp_000009'])
        self.assertEqual([json.loads(x)['update'] for x in (self.out/'metrics.jsonl').read_text().splitlines()], [1, 2])
        self.assertTrue((self.out/'checkpoints/step_000001').is_dir())
        self.assertTrue((self.out/'quarantine/resume_01/metrics.jsonl').is_file())

    def test_quarantine_refuses_metrics_that_do_not_cover_the_snapshot(self):
        self.save(2)
        (self.out/'metrics.jsonl').write_text(json.dumps({'update': 1})+'\n')
        with self.assertRaises(ValueError):
            r.quarantine(self.out, 1, {'checkpoints': []}, 2)


class ThermalTests(unittest.TestCase):
    def test_parse_and_decide(self):
        line = '71, 210.5, 300.00, Not Active, Not Active, 99\n'
        telemetry = r.parse_telemetry(line)
        self.assertEqual(telemetry, dict(temperature_c=71, power_w=210.5, power_limit_w=300.0, thermal_slowdown=False,
                                         utilization=99))
        self.assertFalse(r.too_hot(telemetry))
        self.assertTrue(r.too_hot({**telemetry, 'temperature_c': 83}))
        self.assertTrue(r.too_hot({**telemetry, 'thermal_slowdown': True}))
        self.assertFalse(r.too_hot(None))

    def test_cool_down_sleeps_until_cool_and_gives_up_after_the_limit(self):
        temps = iter([90, 80, 70])
        slept = []
        clock = iter(range(0, 10000, 1))
        seconds = r.cool_down(read=lambda: dict(temperature_c=next(temps), thermal_slowdown=False),
                              sleep=slept.append, clock=lambda: next(clock))
        self.assertEqual(len(slept), 2)
        self.assertIsNotNone(seconds)
        ticks = iter(range(0, 10**6, 700))
        self.assertIsNone(r.cool_down(read=lambda: dict(temperature_c=95, thermal_slowdown=True), sleep=lambda s: None,
                                      clock=lambda: next(ticks)))

    def test_pause_only_when_the_next_snapshot_cannot_finish_before_the_deadline(self):
        self.assertFalse(r.should_pause(100, None, 5, 1000))
        self.assertFalse(r.should_pause(100, 100+5*1000+r.SNAPSHOT_WRITE_S+1, 5, 1000))
        self.assertTrue(r.should_pause(100, 100+5*1000+r.SNAPSHOT_WRITE_S-1, 5, 1000))


def make_row(episode, frame=0):
    return {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
            'observation.state': torch.zeros(9), 'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512),
            'action': torch.full((50, 8), float(frame)), 'action_is_pad': torch.zeros(50, dtype=torch.bool)}


class FakeDataset:
    """Row i belongs to episode i // 452; the action carries i so order can be checked."""

    def __init__(self):
        self.reads = []

    def __getitem__(self, i):
        self.reads.append(i)
        row = make_row(i//FRAMES)
        row['action'] = torch.full((50, 8), float(i))
        return row


class DataStreamTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(c, 'planned_instructions', return_value=PLANNED)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_groups_split_into_microbatches_in_order(self):
        ds = FakeDataset()
        groups = [[0, 1, 2, 3, 4, 5, 6, 7], [452, 453, 454, 455, 456, 457, 458, 459]]
        for microbatch in (1, 2, 4, 8):
            fetched = list(r.serial_fetch(ds, iter(groups), microbatch))
            self.assertEqual([g for g, _ in fetched], groups)
            for (group, raws), expected in zip(fetched, groups):
                self.assertEqual(len(raws), 8//microbatch)
                self.assertEqual([int(a[0, 0]) for raw in raws for a in raw['action']], expected)
                self.assertTrue(all(raw['action'].shape == (microbatch, 50, 8) for raw in raws))

    def test_prefetcher_preserves_the_samplers_order_and_reraises_worker_errors(self):
        groups = [[i*8+k for k in range(8)] for i in range(5)]
        fetch = r.Prefetcher(r.LockedDataset(FakeDataset()), iter(groups), 4)
        try:
            self.assertEqual([next(fetch)[0] for _ in range(5)], groups)
            with self.assertRaises(StopIteration):
                next(fetch)
        finally:
            fetch.close()
        bad = r.Prefetcher(r.LockedDataset(FakeDataset()), iter([[0, 1, 2, 3, 4, 5, 6, 7], [480*FRAMES]]), 4)
        try:
            next(bad)
            with self.assertRaises(ValueError):
                next(bad)
        finally:
            bad.close()

    def test_the_prefetcher_does_not_run_ahead_of_its_depth(self):
        ds = FakeDataset()
        fetch = r.Prefetcher(r.LockedDataset(ds), iter([[i*8+k for k in range(8)] for i in range(50)]), 8, depth=2)
        try:
            time.sleep(0.5)
            self.assertLessEqual(len(ds.reads), 8*(2+2))
        finally:
            fetch.close()

    def test_rows_outside_the_training_episodes_are_refused(self):
        with self.assertRaises(ValueError):
            c.collate_rows([make_row(480)])
        with self.assertRaises(ValueError):
            c.collate_rows([])
        with self.assertRaises(ValueError):
            c.collate_rows([make_row(0)]*9)


class SavedWeightsTests(unittest.TestCase):
    def test_saved_tensors_must_equal_the_live_ones_and_omitted_names_must_be_ties(self):
        module = torch.nn.Module()
        module.a = torch.nn.Linear(3, 3)
        module.b = torch.nn.Linear(3, 3)
        module.b.weight = module.a.weight
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'w.safetensors'
            state = {k: v.detach().clone() for k, v in module.state_dict().items() if k != 'b.weight'}
            save_file(state, str(path))
            self.assertEqual(r.verify_saved_weights(module, path), 3)
            state['a.bias'] = state['a.bias'] + 1e-6
            save_file(state, str(path))
            with self.assertRaisesRegex(ValueError, 'differs'):
                r.verify_saved_weights(module, path)
            save_file({k: v for k, v in module.state_dict().items() if k not in ('b.weight', 'a.weight')}, str(path))
            with self.assertRaisesRegex(ValueError, 'lacks'):
                r.verify_saved_weights(module, path)
            save_file({k: v.to(torch.bfloat16) for k, v in module.state_dict().items() if k != 'b.weight'}, str(path))
            with self.assertRaisesRegex(ValueError, 'differs'):
                r.verify_saved_weights(module, path)


class ProjectionTests(unittest.TestCase):
    def test_projection_uses_the_measured_pace_and_sizes(self):
        sizes = dict(checkpoint_bytes=7_233_582_936, snapshot_bytes=43_000_000_000, final_state_bytes=29_000_000_000)
        p = r.project('bf16_cpu_master', 8, 3.6, sizes, 30*1024**3)
        self.assertAlmostEqual(p['raw_hours'], 3.6*c.TOTAL_UPDATES/3600)
        self.assertEqual(p['disk_peak_bytes'], 7*sizes['checkpoint_bytes'] + sizes['final_state_bytes']
                         + 2*sizes['snapshot_bytes'] + 2*1024**3)
        self.assertEqual(p['disk_at_end_bytes'], 7*sizes['checkpoint_bytes'] + sizes['final_state_bytes']
                         + sizes['snapshot_bytes'])
        self.assertEqual(p['gate_artifacts_bytes'], 2*(2*sizes['checkpoint_bytes'] + sizes['final_state_bytes']
                                                       + sizes['snapshot_bytes']))
        self.assertEqual(p['assets_bytes'], sum(size for size, _ in r.assets.FILES.values()))


class ProbeMeasurementTests(unittest.TestCase):
    def test_changed_fractions_at_peak_and_floor_learning_rate_on_a_bf16_model(self):
        from experiments.pi05_props_p2.tests.test_strategy import Tiny, batches, masters_of
        from experiments.pi05_props_p2 import strategy as st
        model = Tiny()
        strategy = st.make_strategy('bf16_all', model, masters_of(model), 'cpu')
        counter = [0]

        def step(lr):
            counter[0] += 1
            strategy.zero_grad()
            model(batches()[0]).backward()
            strategy.step(counter[0]+20, lr)

        for _ in range(3):
            step(1e-3)
        peak = r.measure_changes(strategy, step, 'peak_lr', 1e-2)
        floor = r.measure_changes(strategy, step, 'floor_lr', 1e-7)
        self.assertEqual(peak['label'], 'peak_lr')
        self.assertEqual(set(peak), {'label', 'forced_learning_rate', 'view', 'overall', 'by_group', 'by_dtype',
                                     'per_tensor'})
        self.assertEqual({row['name'] for row in peak['per_tensor']}, set(strategy.params))
        self.assertGreater(peak['by_dtype']['torch.bfloat16']['element_weighted'],
                           floor['by_dtype']['torch.bfloat16']['element_weighted'])
        self.assertEqual(floor['by_dtype']['torch.bfloat16']['element_weighted'], 0.0)
        self.assertGreater(floor['by_dtype']['torch.float32']['element_weighted'], 0.0)

    def test_cpu_master_reports_its_fp32_masters_as_moving_when_the_bf16_view_does_not(self):
        from experiments.pi05_props_p2.tests.test_strategy import Tiny, batches, masters_of
        from experiments.pi05_props_p2 import strategy as st
        model = Tiny()
        strategy = st.CpuMasterAdam(model, masters_of(model), fused=False)

        def step(lr):
            strategy.zero_grad()
            model(batches()[0]).backward()
            strategy.step(30, lr)

        result = r.measure_changes(strategy, step, 'floor_lr', 1e-7)
        self.assertEqual(result['by_dtype']['torch.bfloat16']['element_weighted'], 0.0)
        self.assertEqual(result['fp32_masters_small_tensors']['tensors'], 6)
        self.assertGreater(result['fp32_masters_small_tensors']['element_weighted'], 0.9)

    def test_sizes_are_written_measured_and_removed(self):
        class FakePolicy:
            def save_pretrained(self, path):
                path = Path(path)
                path.mkdir()
                save_file({'w': torch.zeros(1000)}, str(path/'model.safetensors'))
                (path/'config.json').write_text('{}')

        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)/'sizes'
            sizes = r.measure_sizes(FakePolicy(), FakeStrategy(), scratch)
            self.assertFalse(scratch.exists())
        self.assertEqual(sizes['checkpoint_bytes'], sizes['checkpoint_files']['model.safetensors'])
        self.assertGreater(sizes['checkpoint_bytes'], 4000)
        self.assertEqual(sizes['snapshot_bytes'], sum(sizes['snapshot_files'].values()))
        self.assertEqual(sizes['final_state_bytes'], sizes['snapshot_files']['optimizer.pt'])
        self.assertEqual(set(sizes['snapshot_files']), set(r.SNAPSHOT_FILES))

    def test_tensor_groups(self):
        self.assertEqual(r.tensor_group('model.paligemma_with_expert.paligemma.model.vision_tower.x'), 'vision')
        self.assertEqual(r.tensor_group('model.paligemma_with_expert.paligemma.model.language_model.layers.0.x'), 'language')
        self.assertEqual(r.tensor_group('model.paligemma_with_expert.gemma_expert.model.layers.0.x'), 'expert')
        self.assertEqual(r.tensor_group('model.action_in_proj.weight'), 'projections')


if __name__ == '__main__':
    unittest.main()
