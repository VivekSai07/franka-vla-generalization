"""Scheduler/claim/resource boundaries, no actual expert or physics."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class RecordTests(unittest.TestCase):
    def setUp(self):
        try:
            self.r = importlib.import_module('experiments.local_varied_data.record')
            self.p = importlib.import_module('experiments.local_varied_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing collector: {exc}')

    def bank(self, plan):
        return {x['layout_id']: {'initial_state': np.full(242, i, np.float64)} for i, x in enumerate(plan['layouts'])}

    def test_60_ordered_calls_first_12_gate_and_validation_before_13(self):
        p = self.p.make_plan(); bank = self.bank(p); calls = []; events = []; admitted = []
        def execute(slot, initial):
            i = slot['episode_index']
            self.assertEqual(admitted, list(range(i)))
            self.assertEqual(len(bank), 77)
            self.assertEqual(initial.tobytes(), bank[slot['layout_id']]['initial_state'].tobytes())
            if i == 12: self.assertEqual(events[-2]['event'], 'preparation_pass')
            calls.append(slot.copy()); initial[:] = -1; slot['seed'] = -1
            return {'success': True, 'frames': 528}
        result = self.r.run_schedule(p, bank, execute, lambda s, o: admitted.append(s['episode_index']), events.append)
        self.assertEqual(result, list(range(60)))
        self.assertEqual(calls, p['episodes'])
        self.assertEqual(sum(e['event'] == 'preparation_pass' for e in events), 1)
        self.assertFalse(set(e['layout_id'] for e in calls) & set(p['evaluation_layout_ids']))

    def test_first_failure_no_retry_or_later_execution(self):
        for stage in ('expert', 'validation', 'before'):
            p = self.p.make_plan(); calls = []; events = []
            def execute(slot, state):
                calls.append(slot['episode_index'])
                return {'success': stage != 'expert', 'frames': 528}
            def validate(*args):
                if stage == 'validation': raise ValueError('gate')
            def before():
                if stage == 'before': raise ValueError('resource')
            with self.subTest(stage=stage), self.assertRaises((ValueError, RuntimeError)):
                self.r.run_schedule(p, self.bank(p), execute, validate, events.append, before)
            self.assertEqual(calls, [] if stage == 'before' else [0])
            self.assertEqual(events[-1]['event'], 'failed')

    def test_bank_or_plan_mutation_rejected_before_next_call(self):
        for target in ('bank', 'plan'):
            p = self.p.make_plan(); bank = self.bank(p); calls = []
            def execute(slot, state):
                calls.append(slot['episode_index'])
                if target == 'bank': bank[p['evaluation_layout_ids'][0]]['initial_state'][0] += 1
                else: p['episodes'][1]['seed'] += 1
                return {'success': True, 'frames': 528}
            with self.assertRaises(ValueError): self.r.run_schedule(p, bank, execute, lambda *a: None, lambda e: None)
            self.assertEqual(calls, [0])

    def test_empty_bind_mount_claim_exclusive_nonempty_and_symlink_fail(self):
        root = Path(tempfile.mkdtemp())
        self.r.claim_output(root)
        self.assertTrue((root / 'claim.json').is_file())
        with self.assertRaises(FileExistsError): self.r.claim_output(root)
        other = Path(tempfile.mkdtemp()); (other / 'old-evidence').write_text('preserve')
        with self.assertRaises(FileExistsError): self.r.claim_output(other)
        link = other / 'link'; link.symlink_to(root)
        with self.assertRaises(ValueError): self.r.claim_output(link)

    def test_disk_start_floor_deadline_and_no_override(self):
        with patch.object(self.r, 'disk_usage') as disk, patch.object(self.r, 'monotonic', return_value=10) as clock:
            disk.return_value.free = 40*1024**3-1
            b = self.r.Budget(Path('/tmp'))
            with self.assertRaises(ValueError): b.check_start()
            disk.return_value.free = 40*1024**3; b.check_start()
            disk.return_value.free = 30*1024**3; b.check()
            disk.return_value.free -= 1
            with self.assertRaises(ValueError): b.check()
            disk.return_value.free = 40*1024**3; clock.return_value = 10+21600
            with self.assertRaises(TimeoutError): b.check()
        with self.assertRaises(TypeError): self.r.collect(root='/tmp/override')

    def test_cli_rejects_overrides(self):
        for args in (['--collect', '--resume'], ['--collect', '--root', '/tmp/x'], ['--collect', '--seed', '1'], ['--collect', '--preflight']):
            with patch('sys.argv', ['record', *args]), self.assertRaises(SystemExit) as e: self.r.main()
            self.assertEqual(e.exception.code, 2)

    def test_collect_freezes_every_state_before_fake_attempt_and_preserves_primary_failure(self):
        from experiments.fixed_layout.source import Expert
        from experiments.fixed_layout.record import TickSink
        import mujoco
        from contextlib import ExitStack
        root = Path(tempfile.mkdtemp(prefix='failure-', dir='/vla/outputs/implementation/local_varied_data_20260919'))
        real = Expert(); calls = []; error = RuntimeError('synthetic-first-failure')
        class FakeExpert:
            model, config = real.model, real.config
            def run(self, initial, slot, sink):
                calls.append(slot['episode_index'])
                self_plan = json.loads((root / 'plan.json').read_text())
                if len(list((root / 'scenes').glob('*.npz'))) != 77 or len(self_plan['policy_seeds']) != 3:
                    raise AssertionError('Expert reached before complete freeze')
                sink({'action': np.zeros(8, np.float32)}, {'qpos': initial.copy()})
                raise error
        class FakeDataset:
            def add_frame(self, frame): pass
            def finalize(self): raise OSError('synthetic footer failure')
        with ExitStack() as stack:
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No native execution')))
            stack.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('No real expert')))
            stack.enter_context(patch.object(self.r, 'OUTPUT', root))
            stack.enter_context(patch.object(self.r, 'Expert', return_value=FakeExpert()))
            stack.enter_context(patch.object(self.r, 'create_dataset', return_value=FakeDataset()))
            with self.assertRaises(RuntimeError) as caught: self.r.collect()
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [0])
        failure = json.loads((root / 'failure.json').read_text())
        self.assertEqual(failure['attempted_ids'], [0]); self.assertEqual(failure['missing_ids'], list(range(1, 60)))
        self.assertFalse(failure['training_ready'])
        self.assertTrue((root / 'attempts/000000/ticks/000000.npz').is_file())
        self.assertTrue((root / 'attempts/000000/interrupted.json').is_file())
        self.assertTrue((root / 'attempts/000000/finalization-error.json').is_file())
        self.assertFalse((root / 'seal.json').exists())
        with self.assertRaises(FileExistsError), patch.object(self.r, 'OUTPUT', root): self.r.collect()


if __name__ == '__main__': unittest.main()