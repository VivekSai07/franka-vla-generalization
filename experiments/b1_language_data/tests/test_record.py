"""Broad-workspace scheduling: expert subclass, feasibility gate, GraspController regression."""
import copy
import hashlib
import importlib
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np


class ExpertSubclassTests(unittest.TestCase):
    def setUp(self):
        try:
            self.mod = importlib.import_module('experiments.b1_language_data.expert')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing expert module: {exc}')

    def test_version_string(self):
        e = self.mod.Expert()
        self.assertEqual(e.config['expert_version'], 'paired_scripted_expert_broad_workspace_v1')

    def test_apply_hover_offset_only_modifies_two_states(self):
        from data_collection.collect_libero_dataset import generate_task_states
        original = generate_task_states(.105, robot_speed=1.)
        modified = self.mod._apply_hover_offset(original, dict(pre_grasp=0.01, pre_place=-0.02))
        self.assertEqual([s['name'] for s in modified], [s['name'] for s in original])
        for original_state, mod_state in zip(original, modified):
            if original_state['name'] == 'Approach Pick':
                self.assertAlmostEqual(mod_state['offset'][2], original_state['offset'][2] + 0.01)
                self.assertAlmostEqual(mod_state['offset'][0], original_state['offset'][0])
                self.assertAlmostEqual(mod_state['offset'][1], original_state['offset'][1])
            elif original_state['name'] == 'Move to Place':
                self.assertAlmostEqual(mod_state['offset'][2], original_state['offset'][2] - 0.02)
            else:
                np.testing.assert_array_equal(mod_state['offset'], original_state['offset'])
            for field in ('name', 'phase', 'gripper', 'duration', 'jitter'):
                self.assertEqual(mod_state[field], original_state[field])
        # Original list not mutated:
        original2 = generate_task_states(.105, robot_speed=1.)
        for a, b in zip(original, original2):
            np.testing.assert_array_equal(a['offset'], b['offset'])

    def test_run_refuses_malformed_hover_offset(self):
        from experiments.pose_snapshot_v2.expert import Expert as Legacy
        with patch.object(Legacy, '__init__', return_value=None):
            e = self.mod.Expert.__new__(self.mod.Expert)
            e.config = {'expert_version': 'paired_scripted_expert_broad_workspace_v1'}
            for bad in ({'pre_grasp': .01},                          # missing key
                        {'pre_grasp': .01, 'pre_place': .01, 'x': 0.}, # extra key
                        {'pre_grasp': .05, 'pre_place': 0.},          # out of range
                        {'pre_grasp': float('nan'), 'pre_place': 0.},  # nonfinite
                        {'pre_grasp': 0, 'pre_place': 0.}):            # wrong type
                episode = dict(instruction='pick up the red cube and place it on top of the green cube',
                               pick='red', place='green', hover_z_offset=bad,
                               home_pose_jitter_deg=[0.0]*7, seed=0)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    e.run(np.zeros(1), episode, lambda f, t: None)

    def test_run_refuses_malformed_home_jitter(self):
        e = self.mod.Expert.__new__(self.mod.Expert)
        e.config = {'expert_version': 'paired_scripted_expert_broad_workspace_v1'}
        for bad in ([0.0]*6, [0.0]*8, [1.5]+[0.0]*6, [float('nan')]+[0.0]*6, tuple([0.0]*7), [0]*7):
            episode = dict(instruction='pick up the red cube and place it on top of the green cube',
                           pick='red', place='green',
                           hover_z_offset={'pre_grasp': 0.0, 'pre_place': 0.0},
                           home_pose_jitter_deg=bad, seed=0)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                e.run(np.zeros(1), episode, lambda f, t: None)


class FeasibilityGateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.r = importlib.import_module('experiments.b1_language_data.record')
            self.p = importlib.import_module('experiments.b1_language_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing collector: {exc}')

    def _bank(self, plan):
        b = {f"episode_{e['episode_index']:03d}": {'initial_state': np.zeros(242)} for e in plan['episodes']}
        b['base'] = {'initial_state': np.zeros(242)}
        for lid in plan['evaluation_layout_ids']:
            b[lid] = {'initial_state': np.zeros(242)}
        return b

    def test_redraw_values_deterministic(self):
        first = self.r.redraw_home_pose_jitter(3, 0)
        second = self.r.redraw_home_pose_jitter(3, 0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, self.r.redraw_home_pose_jitter(3, 1))
        self.assertEqual(len(first), 7)
        for v in first:
            self.assertLessEqual(abs(v), self.p.HOME_JITTER_DEG_MAX + 1e-12)

    def test_feasible_first_try_no_redraws(self):
        plan = self.p.make_plan(); bank = self._bank(plan)
        calls = []
        def feasible(layout, jitter):
            calls.append(tuple(jitter['home_pose_jitter_deg'])); return True
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            admitted, rejected = self.r.run_schedule(
                plan, base_source=None, model=None,
                execute=lambda slot, arr: {'success': True, 'frames': 528},
                validate=lambda slot, out: None,
                journal=events.append, feasible_fn=feasible)
        self.assertEqual(len(admitted), 200)
        self.assertEqual(len(rejected), 0)
        # Feasible on first try, so exactly 200 feasibility calls.
        self.assertEqual(len(calls), 200)

    def test_completed_slots_are_skipped_and_still_counted_as_admitted(self):
        """Resume must not re-execute admitted slots, and must keep admits contiguous."""
        plan = self.p.make_plan()
        ran = []
        def execute(slot, arrays):
            ran.append(slot['episode_index'])
            return {'success': True, 'frames': 528}
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            admitted, rejected = self.r.run_schedule(
                plan, base_source=None, model=None, execute=execute,
                validate=lambda s, o: None, journal=events.append,
                feasible_fn=lambda l, j: True, completed=frozenset(range(180)))
        self.assertEqual(ran, list(range(180, self.p.ADMIT_COUNT)))
        self.assertEqual(admitted, list(range(self.p.ADMIT_COUNT)))
        self.assertEqual([e['episode_index'] for e in events if e['event'] == 'started'],
                         list(range(180, self.p.ADMIT_COUNT)))

    def test_infeasible_layout_resample_only_moves_recorded_layouts(self):
        """The substitution must be deterministic and must not disturb any episode slot.

        validation_037 (slot 197) exhausted every feasible draw during collection, so it is the
        one recorded substitution; nothing else may have moved.
        """
        plan = self.p.make_plan()
        self.assertEqual(self.p.INFEASIBLE_LAYOUT_RESAMPLES, {'validation_037': 0})
        layouts = {x['layout_id']: x for x in plan['layouts']}
        for lid, layout in layouts.items():
            if lid == 'base':
                continue
            self.assertIn('region_xy_m', layout)
        # The substituted layout carries its resampled draw, not its Latin-hypercube one.
        self.assertEqual(layouts['validation_037']['region_xy_m'],
                         self.p.resample_layout_xy('validation_037', 0))
        first = self.p.resample_layout_xy('validation_020', 2)
        self.assertEqual(self.p.resample_layout_xy('validation_020', 2), first)  # deterministic
        self.assertNotEqual(self.p.resample_layout_xy('validation_020', 1), first)
        for axis, value in enumerate(first):  # stays inside envelope F7
            self.assertGreaterEqual(value, self.p.AXIS_LOW[axis])
            self.assertLessEqual(value, self.p.AXIS_HIGH[axis])
        self.assertEqual(plan['infeasible_layout_resamples'], {'validation_037': 0})
        self.p.validate_plan(plan)

    def test_only_the_substituted_layout_moves(self):
        """Every other layout must keep the draw the 197 collected episodes were recorded under."""
        import unittest.mock as mock
        plan = self.p.make_plan()
        with mock.patch.object(self.p, 'INFEASIBLE_LAYOUT_RESAMPLES', {}):
            baseline = self.p.make_plan()
        moved = {x['layout_id'] for x, y in zip(plan['layouts'], baseline['layouts']) if x != y}
        self.assertEqual(moved, {'validation_037'})
        self.assertEqual(plan['episodes'], baseline['episodes'])

    def test_infeasible_slot_exhausting_every_redraw_stops(self):
        """A slot with no feasible draw cannot be skipped; admit index must track the plan slot."""
        plan = self.p.make_plan()
        counter = {'checks': 0}
        def feasible(layout, jitter):
            counter['checks'] += 1
            # First episode always infeasible; subsequent episodes feasible on first try.
            return counter['checks'] > 9
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError) as caught:
                self.r.run_schedule(plan, base_source=None, model=None,
                                    execute=lambda s, a: {'success': True, 'frames': 528},
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=feasible)
        self.assertIn('exhausted every feasible draw', str(caught.exception))
        self.assertEqual(counter['checks'], 9)
        failed = [e for e in events if e['event'] == 'failed']
        self.assertEqual([e['episode_index'] for e in failed], [0])

    def test_rejection_budget_stops_beyond_101(self):
        plan = self.p.make_plan()
        def feasible(layout, jitter): return False
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError):
                self.r.run_schedule(plan, base_source=None, model=None,
                                    execute=lambda s, a: {'success': True, 'frames': 528},
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=feasible)
        rejected_ids = [e['episode_index'] for e in events if e['event'] == 'rejected']
        self.assertLessEqual(len(rejected_ids), self.p.REJECTION_BUDGET)

    def test_expert_execution_failure_retries_same_slot_until_admitted(self):
        """Post-feasibility expert failures retry the SAME slot; every slot still yields one admit."""
        plan = self.p.make_plan()
        # Slot 0 rejects on its first two draws, then succeeds on the third.
        state = {'calls': 0}
        def execute(slot, arrays):
            state['calls'] += 1
            if state['calls'] <= 2:
                raise self.r.ExpertRejection('Failed expert or noncanonical duration; raw attempt preserved')
            return {'success': True, 'frames': 528}
        events, discarded = [], []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            admitted, rejected = self.r.run_schedule(
                plan, base_source=None, model=None, execute=execute,
                validate=lambda s, o: None, journal=events.append, feasible_fn=lambda l, j: True,
                discard=lambda index, k, committed=False: discarded.append((index, k)))
        expert_rejects = [e for e in events if e.get('event') == 'rejected' and e.get('reason') == 'expert_execution']
        self.assertEqual([e['episode_index'] for e in expert_rejects], [0, 0])
        self.assertEqual([e['tried'] for e in expert_rejects], [1, 2])
        # The rejected attempt directories are retired so the retry can reuse the slot path.
        self.assertEqual(discarded, [(0, 0), (0, 1)])
        # Crucially: no slot is skipped, so admit index stays equal to plan slot index.
        self.assertEqual(admitted, list(range(self.p.ADMIT_COUNT)))
        self.assertEqual(rejected, [0, 0])
        # The admitted attempt used the third draw, and its journal entry carries that draw.
        started = [e for e in events if e['event'] == 'started' and e['episode_index'] == 0]
        self.assertEqual([e['feasibility_k'] for e in started], [0, 1, 2])

    def test_rejections_keep_admitted_indices_contiguous_and_journal_valid(self):
        """Regression: the seam that broke production three times at slot 52.

        A rejection must not shift later episodes. Admitted ids must stay 0..N-1 so that
        metadata[i], attempts/{i:06d} and resets/episode_{i:03d} keep pointing at one episode,
        and the resulting journal must satisfy the final collection validator.
        """
        from experiments.b1_language_data import validate as v
        plan = self.p.make_plan()
        failing = {52, 53, 140}
        seen = set()
        def execute(slot, arrays):
            index = slot['episode_index']
            if index in failing and index not in seen:
                seen.add(index)
                raise self.r.ExpertRejection('Failed expert or noncanonical duration; raw attempt preserved')
            return {'success': True, 'frames': 528}
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            admitted, rejected = self.r.run_schedule(
                plan, base_source=None, model=None, execute=execute,
                validate=lambda s, o: None, journal=events.append, feasible_fn=lambda l, j: True)
        self.assertEqual(admitted, list(range(self.p.ADMIT_COUNT)))
        self.assertEqual(sorted(rejected), sorted(failing))
        # Every slot is admitted exactly once, in plan order.
        admits = [e['episode_index'] for e in events if e['event'] == 'admitted']
        self.assertEqual(admits, list(range(self.p.ADMIT_COUNT)))
        # The retried slots journal more than one start but only one admit.
        for index in failing:
            starts = [e for e in events if e['event'] == 'started' and e['episode_index'] == index]
            self.assertEqual(len(starts), 2)
            self.assertEqual(admits.count(index), 1)
        enriched = [{**e, 'elapsed_seconds': float(i)} for i, e in enumerate(events)]
        v.validate_journal(enriched, plan)

    def test_expert_execution_failures_exhaust_rejection_budget_and_stop(self):
        """One rejection on each of many slots eventually exhausts the shared 100-rejection budget."""
        plan = self.p.make_plan()
        seen = set()
        def execute(slot, arrays):
            index = slot['episode_index']
            if index not in seen:  # every slot rejects once, then succeeds on its redraw
                seen.add(index)
                raise self.r.ExpertRejection('Failed expert or noncanonical duration; raw attempt preserved')
            return {'success': True, 'frames': 528}
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError) as caught:
                self.r.run_schedule(plan, base_source=None, model=None, execute=execute,
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=lambda l, j: True)
        self.assertIn('Rejection budget exhausted', str(caught.exception))
        expert_rejects = [e for e in events if e.get('event') == 'rejected' and e.get('reason') == 'expert_execution']
        # The over-budget rejection raises before it is journalled, so the budget count is recorded.
        self.assertEqual(len(expert_rejects), self.p.REJECTION_BUDGET)

    def test_slot_rejected_on_every_draw_stops_instead_of_skipping(self):
        """A slot that never succeeds must stop collection rather than leave a gap in the admits."""
        plan = self.p.make_plan()
        def execute(slot, arrays):
            raise self.r.ExpertRejection('Failed expert or noncanonical duration; raw attempt preserved')
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError) as caught:
                self.r.run_schedule(plan, base_source=None, model=None, execute=execute,
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=lambda l, j: True)
        self.assertIn('exhausted every feasible draw', str(caught.exception))
        rejects = [e for e in events if e.get('event') == 'rejected']
        self.assertEqual(len(rejects), self.r.FEASIBILITY_REDRAWS + 1)
        self.assertEqual({e['episode_index'] for e in rejects}, {0})

    def test_non_expert_runtime_errors_still_stop(self):
        """Infrastructure RuntimeErrors (physics NaN, disk full, etc.) MUST still stop the collection."""
        plan = self.p.make_plan()
        def execute(slot, arrays):
            raise RuntimeError('Physics simulation produced NaN state')
        events = []
        with patch('experiments.b1_language_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError) as caught:
                self.r.run_schedule(plan, base_source=None, model=None, execute=execute,
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=lambda l, j: True)
        self.assertIn('Physics simulation produced NaN', str(caught.exception))
        failed_events = [e for e in events if e.get('event') == 'failed']
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0]['episode_index'], 0)


class GraspControllerRegressionTests(unittest.TestCase):
    """Ensure b1_language_data does NOT modify GraspController."""

    GRASP_CONTROLLER_HEAD_SHA256 = 'd9da36d49f8b43f43c81c1a4a26bf58415b0c0cb7056bc0dc62b626cd88470d9'

    def test_grasp_controller_file_hash_unchanged(self):
        path = Path('/vla/src/controllers/grasp_controller.py')
        current = path.read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(), self.GRASP_CONTROLLER_HEAD_SHA256)

    def test_grasp_controller_has_no_hover_z_offset_argument(self):
        from src.controllers.grasp_controller import GraspController
        import inspect
        params = inspect.signature(GraspController.__init__).parameters
        self.assertNotIn('hover_z_offset', params)
        self.assertEqual(list(params), ['self', 'model', 'data', 'left_finger', 'right_finger', 'target'])


class NoForbiddenImportsTest(unittest.TestCase):
    def test_no_forbidden_module_imports(self):
        package = Path('/home/utxowqr/smolvla_franka/experiments/b1_language_data')
        forbidden = ('from experiments.smolvla_local_varied',
                     'from experiments.fixed_layout.validate import require_fixed_layout_pass',
                     'require_fixed_layout_pass')
        for path in package.rglob('*.py'):
            text = path.read_text()
            for token in forbidden:
                self.assertNotIn(token, text, f'{path} contains forbidden {token!r}')


if __name__ == '__main__':
    unittest.main()
