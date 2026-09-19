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
            self.mod = importlib.import_module('experiments.broad_workspace_data.expert')
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
            self.r = importlib.import_module('experiments.broad_workspace_data.record')
            self.p = importlib.import_module('experiments.broad_workspace_data.plan')
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
        with patch('experiments.broad_workspace_data.record.derive_scene',
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

    def test_infeasible_triggers_8_redraws_then_reject(self):
        plan = self.p.make_plan()
        counter = {'checks': 0}
        def feasible(layout, jitter):
            counter['checks'] += 1
            # First episode always infeasible; subsequent episodes feasible on first try.
            return counter['checks'] > 9
        events = []
        with patch('experiments.broad_workspace_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            self.r.run_schedule(plan, base_source=None, model=None,
                                execute=lambda s, a: {'success': True, 'frames': 528},
                                validate=lambda s, o: None, journal=events.append,
                                feasible_fn=feasible)
        rejected = [e for e in events if e['event'] == 'rejected']
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['episode_index'], 0)
        self.assertEqual(rejected[0]['tried'], 9)

    def test_rejection_budget_stops_beyond_101(self):
        plan = self.p.make_plan()
        def feasible(layout, jitter): return False
        events = []
        with patch('experiments.broad_workspace_data.record.derive_scene',
                   return_value={'initial_state': np.zeros(242), 'qpos': np.zeros(37)}):
            with self.assertRaises(RuntimeError):
                self.r.run_schedule(plan, base_source=None, model=None,
                                    execute=lambda s, a: {'success': True, 'frames': 528},
                                    validate=lambda s, o: None, journal=events.append,
                                    feasible_fn=feasible)
        rejected_ids = [e['episode_index'] for e in events if e['event'] == 'rejected']
        self.assertLessEqual(len(rejected_ids), self.p.REJECTION_BUDGET)


class GraspControllerRegressionTests(unittest.TestCase):
    """Ensure broad_workspace_data does NOT modify GraspController."""

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
        package = Path('/home/utxowqr/smolvla_franka/experiments/broad_workspace_data')
        forbidden = ('from experiments.smolvla_local_varied',
                     'from experiments.fixed_layout.validate import require_fixed_layout_pass',
                     'require_fixed_layout_pass')
        for path in package.rglob('*.py'):
            text = path.read_text()
            for token in forbidden:
                self.assertNotIn(token, text, f'{path} contains forbidden {token!r}')


if __name__ == '__main__':
    unittest.main()
