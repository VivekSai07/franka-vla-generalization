"""Numerical tests: no simulation stepping and no saved-case route solving."""
from types import SimpleNamespace
import unittest
import numpy as np
import pinocchio as pin

from experiments.bounded_placement_v3.solver import PlanningFailure, solve_waypoint
from experiments.bounded_placement_v3.path import plan_route as fixed_route
try:
    from experiments.adaptive_placement_v4.planner import plan_route, Dependencies, Limits
except ImportError:
    plan_route, Dependencies, Limits = fixed_route, None, None


class ToyRobot:
    """A real linear FK/Jacobian with poor Cartesian-to-joint conditioning."""
    model = SimpleNamespace(nq=7)

    def forward_kinematics(self, q):
        return pin.SE3(np.eye(3), np.array([.04*q[0], 0., 0.]))

    def jacobian(self, q):
        J = np.zeros((6, 7))
        J[0, 0] = .04
        return J


class ClearChecker:
    def __init__(self):
        self.checked, self.journal = 0, []

    def segment(self, start, end, final_lowering, carrying=True):
        self.checked += 1
        self.journal.append((start.copy(), end.copy()))


class PlannerTests(unittest.TestCase):
    def route(self, endpoint=None, **kw):
        journal, checker = {}, ClearChecker()
        if endpoint is None:
            endpoint = pin.SE3(np.eye(3), np.array([.004, 0., 0.]))
        result = plan_route(ToyRobot(), np.zeros(7), np.zeros(7), [endpoint],
                            np.tile([-2., 2.], (7, 1)), checker, [1], 1000, journal, **kw)
        return result, journal, checker

    def test_solvable_large_joint_parent_is_split_not_accepted(self):
        try:
            schedule, journal, checker = self.route()
        except PlanningFailure as exc:
            self.fail(f'solvable path must subdivide instead of {exc.gate}')
        attempts = journal['attempts']
        # The right half needs a second split: its two IK updates total .05899
        # rad. Count that rejected attempt too; residual tolerance is unchanged.
        self.assertEqual(len(attempts), 5)
        self.assertEqual([a['status'] for a in attempts],
                 ['subdivided', 'accepted', 'subdivided', 'accepted', 'accepted'])
        self.assertGreater(attempts[0]['joint_step_rad'], .05)
        self.assertTrue(all(a['joint_step_rad'] <= .05 for a in attempts if a['status'] == 'accepted'))
        self.assertEqual(journal['budgets']['attempts'], 5)
        np.testing.assert_array_equal(attempts[-1]['pose'], pin.SE3(np.eye(3), np.array([.004, 0., 0.])).homogeneous)
        self.assertLessEqual(max(abs(v[1][0]-v[0][0]) for v in checker.journal), .05)
        self.assertTrue(schedule)

    def test_unreachable_real_solver_stops_without_retry(self):
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(ToyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([0., .004, 0.]))], np.tile([-2., 2.], (7, 1)),
                ClearChecker(), [1], 1000, journal)
        self.assertEqual(raised.exception.gate, 'ik_backtracking_failure')
        self.assertEqual(len(journal.get('attempts', [])), 1)
        self.assertEqual(journal['budgets']['iterations'], 1)
        self.assertEqual(journal['attempts'][0]['status'], 'failed')

    def test_attempt_limit_counts_rejected_parent(self):
        self.assertIsNotNone(Limits, 'bounded accounting is missing')
        with self.assertRaises(PlanningFailure) as raised:
            self.route(limits=Limits(attempts=2))
        self.assertEqual(raised.exception.gate, 'waypoint_attempt_limit')

    def test_iteration_budget_stops_before_unaccounted_update(self):
        self.assertIsNotNone(Limits, 'bounded accounting is missing')
        with self.assertRaises(PlanningFailure) as raised:
            self.route(limits=Limits(iterations=1))
        self.assertEqual(raised.exception.gate, 'total_ik_iteration_limit')

    def test_depth_stop_preserves_original_failure(self):
        self.assertIsNotNone(Limits, 'bounded accounting is missing')
        with self.assertRaises(PlanningFailure) as raised:
            self.route(limits=Limits(depth=0))
        self.assertEqual(raised.exception.gate, 'subdivision_depth_limit')

    def test_solver_nonfinite_q_is_rejected_before_fk(self):
        self.assertIsNotNone(Dependencies, 'explicit dependencies are missing')
        def invalid(*args):
            return np.full(7, np.nan)
        with self.assertRaises(PlanningFailure) as raised:
            self.route(dependencies=Dependencies(solve=invalid))
        self.assertEqual(raised.exception.gate, 'nonfinite')

    def test_bad_residual_does_not_trigger_subdivision(self):
        self.assertIsNotNone(Dependencies, 'explicit dependencies are missing')
        with self.assertRaises(PlanningFailure) as raised:
            self.route(dependencies=Dependencies(solve=lambda robot, seed, *args: seed.copy()))
        self.assertEqual(raised.exception.gate, 'independent_fk_residual')

    def test_input_endpoint_not_mutated(self):
        target = pin.SE3(np.eye(3), np.array([.004, 0., 0.]))
        original = target.homogeneous.copy()
        try:
            self.route(target)
        except PlanningFailure as exc:
            self.fail(f'solvable path must subdivide instead of {exc.gate}')
        np.testing.assert_array_equal(target.homogeneous, original)

    def test_all_failed_attempt_updates_and_residual_rows_are_visible(self):
        from experiments.adaptive_placement_v4.planner import CandidateJournal
        budget = dict(iterations=0, residual_evaluations=0)
        history = CandidateJournal(budget, Limits())
        # 100 attempted updates, then the terminal nonconverged residual check.
        for i in range(101):
            history.append(dict(iteration=i, q=np.zeros(7), position_error_m=.01,
                                orientation_error_rad=0., objective=1.))
        self.assertEqual(budget, dict(iterations=100, residual_evaluations=101))
        self.assertEqual(sum(r['update_counted'] for r in history), 100)
        self.assertEqual(history.updates, 100)
        with self.assertRaises(PlanningFailure):
            history.append(dict(iteration=101))

    def test_total_budget_includes_updates_from_subdivided_parent(self):
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(ToyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([.004, 0., 0.]))], np.tile([-2., 2.], (7, 1)),
                ClearChecker(), [1], 1000, journal, limits=Limits(iterations=2))
        self.assertEqual(raised.exception.gate, 'total_ik_iteration_limit')
        self.assertEqual(journal['budgets']['iterations'], 2)
        self.assertEqual(journal['budgets']['attempts'], 2)
        self.assertEqual(journal['attempts'][0]['status'], 'subdivided')
        self.assertEqual(journal['attempts'][0]['iterations'], 2)
        self.assertEqual(journal['attempts'][1]['iterations'], 0)

    def test_minimum_motion_stops_without_child_attempt(self):
        class TinyRobot(ToyRobot):
            def forward_kinematics(self, q):
                return pin.SE3(np.eye(3), np.array([1e-4*q[0], 0., 0.]))
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(TinyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([1e-5, 0., 0.]))], np.tile([-2., 2.], (7, 1)),
                ClearChecker(), [1], 1000, journal,
                dependencies=Dependencies(solve=lambda *args: np.array([.1]+[0.]*6)))
        self.assertEqual(raised.exception.gate, 'subdivision_minimum_motion')
        self.assertEqual(journal['budgets']['attempts'], 1)
        self.assertEqual(journal['attempts'][0]['status'], 'failed')

    def test_exact_multistage_endpoints_and_rotation_spacing(self):
        class PoseRobot(ToyRobot):
            def forward_kinematics(self, q):
                return pin.SE3(pin.exp3(np.array([0., 0., .2*q[1]])), np.array([.04*q[0], 0., 0.]))
        def exact(robot, seed, target, bounds, history):
            q = seed.copy()
            q[0] = target.translation[0]/.04
            q[1] = pin.log3(target.rotation)[2]/.2
            history.append(dict(iteration=0, q=q.copy(), position_error_m=0.,
                                orientation_error_rad=0., objective=0.))
            return q
        endpoints = [pin.SE3(pin.exp3(np.array([0., 0., angle])), np.array([x, 0., 0.]))
                     for x, angle in ((.004, .016), (.008, .032), (.006, .008))]
        before = [p.homogeneous.copy() for p in endpoints]
        journal = {}
        schedule = plan_route(PoseRobot(), np.zeros(7), np.full(7, .001), endpoints,
            np.tile([-2., 2.], (7, 1)), ClearChecker(), [3, 3, 3], 1000, journal,
            dependencies=Dependencies(solve=exact))
        accepted = [a for a in journal['attempts'] if a['status'] == 'accepted']
        for a in journal['attempts']:
            self.assertLessEqual(np.linalg.norm(a['pose'][:3, 3]-a['start_pose'][:3, 3]), .005+1e-14)
            self.assertLessEqual(np.linalg.norm(pin.log3(a['start_pose'][:3, :3].T@a['pose'][:3, :3])), .02+1e-14)
        for stage, endpoint in enumerate(before):
            np.testing.assert_array_equal([a for a in accepted if a['stage'] == stage][-1]['pose'], endpoint)
            np.testing.assert_array_equal(endpoints[stage].homogeneous, endpoint)
        self.assertEqual(len(journal['attempts']), journal['budgets']['attempts'])
        self.assertTrue(all(a['joint_step_rad'] <= .05 for a in accepted))
        commands = np.vstack([np.full(7, .001)]+[q for _, q in schedule])
        self.assertLessEqual(np.max(np.abs(np.diff(commands, axis=0))), 2/30+1e-14)

    def test_nonfinite_seed_rejected_before_native_fk(self):
        from experiments.adaptive_placement_v4.planner import FiniteRobot
        class NoCall(ToyRobot):
            def forward_kinematics(self, q):
                raise AssertionError('native FK must not see nonfinite q')
        with self.assertRaises(PlanningFailure) as raised:
            FiniteRobot(NoCall()).forward_kinematics(np.full(7, np.nan))
        self.assertEqual(raised.exception.gate, 'nonfinite')

    def test_limits_cannot_be_loosened(self):
        for args in (dict(attempts=401), dict(iterations=40001), dict(depth=9), dict(depth=-1)):
            with self.assertRaises(PlanningFailure):
                Limits(**args)

    def test_depth_eight_is_finite_with_no_hidden_parent_attempts(self):
        class TinyRobot(ToyRobot):
            def forward_kinematics(self, q):
                return pin.SE3(np.eye(3), np.array([1e-4*q[0], 0., 0.]))
        def exact(robot, seed, target, bounds, history):
            q = seed.copy(); q[0] = target.translation[0]/1e-4
            history.append(dict(iteration=0, q=q.copy(), position_error_m=0.,
                                orientation_error_rad=0., objective=0.))
            return q
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(TinyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([.004, 0., 0.]))], np.tile([-100., 100.], (7, 1)),
                ClearChecker(), [1], 1000, journal, dependencies=Dependencies(solve=exact))
        self.assertEqual(raised.exception.gate, 'subdivision_depth_limit')
        self.assertEqual(journal['budgets']['attempts'], 9)
        self.assertEqual([a['depth'] for a in journal['attempts']], list(range(9)))
        self.assertEqual(journal['attempts'][-1]['status'], 'failed')

    def real_checker(self):
        import mujoco
        from experiments.bounded_placement_v3.collision import CollisionChecker
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="hand"><joint type="slide" axis="1 0 0"/><geom type="sphere" size=".001"/></body>
          <body name="payload" pos="0 .1 0"><freejoint/><geom type="sphere" size=".002"/></body>
          <body name="obstacle" pos=".02 .1 0"><geom type="sphere" size=".002"/></body>
          <body name="base" pos=".5 .1 0"><geom type="sphere" size=".002"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        class OneJointChecker(CollisionChecker):
            def segment(self, start, end, final_lowering, carrying=True):
                return super().segment(start[:1], end[:1], final_lowering, carrying)
        return OneJointChecker(model, data, 'hand', 'payload', 'base', ['hand'], [], [0]), data

    def test_shared_collision_budget_includes_bridge_and_stops_before_solver(self):
        checker, data = self.real_checker()
        before = data.qpos.copy()
        checker.checked = 9999
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(ToyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([.004, 0., 0.]))], np.tile([-2., 2.], (7, 1)),
                checker, [1], 1000, journal)
        self.assertEqual(raised.exception.gate, 'collision_sample_limit')
        self.assertEqual(journal['budgets']['collision_samples'], 10000)
        self.assertEqual(journal['budgets']['attempts'], 0)
        np.testing.assert_array_equal(before, data.qpos)

    def test_payload_collision_on_subdivided_segment_stops_without_fallback(self):
        checker, data = self.real_checker()
        before = data.qpos.copy()
        journal = {}
        with self.assertRaises(PlanningFailure) as raised:
            plan_route(ToyRobot(), np.zeros(7), np.zeros(7),
                [pin.SE3(np.eye(3), np.array([.004, 0., 0.]))], np.tile([-2., 2.], (7, 1)),
                checker, [1], 1000, journal)
        self.assertEqual(raised.exception.gate, 'path_collision')
        self.assertEqual(len(journal['attempts']), 2)
        self.assertEqual(journal['attempts'][1]['status'], 'failed')
        self.assertTrue(any('payload' in c['bodies'] for c in checker.journal[-1]['unexpected']))
        self.assertEqual(journal['budgets']['collision_samples'], checker.checked)
        np.testing.assert_array_equal(before, data.qpos)


if __name__ == '__main__':
    unittest.main()