"""Real numerical/geometry tests. No expert invocation or policy loading."""
import importlib
from pathlib import Path
import unittest

import mujoco
import numpy as np
import pinocchio as pin


def candidate(name):
    path = Path(__file__).resolve().parents[1] / 'experiments/bounded_placement_v3' / (name + '.py')
    assert path.is_file(), f'Missing bounded-placement implementation: {name}'
    return importlib.import_module('experiments.bounded_placement_v3.' + name)


class SolverTests(unittest.TestCase):
    def test_box_solver_reoptimizes_free_joint_not_solve_then_clip(self):
        # Objective .5*((x+y-1)^2+y^2); x<=.2 => y=.4.
        s = candidate('solver')
        A = np.array([[1., 1.], [0., 1.]])
        H, g = A.T @ A, -A.T @ np.array([1., 0.])
        x = s.box_quadratic(H, g, np.array([-1., -1.]), np.array([.2, 1.]))
        np.testing.assert_allclose(x, [.2, .4], atol=1e-9, rtol=0)
        self.assertLess(float(.5*x@H@x+g@x), -.3)

    def test_box_solver_rejects_nonfinite_and_inverted_bounds(self):
        s = candidate('solver')
        for lo, hi in (([1.], [0.]), ([np.nan], [1.])):
            with self.assertRaises(s.PlanningFailure):
                s.box_quadratic(np.eye(1), np.ones(1), np.array(lo), np.array(hi))

    def test_real_franka_waypoint_solver_and_independent_fk(self):
        from experiments.paired_demonstrations.expert_v1 import Expert
        from src.robots.franka_panda.config import Q_HOME
        s = candidate('solver')
        expert = Expert()
        q = np.asarray(Q_HOME).copy()
        goal = q.copy(); goal[0] += .012; goal[2] += .006
        target = expert.robot.forward_kinematics(goal).copy()
        history = []
        solved = s.solve_waypoint(expert.robot, q, target, expert.bounds, history)
        fk = expert.robot.forward_kinematics(solved).copy()
        self.assertLessEqual(np.linalg.norm(fk.translation-target.translation), .001)
        self.assertLessEqual(np.linalg.norm(pin.log3(fk.rotation.T@target.rotation)), .01)
        self.assertTrue(np.all(solved[:7] >= expert.bounds[:, 0]+1e-6))
        self.assertTrue(np.all(solved[:7] <= expert.bounds[:, 1]-1e-6))
        self.assertGreater(len(history), 1)
        self.assertLessEqual(len(history), 101)
        for a, b in zip(history, history[1:]):
            self.assertLessEqual(np.max(np.abs(np.array(a['q'])-b['q'])), .05+1e-14)

    def test_impossible_real_pose_fails_with_finite_candidates(self):
        from experiments.paired_demonstrations.expert_v1 import Expert
        from src.robots.franka_panda.config import Q_HOME
        s = candidate('solver'); expert = Expert(); history = []
        with self.assertRaises(s.PlanningFailure):
            s.solve_waypoint(expert.robot, np.asarray(Q_HOME), pin.SE3(np.eye(3), np.ones(3)*10),
                             expert.bounds, history)
        self.assertTrue(history)
        self.assertLessEqual(len(history), 101)
        self.assertTrue(all(np.isfinite(r['q']).all() for r in history))


class PathTests(unittest.TestCase):
    def test_pose_spacing_snapshots_and_exact_endpoints(self):
        p = candidate('path')
        a = pin.SE3.Identity(); b = pin.SE3(pin.exp3(np.array([0., 0., .051])), np.array([.012, 0., 0.]))
        expected = b.copy(); route = p.pose_waypoints(a, [b])
        b.translation[:] += 2
        np.testing.assert_array_equal(route[-1][1].homogeneous, expected.homogeneous)
        previous = a
        for _, pose in route:
            self.assertLessEqual(np.linalg.norm(pose.translation-previous.translation), .005+1e-14)
            self.assertLessEqual(np.linalg.norm(pin.log3(previous.rotation.T@pose.rotation)), .02+1e-14)
            previous = pose

    def test_waypoint_cap_and_branch_jump_reject(self):
        p = candidate('path'); s = candidate('solver')
        with self.assertRaises(s.PlanningFailure):
            p.pose_waypoints(pin.SE3.Identity(), [pin.SE3(np.eye(3), np.array([2.01, 0., 0.]))])
        with self.assertRaises(s.PlanningFailure):
            p.check_branch(np.zeros(7), np.array([.050001]+[0.]*6))
        p.check_branch(np.zeros(7), np.full(7, .05))

    def test_command_bridge_time_scaling_preserves_vertices_and_minima(self):
        p = candidate('path')
        bounds = np.tile([-2., 2.], (7, 1)); previous = np.zeros(7)
        start = np.full(7, .1); end = np.full(7, .15)
        schedule = p.time_parameterize(previous, [(0, start), (1, end)], [3, 4], bounds, 100)
        self.assertEqual(len(schedule), 7)
        commands = np.array([q for _, q in schedule])
        self.assertLessEqual(np.max(np.abs(np.diff(np.vstack([previous, commands]), axis=0))), 2/30)
        np.testing.assert_array_equal(commands[2], start)
        np.testing.assert_array_equal(commands[-1], end)
        with self.assertRaises(candidate('solver').PlanningFailure):
            p.time_parameterize(previous, [(0, start), (1, end)], [3, 4], bounds, 6)

    def test_invalid_bridge_bounds_do_not_clip(self):
        p = candidate('path'); s = candidate('solver'); bounds = np.tile([-1., 1.], (7, 1))
        for previous, goal in ((np.full(7, 1.1), np.zeros(7)), (np.zeros(7), np.ones(7))):
            with self.assertRaises(s.PlanningFailure):
                p.time_parameterize(previous, [(0, goal)], [1], bounds, 10)


class CollisionTests(unittest.TestCase):
    def scene(self):
        xml = '''<mujoco><worldbody>
          <body name="hand"><joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01"/></body>
          <body name="payload" pos="0 .1 0"><freejoint/><geom type="sphere" size=".02"/></body>
          <body name="obstacle" pos=".1 .1 0"><geom type="sphere" size=".02"/></body>
          <body name="base" pos=".5 .1 0"><geom type="sphere" size=".02"/></body>
        </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_correct_hand_relative_payload_and_midpath_collision(self):
        c = candidate('collision'); s = candidate('solver'); m, d = self.scene()
        before = d.qpos.copy()
        checker = c.CollisionChecker(m, d, 'hand', 'payload', 'base', ['hand'], [], [0])
        np.testing.assert_array_equal(checker.relative.translation, [0., .1, 0.])
        self.assertFalse(checker.sample(np.array([0.]), False))
        self.assertFalse(checker.sample(np.array([.2]), False))
        with self.assertRaises(s.PlanningFailure):
            checker.segment(np.array([0.]), np.array([.2]), False)
        np.testing.assert_array_equal(d.qpos, before)
        self.assertTrue(checker.journal[-1]['unexpected'])
        self.assertTrue(any('payload' in row['bodies'] for row in checker.journal[-1]['contacts']))

    def test_contact_exceptions_and_penetration_tolerance(self):
        c = candidate('collision')
        self.assertFalse(c.unexpected_contact({'hand','payload'}, -.1, {'hand'}, {'hand'}, 'payload', 'base', False))
        self.assertTrue(c.unexpected_contact({'payload','base'}, 0., {'hand'}, set(), 'payload', 'base', False))
        self.assertFalse(c.unexpected_contact({'payload','base'}, -1e-5, {'hand'}, set(), 'payload', 'base', True))
        self.assertTrue(c.unexpected_contact({'payload','base'}, -1.01e-5, {'hand'}, set(), 'payload', 'base', True))
        self.assertTrue(c.unexpected_contact({'payload','obstacle'}, -.001, {'hand'}, set(), 'payload', 'base', True))

    def test_sample_budget_fails_before_unchecked_sample(self):
        c = candidate('collision'); m, d = self.scene()
        checker = c.CollisionChecker(m, d, 'hand', 'payload', 'base', ['hand'], [], [0])
        checker.checked = 10000
        with self.assertRaises(candidate('solver').PlanningFailure):
            checker.sample(np.array([0.]), False)
        self.assertEqual(checker.checked, 10000)


class GateTests(unittest.TestCase):
    def ready(self):
        return dict(grasped=True, xy=.01, z=.04, hand_error=.005, rotation_error=.03,
                    max_speed=.10, unexpected=False)

    def test_inclusive_readiness_boundaries_and_streak_reset(self):
        g = candidate('gates'); gate = g.ReleaseGate()
        for _ in range(9): self.assertFalse(gate.update(**self.ready()))
        self.assertTrue(gate.update(**self.ready()))
        for field, value in [('grasped', False), ('xy', .010001), ('z', .039999),
                             ('z', .090001), ('hand_error', .005001), ('rotation_error', .030001),
                             ('max_speed', .100001), ('unexpected', True), ('xy', np.nan)]:
            row = self.ready(); row[field] = value
            self.assertFalse(gate.update(**row)); self.assertEqual(gate.streak, 0)
        row = self.ready(); row['z'] = .09
        self.assertTrue(g.ready(**row))

    def test_deadline_tracking_and_grasp_loss_hold_closed(self):
        g = candidate('gates'); s = candidate('solver')
        guard = g.ExecutionGuard(1199)
        guard.check(np.zeros(7), np.full(7, .06), True, False)
        self.assertFalse(guard.can_advance)
        guard.tick()
        with self.assertRaises(s.PlanningFailure):
            guard.check(np.zeros(7), np.zeros(7), True, False)
        guard = g.ExecutionGuard(197)
        for _ in range(9): guard.check(np.zeros(7), np.zeros(7), False, False)
        with self.assertRaises(s.PlanningFailure):
            guard.check(np.zeros(7), np.zeros(7), False, False)
        self.assertFalse(guard.opened)

    def test_release_height_contradiction_is_not_waived(self):
        g = candidate('gates'); s = candidate('solver')
        hand = pin.SE3(np.eye(3), np.array([0., 0., .18]))
        relative = pin.SE3(np.eye(3), np.array([0., 0., -.10]))
        self.assertAlmostEqual(g.validate_release_geometry(hand, relative, np.array([0., 0., .02])), .06)
        relative.translation[2] = -.04
        with self.assertRaises(s.PlanningFailure):
            g.validate_release_geometry(hand, relative, np.array([0., 0., .02]))


if __name__ == '__main__':
    unittest.main()