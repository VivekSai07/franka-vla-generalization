"""Real FK regressions; no actual expert invocation in this test module.

POSE_SNAPSHOT_TEST_LEGACY=1 runs the stability contract against the old planner
to demonstrate RED, without changing or mocking either implementation.
"""
import ast
import inspect
import os
import textwrap
import unittest

import numpy as np
import pinocchio as pin

from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import Q_HOME


def candidate():
    if os.environ.get("POSE_SNAPSHOT_TEST_LEGACY") == "1":
        return TaskSpaceTrajectory
    from experiments.pose_snapshot_v2.planner import SnapshotTaskSpaceTrajectory
    return SnapshotTaskSpaceTrajectory


class PoseSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.robot = FrankaPanda()
        self.q = np.asarray(Q_HOME).copy()

    def test_legacy_live_fk_mutates_retained_start_and_end(self):
        for endpoint in ("start", "end"):
            with self.subTest(endpoint=endpoint):
                live = self.robot.forward_kinematics(self.q)
                frozen = live.copy()
                other = pin.SE3.Identity()
                trajectory = (TaskSpaceTrajectory(live, other, 2.) if endpoint == "start"
                              else TaskSpaceTrajectory(other, live, 2.))
                changed = self.q.copy()
                changed[0] += .2
                self.robot.forward_kinematics(changed)
                time = 0. if endpoint == "start" else 2.
                self.assertGreater(np.linalg.norm(
                    trajectory.get_pose(time).translation - frozen.translation), .06)

    def test_start_snapshot_survives_real_fk_update(self):
        live = self.robot.forward_kinematics(self.q)
        frozen = live.copy()
        trajectory = candidate()(live, pin.SE3.Identity(), 2.)
        changed = self.q.copy()
        changed[0] += .2
        self.robot.forward_kinematics(changed)
        np.testing.assert_array_equal(trajectory.start_pose.homogeneous, frozen.homogeneous)
        np.testing.assert_allclose(trajectory.get_pose(0).homogeneous, frozen.homogeneous, atol=1e-14)

    def test_end_snapshot_survives_real_fk_update(self):
        live = self.robot.forward_kinematics(self.q)
        frozen = live.copy()
        trajectory = candidate()(pin.SE3.Identity(), live, 2.)
        changed = self.q.copy()
        changed[0] += .2
        self.robot.forward_kinematics(changed)
        np.testing.assert_array_equal(trajectory.end_pose.homogeneous, frozen.homogeneous)
        np.testing.assert_allclose(trajectory.get_pose(2).homogeneous, frozen.homogeneous, atol=1e-14)

    def test_both_inputs_are_owned_copies(self):
        start = self.robot.forward_kinematics(self.q).copy()
        end = pin.SE3(pin.exp3(np.array([.2, -.3, .4])), np.array([.2, .3, .5]))
        expected_start, expected_end = start.copy(), end.copy()
        trajectory = candidate()(start, end, 2.)
        start.translation[:] += 1.
        end.translation[:] -= 1.
        start.rotation = np.eye(3)
        end.rotation = np.eye(3)
        np.testing.assert_array_equal(trajectory.start_pose.homogeneous, expected_start.homogeneous)
        np.testing.assert_array_equal(trajectory.end_pose.homogeneous, expected_end.homogeneous)

    def test_linear_and_cubic_match_frozen_expected_after_fk(self):
        # Removing either endpoint copy makes at least one independent FK case fail.
        for method in ("linear", "cubic"):
            for endpoint in ("start", "end"):
                with self.subTest(method=method, endpoint=endpoint):
                    live = self.robot.forward_kinematics(self.q)
                    other = pin.SE3(pin.exp3(np.array([.2, -.3, .4])), np.array([.2, .3, .5]))
                    start, end = (live, other) if endpoint == "start" else (other, live)
                    frozen_start, frozen_end = start.copy(), end.copy()
                    trajectory = candidate()(start, end, 2., method)
                    changed = self.q.copy()
                    changed[0] += .2
                    self.robot.forward_kinematics(changed)
                    for t in (-1., 0., .25, .7, 1., 1.75, 2., 3.):
                        alpha = np.clip(t / 2., 0., 1.)
                        s = alpha if method == "linear" else 3 * alpha**2 - 2 * alpha**3
                        translation = (1 - s) * frozen_start.translation + s * frozen_end.translation
                        rotation = pin.SE3.Interpolate(frozen_start, frozen_end, s).rotation
                        expected = pin.SE3(rotation, translation)
                        np.testing.assert_allclose(trajectory.get_pose(t).homogeneous,
                                                   expected.homogeneous, rtol=0, atol=1e-14)

    def test_duration_floor_and_inherited_interpolation_are_unchanged(self):
        cls = candidate()
        self.assertIs(cls.get_pose, TaskSpaceTrajectory.get_pose)
        start, end = pin.SE3.Identity(), pin.SE3.Identity()
        end.translation = np.ones(3)
        for duration in (0., -1., 1e-8, 2.):
            for method in ("linear", "cubic", "fallback"):
                actual, old = cls(start, end, duration, method), TaskSpaceTrajectory(start, end, duration, method)
                self.assertEqual(actual.duration, old.duration)
                for t in (-1., 0., 1e-6, 1., 3.):
                    np.testing.assert_array_equal(actual.get_pose(t).homogeneous, old.get_pose(t).homogeneous)


class VersionedAdapterTests(unittest.TestCase):
    def test_copied_run_ast_diff_is_exactly_one_constructor_name(self):
        from experiments.paired_demonstrations.expert_v1 import Expert as OldExpert
        from experiments.pose_snapshot_v2.expert import Expert
        old = ast.parse(textwrap.dedent(inspect.getsource(OldExpert.run)))
        new = ast.parse(textwrap.dedent(inspect.getsource(Expert.run)))
        replacements = 0
        for node in ast.walk(new):
            if isinstance(node, ast.Name) and node.id == "SnapshotTaskSpaceTrajectory":
                node.id = "TaskSpaceTrajectory"
                replacements += 1
        self.assertEqual(replacements, 1)
        self.assertEqual(ast.dump(new, include_attributes=False), ast.dump(old, include_attributes=False))

    def test_version_is_only_configuration_difference_and_old_global_unchanged(self):
        from experiments.paired_demonstrations import expert_v1
        old_class = expert_v1.TaskSpaceTrajectory
        from experiments.pose_snapshot_v2.expert import Expert, EXPERT_VERSION
        old, new = expert_v1.Expert(), Expert()
        self.assertIs(expert_v1.TaskSpaceTrajectory, old_class)
        self.assertIs(old_class, TaskSpaceTrajectory)
        self.assertIsInstance(new, expert_v1.Expert)
        expected = dict(old.config, expert_version=EXPERT_VERSION)
        self.assertEqual(new.config, expected)
        np.testing.assert_array_equal(old.bounds, new.bounds)
        self.assertNotEqual(EXPERT_VERSION, expert_v1.EXPERT_VERSION)


if __name__ == "__main__":
    unittest.main()