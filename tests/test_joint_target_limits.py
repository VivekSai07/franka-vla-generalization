import unittest

import numpy as np

from src.controllers.joint_target_limits import bounded_joint_target


class JointTargetLimitTests(unittest.TestCase):
    def test_unconstrained_command_is_unchanged(self):
        target, velocity, saturated = bounded_joint_target(np.zeros(7), np.ones(7), 1/30,
                                                          np.array([[-2., 2.]] * 7))
        np.testing.assert_allclose(target, np.ones(7)/30)
        np.testing.assert_array_equal(velocity, np.ones(7))
        self.assertFalse(saturated.any())

    def test_joint_four_regression_does_not_cross_lower_bound(self):
        bounds = np.array([[-4., 4.]] * 7)
        bounds[3] = [-3.0718, -.0698]
        previous = np.zeros(7); previous[3] = -3.0619371
        requested = np.zeros(7); requested[3] = -1.0079888669381085
        target, _, saturated = bounded_joint_target(previous, requested, 1/30, bounds)
        self.assertGreaterEqual(target[3], bounds[3, 0])
        self.assertTrue(saturated[3])
        for _ in range(100):
            target, _, _ = bounded_joint_target(target, requested, 1/30, bounds)
        self.assertGreaterEqual(target[3], bounds[3, 0])
        requested[3] = 1.
        recovered, _, _ = bounded_joint_target(target, requested, 1/30, bounds)
        self.assertGreater(recovered[3], target[3])

    def test_velocity_cap_and_both_position_bounds(self):
        target, velocity, saturated = bounded_joint_target(np.array([.999, -.999]),
                                                          np.array([100., -100.]), .1,
                                                          np.array([[-1., 1.], [-1., 1.]]))
        self.assertTrue(np.all(np.abs(target) <= 1))
        self.assertTrue(np.all(np.abs(velocity) <= 2))
        self.assertTrue(saturated.all())

    def test_invalid_inputs_and_outside_prior_target_fail(self):
        for previous, velocity, dt, bounds in (
            ([2.], [0.], .1, [[-1., 1.]]), ([0.], [np.nan], .1, [[-1., 1.]]),
            ([0.], [1.], 0., [[-1., 1.]]), ([0.], [1., 2.], .1, [[-1., 1.]]),
            ([0.], [1.], .1, [[1., -1.]]),
        ):
            with self.assertRaises(ValueError):
                bounded_joint_target(previous, velocity, dt, bounds)


if __name__ == "__main__": unittest.main()