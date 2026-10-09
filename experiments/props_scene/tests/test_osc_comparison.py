import unittest

import numpy as np

from experiments.props_scene import osc_comparison as o


class RotErrorTests(unittest.TestCase):
    def test_a_yaw_difference_is_an_axis_angle_about_z(self):
        c, s = np.cos(0.3), np.sin(0.3)
        desired = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(o._rot_error(np.eye(3), desired), [0.0, 0.0, 0.3], atol=1e-9)
        np.testing.assert_allclose(o._rot_error(desired, np.eye(3)), [0.0, 0.0, -0.3], atol=1e-9)


class OscAttemptTests(unittest.TestCase):
    def test_one_central_bread_attempt_succeeds_under_osc(self):
        attempt = {"id": "t", "object": "bread", "bin": "blue", "blue_slot": "left", "target_slot": "left",
                   "pose": [0.33, 0.04, 20.0]}
        row = o.run_attempt(attempt)
        self.assertTrue(row["success"], row)
        self.assertTrue(row["last_tick"]["gripper_open"])


if __name__ == "__main__":
    unittest.main()
