import unittest

import numpy as np

from experiments.xvla_props_p2_eval.diagnostics import grasp


class ClosedRunTests(unittest.TestCase):
    def test_short_runs_are_chatter(self):
        command = [255]*3 + [0]*4 + [255]*2 + [0]*5 + [255]
        self.assertEqual(grasp.closed_runs(command, 5), [(9, 13)])

    def test_threshold_is_exclusive_at_127_5(self):
        self.assertEqual(grasp.closed_runs([127.4]*5), [(0, 4)])
        self.assertEqual(grasp.closed_runs([127.5]*5), [])

    def test_first_closure_and_none(self):
        self.assertIsNone(grasp.first_closure([255]*10))
        self.assertEqual(grasp.first_closure([255, 0, 0, 0, 0, 0, 255]), 1)

    def test_grasp_closure_picks_the_run_containing_the_lift(self):
        command = [255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0]
        self.assertEqual(grasp.grasp_closure(command, lift_tick=9), 7)
        self.assertEqual(grasp.grasp_closure(command, lift_tick=None), 1)
        self.assertEqual(grasp.grasp_closure(command, lift_tick=6), 1, 'a lift outside every run falls back to the first closure')
        self.assertIsNone(grasp.grasp_closure([255]*5, lift_tick=2))


class FrameTests(unittest.TestCase):
    def test_identity_rotation(self):
        out = grasp.to_hand_frame(np.array([1, 2, 3.]), np.eye(3).ravel(), np.array([1.5, 2, 3.25]))
        np.testing.assert_allclose(out, [0.5, 0, 0.25])

    def test_quarter_turn_about_z(self):
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]).ravel()
        out = grasp.to_hand_frame(np.array([1, 2, 3.]), rot, np.array([1, 3, 3.]))
        np.testing.assert_allclose(out, [1, 0, 0], atol=1e-12)

    def test_yaw_error_is_modulo_ninety(self):
        self.assertAlmostEqual(grasp.yaw_error_deg((1, 0), 0), 0)
        self.assertAlmostEqual(grasp.yaw_error_deg((0, 1), 0), 0)
        self.assertAlmostEqual(grasp.yaw_error_deg((1, 0), 30), -30)
        self.assertAlmostEqual(grasp.yaw_error_deg((1, 0), 50), 40)
        self.assertGreaterEqual(grasp.yaw_error_deg((1, 0), 45), -45)
        self.assertLess(grasp.yaw_error_deg((1, 0), 45), 45)


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.env = grasp.envelope(np.tile(np.arange(101.0)[:, None], (1, 4)))

    def test_percentiles(self):
        np.testing.assert_allclose(self.env['lo'], [1]*4)
        np.testing.assert_allclose(self.env['hi'], [99]*4)

    def test_inside_is_inclusive_and_per_component(self):
        self.assertTrue(grasp.inside(self.env, np.array([1, 99, 50, 50.])).all())
        out = grasp.inside(self.env, np.array([np.nextafter(1, 0), 99, 50, np.nextafter(99, 100)]))
        self.assertEqual(out.tolist(), [False, True, True, False])


class HeldTests(unittest.TestCase):
    def test_both_fingers_for_ten_ticks_after_start(self):
        left = np.zeros(40, bool); right = np.zeros(40, bool)
        left[20:30] = right[20:30] = True
        self.assertTrue(grasp.held_ticks(left, right, 15))
        self.assertFalse(grasp.held_ticks(left, right, 21))
        self.assertFalse(grasp.held_ticks(left, np.zeros(40, bool), 0))


if __name__ == '__main__':
    unittest.main()
