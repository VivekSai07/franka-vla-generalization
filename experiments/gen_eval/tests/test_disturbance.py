import unittest

from experiments.gen_eval.disturbance import FORCED_OPEN_COMMAND, OPEN_THRESHOLD, Disturbance
from experiments.gen_scene.conditions import NOMINAL, Condition

MOVE = Condition(axis="disturbance", level="move", disturbance="move", move_to=(0.35, 0.05))
DROP = Condition(axis="disturbance", level="drop", disturbance="drop")


class Move(unittest.TestCase):
    def test_fires_at_six_cm_horizontally_before_any_close(self):
        d = Disturbance(MOVE)
        self.assertFalse(d.move_due(0, (0.30, 0.0), (0.30 + 0.0601, 0.0)))
        self.assertTrue(d.move_due(1, (0.30, 0.0), (0.30 + 0.06, 0.0)))
        self.assertTrue(d.move_due(2, (0.30, 0.0), (0.30, 0.03)))

    def test_distance_is_horizontal_height_does_not_count(self):
        d = Disturbance(MOVE)
        self.assertTrue(d.move_due(0, (0.3, 0.0), (0.3, 0.0)))   # the method takes xy only: a hand high above still counts

    def test_a_close_command_before_the_hand_arrives_means_it_never_fires(self):
        d = Disturbance(MOVE)
        d.note_applied_gripper(OPEN_THRESHOLD)       # at or below the open threshold is a close command
        self.assertFalse(d.move_due(3, (0.3, 0.0), (0.3, 0.0)))

    def test_an_open_command_does_not_count_as_closing(self):
        d = Disturbance(MOVE)
        d.note_applied_gripper(OPEN_THRESHOLD + 1)
        d.note_applied_gripper(255.0)
        self.assertTrue(d.move_due(3, (0.3, 0.0), (0.3, 0.0)))

    def test_fires_once(self):
        d = Disturbance(MOVE)
        d.fire(5)
        self.assertFalse(d.move_due(6, (0.3, 0.0), (0.3, 0.0)))
        with self.assertRaises(RuntimeError):
            d.fire(6)
        self.assertEqual(d.summary(), dict(disturbance_fired=True, disturbance_tick=5))

    def test_move_never_fires_a_drop_and_the_reverse(self):
        self.assertFalse(Disturbance(DROP).move_due(0, (0.3, 0.0), (0.3, 0.0)))
        self.assertFalse(Disturbance(MOVE).drop_due(0, 0.2, 0.0))
        self.assertFalse(Disturbance(NOMINAL).move_due(0, (0.3, 0.0), (0.3, 0.0)))
        self.assertFalse(Disturbance(NOMINAL).drop_due(0, 0.2, 0.0))


class Drop(unittest.TestCase):
    def test_fires_when_the_target_is_five_cm_above_its_start(self):
        d = Disturbance(DROP)
        self.assertFalse(d.drop_due(10, 1.0499, 1.0))
        self.assertTrue(d.drop_due(11, 1.05, 1.0))     # 1.05 - 1.0 is 0.05 plus float noise, at or above the threshold
        self.assertTrue(d.drop_due(12, 0.30, 0.04))

    def test_only_after_a_lift_never_for_a_target_on_the_table(self):
        d = Disturbance(DROP)
        for tick in range(200):
            self.assertFalse(d.drop_due(tick, 0.04, 0.04))
        self.assertFalse(d.drop_due(200, 0.04 - 0.01, 0.04))   # pushed down is not a lift

    def test_forced_open_for_exactly_ten_ticks_from_the_firing_tick(self):
        d = Disturbance(DROP)
        self.assertFalse(d.forced_open(30))
        d.fire(40)
        self.assertEqual([t for t in range(30, 60) if d.forced_open(t)], list(range(40, 50)))

    def test_fires_once_and_forced_open_needs_a_fired_drop(self):
        d = Disturbance(DROP)
        d.fire(40)
        self.assertFalse(d.drop_due(41, 0.3, 0.04))
        m = Disturbance(MOVE)
        m.fire(5)
        self.assertFalse(m.forced_open(5))

    def test_the_forced_command_is_the_experts_open_value(self):
        self.assertEqual(FORCED_OPEN_COMMAND, 255.0)
        self.assertGreater(FORCED_OPEN_COMMAND, OPEN_THRESHOLD)
        self.assertEqual(OPEN_THRESHOLD, 127)


class Unfired(unittest.TestCase):
    def test_summary_of_a_disturbance_that_never_fired(self):
        self.assertEqual(Disturbance(DROP).summary(), dict(disturbance_fired=False, disturbance_tick=None))


if __name__ == "__main__":
    unittest.main()
