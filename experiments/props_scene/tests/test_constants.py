import unittest

from experiments.props_scene import constants as c
from experiments.robosuite_scene import contract as scene_contract


class SpecValuesTests(unittest.TestCase):
    def test_bins(self):
        self.assertEqual(c.BIN_INNER_M, 0.22)
        self.assertEqual(c.BIN_WALL_HEIGHT_M, 0.07)
        self.assertEqual(c.BIN_WALL_THICKNESS_M, 0.01)
        self.assertEqual(c.SLOT_XY, {"left": (0.32, 0.33), "right": (0.32, -0.33)})
        self.assertEqual(c.BIN_RGBA["blue"], (0.15, 0.35, 0.85, 1.0))
        self.assertEqual(c.BIN_RGBA["orange"], (0.95, 0.55, 0.10, 1.0))

    def test_objects_and_masses(self):
        self.assertEqual(c.OBJECTS, ("can", "bread", "lemon", "milk", "bottle", "cereal"))
        self.assertEqual(c.MASS_KG, {"can": 0.15, "bread": 0.05, "lemon": 0.08,
                                     "milk": 0.20, "bottle": 0.20, "cereal": 0.15})

    def test_start_region(self):
        self.assertEqual(c.START_X_M, (0.27, 0.40))   # amended 2026-10-05: clear of the arm's working envelope
        self.assertEqual(c.START_Y_M, (-0.12, 0.12))
        self.assertEqual(c.START_YAW_DEG, (-90.0, 90.0))
        self.assertFalse(hasattr(c, "CORNER_X_M"))    # no near-base stratum any more
        # amended 2026-10-06: a rectangle bound alone still admits a reach-limited band near its near corner
        self.assertEqual(c.START_RADIUS_MIN_M, 0.285)
        self.assertEqual(c.YAW_MAX_DEG_BY_OBJECT["lemon"], 75.0)   # its long side nearly aligns with the grasp axis
        for obj in c.OBJECTS:
            if obj != "lemon":
                self.assertEqual(c.YAW_MAX_DEG_BY_OBJECT[obj], 90.0)

    def test_expert_geometry(self):
        self.assertEqual(c.GRASP_BELOW_TOP_M, 0.03)
        self.assertEqual(c.LIFT_CLEARANCE_M, 0.05)
        self.assertEqual(c.RELEASE_ABOVE_WALL_M, 0.02)
        # the hand offset is the sealed cube expert's; two copies must not drift apart
        self.assertEqual(c.HAND_OFFSET_M, scene_contract.GRASP_Z_OFFSET_M)

    def test_success_and_timing(self):
        self.assertEqual((c.FPS, c.SUBSTEPS), (30, 20))
        self.assertEqual((c.SUCCESS_TICKS, c.MAX_TICKS), (30, 900))
        self.assertEqual(c.GRIPPER_OPEN_CTRL, 127)

    def test_gate_thresholds(self):
        self.assertEqual((c.GATE_OVERALL, c.GATE_PER_OBJECT, c.GATE_PER_SLOT), (0.95, 0.90, 0.93))
        self.assertEqual(c.GATE_ATTEMPTS, 288)

    def test_seed_is_b3_style(self):
        import hashlib
        expected = int.from_bytes(hashlib.sha256(b"props_p0_v1:20261005:x").digest()[:4], "big")
        self.assertEqual(c.seed("x"), expected)

    def test_phase_durations_sum(self):
        self.assertEqual(c.PHASE_NAMES, ("approach", "descend", "close", "lift", "move", "lower",
                                         "open", "retract", "home"))
        self.assertEqual(sum(c.PHASE_DURATION_S.values()), 15.0)


if __name__ == "__main__":
    unittest.main()
