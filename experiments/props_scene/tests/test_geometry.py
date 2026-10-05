import unittest

import mujoco
import numpy as np

from experiments.props_scene import constants as c
from experiments.props_scene import geometry as g

# Bounding boxes measured in the robosuite image (TODO.md, 2026-10-05), full sizes in cm.
SPEC_CM = {"milk": (4.0, 4.0, 14.4), "bread": (4.8, 4.0, 4.8), "cereal": (10, 3, 15),
           "can": (5.2, 5.2, 8), "lemon": (7, 4, 4.3), "bottle": (5.5, 5.9, 16)}


def _settled(model, name, yaw_deg):
    from experiments.props_scene.scene import reset
    scene = reset(model, {name: (0.30, 0.0, yaw_deg)}, [0.0] * 7)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, scene["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    return data


class ExtentsTests(unittest.TestCase):
    def test_extents_match_the_measured_bounding_boxes(self):
        from experiments.props_scene.scene import build_model
        for name, spec in SPEC_CM.items():
            _, model = build_model((name,), "left")
            full = np.sort(2 * g.extents(model, name)) * 100
            np.testing.assert_allclose(full, sorted(spec), atol=0.3, err_msg=name)   # 3 mm

    def test_every_narrow_extent_fits_the_gripper_and_the_wall(self):
        from experiments.props_scene.scene import build_model
        for name in c.OBJECTS:
            _, model = build_model((name,), "left")
            narrow = 2 * float(g.extents(model, name)[:2].min())
            self.assertLess(narrow, 0.07, name)
            self.assertLess(narrow, 0.08, name)                                       # gripper opening


class NarrowAxisTests(unittest.TestCase):
    def test_cereal_and_lemon_follow_their_yaw(self):
        from experiments.props_scene.scene import build_model
        for name in ("cereal", "lemon"):
            _, model = build_model((name,), "left")
            half = g.extents(model, name)[:2]
            narrow_axis = int(np.argmin(half))
            for yaw_deg in (-89.0, 0.0, 45.0, 89.0):
                data = _settled(model, name, yaw_deg)
                got = g.narrow_axis_yaw(model, data, name)
                expected = np.deg2rad(yaw_deg) + (np.pi / 2 if narrow_axis == 1 else 0.0)
                difference = (got - expected + np.pi / 2) % np.pi - np.pi / 2           # period 180 degrees
                self.assertLess(abs(np.degrees(difference)), 2.0, (name, yaw_deg))
                self.assertTrue(-np.pi / 2 <= got < np.pi / 2)
                # independent of the extents: the hull really is thinner along that direction
                perpendicular = got + np.pi / 2
                self.assertLess(g.horizontal_width(model, data, name, got) + 0.01,
                                g.horizontal_width(model, data, name, perpendicular), (name, yaw_deg))

    def test_top_and_bottom(self):
        from experiments.props_scene.scene import build_model
        _, model = build_model(("milk",), "left")
        data = _settled(model, "milk", 10.0)
        self.assertLess(abs(g.bottom_z(model, data, "milk")), 0.002)
        self.assertAlmostEqual(g.top_z(model, data, "milk") - g.bottom_z(model, data, "milk"), 0.144, delta=0.003)


if __name__ == "__main__":
    unittest.main()
