import unittest

import mujoco
import numpy as np

from experiments.props_scene import constants as c
from experiments.props_scene import metrics as m

XML = """
<mujoco>
  <worldbody>
    <body name="bin_blue" pos="0.32 0.33 0"/>
    <body name="bin_orange" pos="0.32 -0.33 0"/>
    <body name="prop_milk" pos="0 0 1"><freejoint/><geom type="box" size="0.02 0.02 0.07" mass="0.2"/></body>
    <body name="left_finger" pos="2 0 1"><freejoint/><geom type="box" size="0.01 0.01 0.03" mass="0.01"/></body>
    <body name="right_finger" pos="2 1 1"><freejoint/><geom type="box" size="0.01 0.01 0.03" mass="0.01"/></body>
    <body name="gripper_dummy"><joint name="g" type="slide" axis="1 0 0"/><geom type="sphere" size="0.01" mass="0.01" contype="0" conaffinity="0"/></body>
  </worldbody>
  <actuator><motor joint="g" ctrlrange="0 255"/></actuator>
</mujoco>
"""


class Scene:
    """A hand-built model with the names the metric looks up; the prop is placed by hand."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def place(self, xyz, ctrl=255, left_finger=None):
        prop = self.model.joint(self.model.body("prop_milk").jntadr[0]).qposadr[0]
        self.data.qpos[prop:prop + 3] = xyz
        if left_finger is not None:
            adr = self.model.joint(self.model.body("left_finger").jntadr[0]).qposadr[0]
            self.data.qpos[adr:adr + 3] = left_finger
        self.data.ctrl[-1] = ctrl
        mujoco.mj_forward(self.model, self.data)
        return m.in_bin(self.model, self.data, "milk", "blue")


class InBinTests(unittest.TestCase):
    def test_inside_and_released_is_a_success_tick(self):
        result = Scene().place((0.32, 0.33, 0.05))
        self.assertTrue(result["inside"] and result["gripper_open"])
        self.assertFalse(result["finger_contact"])
        self.assertTrue(result["success_tick"])
        self.assertAlmostEqual(result["xy_margin_m"], 0.11, places=6)

    def test_on_the_wall_top_is_not_inside(self):
        # centre over the wall footprint, resting on its top (half height 0.07)
        result = Scene().place((0.32 + 0.115, 0.33, 0.07 + 0.07))
        self.assertFalse(result["inside"])
        self.assertFalse(result["success_tick"])
        self.assertLess(result["xy_margin_m"], 0.0)

    def test_balanced_on_the_rim_with_centre_over_the_inner_box_is_not_inside(self):
        result = Scene().place((0.32 + 0.10, 0.33, 0.07 + 0.07))
        self.assertGreater(result["xy_margin_m"], 0.0)
        self.assertFalse(result["inside"])
        self.assertFalse(result["success_tick"])

    def test_the_other_bin_does_not_count(self):
        scene = Scene()
        scene.place((0.32, -0.33, 0.05))
        self.assertFalse(m.in_bin(scene.model, scene.data, "milk", "blue")["success_tick"])
        self.assertTrue(m.in_bin(scene.model, scene.data, "milk", "orange")["success_tick"])

    def test_closed_gripper_is_not_a_success_tick(self):
        result = Scene().place((0.32, 0.33, 0.05), ctrl=0)
        self.assertTrue(result["inside"])
        self.assertFalse(result["gripper_open"])
        self.assertFalse(result["success_tick"])

    def test_a_finger_touching_is_not_a_success_tick(self):
        result = Scene().place((0.32, 0.33, 0.05), left_finger=(0.32 + 0.025, 0.33, 0.05))
        self.assertTrue(result["finger_contact"])
        self.assertFalse(result["success_tick"])

    def test_open_threshold_is_strictly_above_127(self):
        self.assertFalse(Scene().place((0.32, 0.33, 0.05), ctrl=127)["gripper_open"])
        self.assertTrue(Scene().place((0.32, 0.33, 0.05), ctrl=128)["gripper_open"])

    def test_unknown_bin_is_rejected(self):
        scene = Scene()
        with self.assertRaises(ValueError):
            m.in_bin(scene.model, scene.data, "milk", "green")


class StreakTests(unittest.TestCase):
    def test_thirty_consecutive_ticks(self):
        good, stable, reached = {"success_tick": True}, 0, False
        for tick in range(c.SUCCESS_TICKS - 1):
            stable, reached = m.success_streak(stable, good)
        self.assertEqual((stable, reached), (29, False))
        stable, reached = m.success_streak(stable, good)
        self.assertEqual((stable, reached), (30, True))

    def test_a_break_resets(self):
        stable = 29
        stable, reached = m.success_streak(stable, {"success_tick": False})
        self.assertEqual((stable, reached), (0, False))


if __name__ == "__main__":
    unittest.main()
