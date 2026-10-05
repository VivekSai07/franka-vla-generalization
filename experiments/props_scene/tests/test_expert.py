import unittest

import mujoco
import numpy as np

from experiments.props_scene import constants as c
from experiments.props_scene import geometry as geo
from experiments.props_scene.expert import PropsExpert, hand_yaw_for, task_text
from experiments.props_scene.scene import build_model, reset

START = (0.30, 0.0, 30.0)


def _episode(obj, bin_colour, seed=1):
    return {"object": obj, "bin": bin_colour, "seed": seed, "home_pose_jitter_deg": [0.0] * 7}


def _settled(model, obj, pose):
    scene = reset(model, {obj: pose}, [0.0] * 7)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, scene["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    return scene, data


class HandYawTests(unittest.TestCase):
    def test_hand_yaw_is_periodic_and_bounded(self):
        for finger in (0.0, 1.0, 2.5):
            for narrow in np.linspace(-1.5, 1.5, 13):
                theta = hand_yaw_for(finger, narrow)
                self.assertTrue(-np.pi / 2 <= theta < np.pi / 2)
                self.assertAlmostEqual(np.sin(finger + theta - narrow) ** 2, 0.0, places=9)

    def test_cereal_fingers_land_on_the_narrow_axis_at_every_yaw(self):
        expert = PropsExpert(("cereal",), "left")
        for yaw_deg in (-89.0, 0.0, 45.0, 89.0):
            _, data = _settled(expert.model, "cereal", (0.30, 0.0, yaw_deg))
            plan = expert.plan_grasp(data, "cereal")
            self.assertFalse(plan["round"])
            difference = (plan["finger_axis_yaw"] - geo.narrow_axis_yaw(expert.model, data, "cereal") + np.pi / 2) % np.pi - np.pi / 2
            self.assertLess(abs(np.degrees(difference)), 5.0, yaw_deg)
            # independent check: the hull is thin along the finger axis and long across it
            finger = plan["finger_axis_yaw"]
            thin = geo.horizontal_width(expert.model, data, "cereal", finger)
            wide = geo.horizontal_width(expert.model, data, "cereal", finger + np.pi / 2)
            self.assertLess(thin, 0.04, yaw_deg)
            self.assertGreater(wide, 0.09, yaw_deg)

    def test_a_square_footprint_is_gripped_by_a_face_not_a_corner(self):
        # Found by the first gate: milk (4 x 4 cm) was treated as round, so at a diagonal yaw the fingers
        # closed on two corners and the carton squeezed out (4 of its 17 grasp failures, all at 47-65 deg).
        expert = PropsExpert(("milk",), "left")
        for yaw_deg in (-89.0, -60.0, -45.0, -30.0, 0.0, 17.0, 30.0, 47.0, 65.0, 89.0):
            _, data = _settled(expert.model, "milk", (0.30, 0.0, yaw_deg))
            plan = expert.plan_grasp(data, "milk")
            self.assertFalse(plan["round"], yaw_deg)
            finger = plan["finger_axis_yaw"]
            width = geo.horizontal_width(expert.model, data, "milk", finger)
            self.assertLess(width, 0.042, yaw_deg)                       # a face is 4.0 cm, a diagonal 5.7 cm
            self.assertLessEqual(abs(plan["hand_yaw"]), np.pi / 4 + 1e-9, yaw_deg)   # the nearer of the two faces

    def test_round_and_near_round_footprints_keep_the_home_yaw(self):
        for obj in ("can", "bottle"):
            expert = PropsExpert((obj,), "left")
            for yaw_deg in (-70.0, 0.0, 33.0, 80.0):
                _, data = _settled(expert.model, obj, (0.30, 0.0, yaw_deg))
                plan = expert.plan_grasp(data, obj)
                self.assertTrue(plan["round"], (obj, yaw_deg))
                self.assertEqual(plan["hand_yaw"], 0.0)

    def test_grasp_point_is_below_the_top(self):
        for obj, top_expected in (("milk", 0.144), ("lemon", 0.043), ("cereal", 0.15)):
            expert = PropsExpert((obj,), "left")
            _, data = _settled(expert.model, obj, START)
            point = expert.plan_grasp(data, obj)["hand_point"]
            below = min(c.GRASP_BELOW_TOP_M, top_expected / 2)
            self.assertAlmostEqual(point[2], top_expected - below + c.HAND_OFFSET_M, delta=0.004, msg=obj)


class EpisodeTests(unittest.TestCase):
    def _run(self, obj, bin_colour, blue_slot="left", pose=START, render=False):
        expert = PropsExpert((obj,), blue_slot)
        scene, _ = _settled(expert.model, obj, pose)
        frames, traces = [], []
        result = expert.run(scene["initial_state"], _episode(obj, bin_colour), lambda f, t: (frames.append(f), traces.append(t)),
                            render=render)
        return expert, result, frames, traces

    def test_every_object_is_placed_in_the_bin_and_released(self):
        for obj in c.OBJECTS:
            with self.subTest(obj=obj):
                _, result, _, _ = self._run(obj, "blue")
                placed, final = result["final_placement"], result["final_metrics"]
                self.assertTrue(placed["resting_in_bin"], (obj, placed))
                self.assertLess(placed["bottom_z_m"], 0.02, (obj, placed))          # on the bin floor
                self.assertTrue(final["gripper_open"] and not final["finger_contact"], (obj, final))

    def test_written_success_for_objects_whose_centre_can_sit_below_the_wall_top(self):
        for obj in ("can", "bread", "lemon", "bottle"):
            with self.subTest(obj=obj):
                _, result, _, _ = self._run(obj, "blue")
                self.assertTrue(result["success"], (obj, result["reason"], result["final_metrics"]))
                self.assertGreaterEqual(result["stable_ticks"], c.SUCCESS_TICKS)

    def test_a_tall_box_stands_upright_so_its_centre_stays_above_the_wall_top(self):
        # Finding, not a goal: the written definition (centre below the 7 cm wall top) cannot be met by
        # a 15 cm box standing on the bin floor, whatever the expert does. The definition is not changed.
        _, result, _, _ = self._run("cereal", "blue")
        self.assertLess(result["final_placement"]["tilt_deg"], 5.0)
        self.assertGreater(result["final_metrics"]["z_m"], c.BIN_WALL_HEIGHT_M)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "not_in_bin")

    def test_the_other_bin_and_the_other_arrangement(self):
        for blue_slot in ("left", "right"):
            for bin_colour in ("blue", "orange"):
                with self.subTest(slot=blue_slot, bin=bin_colour):
                    _, result, _, _ = self._run("lemon", bin_colour, blue_slot=blue_slot)
                    self.assertTrue(result["success"], (result["reason"], result["final_metrics"]))

    def test_the_finger_axis_follows_the_object_at_the_grasp(self):
        expert, result, _, traces = self._run("cereal", "blue", pose=(0.30, 0.0, 45.0))
        model = expert.model
        close = next(t for t in traces if t["phase_index"] == c.PHASE_NAMES.index("close"))
        data = mujoco.MjData(model)
        data.qpos[:] = close["qpos"]
        mujoco.mj_forward(model, data)
        gap = data.xpos[model.body("right_finger").id] - data.xpos[model.body("left_finger").id]
        measured = np.arctan2(gap[1], gap[0]) % np.pi
        plan_data = mujoco.MjData(model)
        mujoco.mj_setState(model, plan_data, result["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, plan_data)
        narrow = geo.narrow_axis_yaw(model, plan_data, "cereal") % np.pi
        difference = (measured - narrow + np.pi / 2) % np.pi - np.pi / 2
        self.assertLess(abs(np.degrees(difference)), 5.0)

    def test_frames_and_traces_have_the_dataset_schema(self):
        _, result, frames, traces = self._run("can", "orange", render=True)
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(len(frames), len(traces))
        self.assertEqual(len(frames), result["frames"])
        frame, trace = frames[10], traces[10]
        self.assertEqual(frame["observation.state"].shape, (9,))
        self.assertEqual(frame["observation.state"].dtype, np.float32)
        self.assertEqual(frame["action"].shape, (8,))
        self.assertEqual(frame["action"].dtype, np.float32)
        self.assertEqual(frame["task"], task_text("can", "orange"))
        self.assertEqual(frame["task"], "put the can in the orange bin")
        for key in ("observation.images.image", "observation.images.image2"):
            self.assertEqual(frame[key].shape, (512, 512, 3))
            self.assertGreater(float(frame[key].std()), 1.0)
        self.assertEqual(trace["props_pose"].shape, (1, 7))
        self.assertEqual(trace["image_sha256"].shape, (2,))
        self.assertTrue({"qpos", "ctrl", "sim_time", "substeps", "phase_index"} <= set(trace))
        self.assertEqual(set(frame["action"][7:].tolist()) <= {0.0, 255.0}, True)

    def test_a_missed_grasp_is_reported_as_grasp(self):
        expert = PropsExpert(("milk",), "left")
        scene, _ = _settled(expert.model, "milk", START)
        orig = expert._lifted
        expert._lifted = lambda *args, **kwargs: False
        result = expert.run(scene["initial_state"], _episode("milk", "blue"), lambda f, t: None, render=False)
        expert._lifted = orig
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "grasp")


if __name__ == "__main__":
    unittest.main()
