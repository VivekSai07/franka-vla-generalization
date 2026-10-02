import json
import hashlib
import unittest

import mujoco
import numpy as np

from experiments.b3_robosuite_data.plan import make_plan
from experiments.b3_robosuite_data.scenes import (
    build_model, make_bank, present_colours, reset, validate_geometry,
)


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = make_plan()
        cls.episode = next(e for e in cls.plan["episodes"] if len(e["distractors"]) == 2)

    def test_models_are_cached_per_colour_set_in_any_order(self):
        e = self.episode
        a = build_model(tuple(e["colours"]))
        self.assertIs(a, build_model(tuple(reversed(e["colours"]))))
        self.assertEqual(present_colours(a), e["colours"])
        self.assertEqual(a.nq, 9 + 7 * len(e["colours"]))

    def test_reset_is_deterministic_and_matches_the_plan(self):
        from src.robots.franka_panda.config import Q_HOME
        e = self.episode
        model = build_model(tuple(e["colours"]))
        first = reset(model, e["cube_xy_m"], e["home_pose_jitter_deg"])
        second = reset(model, e["cube_xy_m"], e["home_pose_jitter_deg"])
        np.testing.assert_array_equal(first["qpos"], second["qpos"])
        np.testing.assert_array_equal(first["initial_state"], second["initial_state"])
        data = mujoco.MjData(model)
        data.qpos[:] = first["qpos"]
        mujoco.mj_forward(model, data)
        validate_geometry(model, data, e["cube_xy_m"])
        for colour, xy in e["cube_xy_m"].items():
            body = model.body(f"target_cube_{colour}").id
            self.assertLess(float(np.hypot(*(data.xpos[body][:2] - np.asarray(xy)))), 1e-3)
            self.assertLess(abs(float(data.xpos[body][2]) - 0.021), 1e-3, colour)
        home = np.asarray(Q_HOME[:7]) + np.deg2rad(e["home_pose_jitter_deg"])
        worst = float(np.max(np.abs(first["qpos"][:7] - home)))
        print(f"max arm joint error after settling: {worst:.2e} rad")
        self.assertLess(worst, 5e-3)
        np.testing.assert_allclose(first["qpos"][7:9], 0.04, atol=1e-3)

    def test_reset_controls(self):
        e = self.episode
        model = build_model(tuple(e["colours"]))
        from data_collection.benchmark import restore_state
        from src.robots.franka_panda.config import Q_HOME
        scene = reset(model, e["cube_xy_m"], e["home_pose_jitter_deg"])
        data = mujoco.MjData(model)
        restore_state(model, data, scene["initial_state"])
        home = np.asarray(Q_HOME[:7]) + np.deg2rad(e["home_pose_jitter_deg"])
        np.testing.assert_allclose(data.ctrl[:7], home, atol=1e-12)
        self.assertEqual(data.ctrl[-1], 255)

    def test_geometry_check_rejects_a_moved_cube(self):
        e = self.episode
        model = build_model(tuple(e["colours"]))
        scene = reset(model, e["cube_xy_m"], e["home_pose_jitter_deg"])
        data = mujoco.MjData(model)
        data.qpos[:] = scene["qpos"]
        colour = e["pick"]
        adr = int(model.joint(int(model.body(f"target_cube_{colour}").jntadr[0])).qposadr[0])
        data.qpos[adr] += 0.005
        mujoco.mj_forward(model, data)
        with self.assertRaises(ValueError):
            validate_geometry(model, data, e["cube_xy_m"])

    def test_reset_rejects_wrong_cubes(self):
        e = self.episode
        model = build_model(tuple(e["colours"]))
        xy = dict(e["cube_xy_m"])
        xy.pop(e["distractors"][0])
        with self.assertRaises(ValueError):
            reset(model, xy, e["home_pose_jitter_deg"])

    def test_bank(self):
        arrays, manifest = make_bank(self.plan)
        self.assertEqual(len(manifest), 100)
        for key, entry in manifest.items():
            self.assertIn(f"{key}:qpos", arrays)
            self.assertIn(f"{key}:initial_state", arrays)
            self.assertEqual(len(entry["scene_sha256"]), 64)
        again = make_bank(self.plan)[1]
        self.assertEqual(manifest, again)
        print("BANK_SHA256", hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
