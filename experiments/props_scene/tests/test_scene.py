import unittest
from pathlib import Path

import mujoco
import numpy as np

from experiments.props_scene import constants as c

OUT = Path("/out/scene_check")


def _subtree_mass(model, body):
    return float(model.body_subtreemass[model.body(body).id])


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.props_scene.scene import build_model
        from src.robots.franka_panda.config import SCENE_PATH
        cls.env, cls.model = build_model(("cereal", "can"), "left")
        cls.current = mujoco.MjModel.from_xml_path(SCENE_PATH)

    def test_only_the_requested_props_and_both_bins(self):
        m = self.model
        props = sorted(m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("prop_"))
        self.assertEqual(props, ["prop_can", "prop_cereal"])
        m.body("bin_blue"), m.body("bin_orange")

    def test_bins_have_the_spec_geometry_and_colours(self):
        m = self.model
        blue = m.body("bin_blue")
        np.testing.assert_allclose(blue.pos, [0.32, 0.33, 0.0], atol=1e-9)    # blue_slot="left" is y > 0
        np.testing.assert_allclose(m.body("bin_orange").pos, [0.32, -0.33, 0.0], atol=1e-9)
        for colour in ("blue", "orange"):
            body = m.body(f"bin_{colour}").id
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
            self.assertEqual(len(geoms), 5)                                    # floor + four walls
            for g in geoms:
                np.testing.assert_allclose(m.geom_rgba[g], c.BIN_RGBA[colour], atol=1e-6)
                self.assertTrue(m.geom_contype[g] and m.geom_conaffinity[g])
                self.assertEqual(int(m.geom_group[g]), 1)                      # drawn
            tops = [m.geom_pos[g][2] + m.geom_size[g][2] for g in geoms]
            self.assertAlmostEqual(max(tops), c.BIN_WALL_HEIGHT_M, places=9)

    def test_bin_inner_box(self):
        m = self.model
        body = m.body("bin_blue").id
        walls = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_size[g][2] > 0.02]
        xs = sorted({round(float(abs(m.geom_pos[g][0])), 6) for g in walls if m.geom_size[g][0] < 0.02})
        ys = sorted({round(float(abs(m.geom_pos[g][1])), 6) for g in walls if m.geom_size[g][1] < 0.02})
        self.assertEqual(xs, [0.115])                                          # inner 0.11 + half wall 0.005
        self.assertEqual(ys, [0.115])

    def test_masses_are_the_spec_values(self):
        for name in ("cereal", "can"):
            self.assertAlmostEqual(_subtree_mass(self.model, f"prop_{name}") / c.MASS_KG[name], 1.0, delta=0.01)

    def test_robot_names_and_layout(self):
        m = self.model
        for body in ("left_finger", "right_finger"):
            m.body(body)
        self.assertEqual([int(m.jnt_qposadr[i]) for i in range(9)], list(range(9)))
        np.testing.assert_allclose(m.jnt_range[7:9], [[0, 0.04], [0, 0.04]])
        np.testing.assert_allclose(m.jnt_range[:7], self.current.jnt_range[:7])
        d = mujoco.MjData(m)
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        gap = d.xpos[m.body("right_finger").id] - d.xpos[m.body("left_finger").id]
        self.assertAlmostEqual(float(np.linalg.norm(gap)), 0.08, places=3)

    def test_actuators_and_dynamics_match_the_current_scene(self):
        m, cur = self.model, self.current
        self.assertEqual(m.nu, 8)
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_allclose(getattr(m, field), getattr(cur, field), err_msg=field)
        self.assertEqual(int(m.actuator_trntype[7]), int(mujoco.mjtTrn.mjTRN_TENDON))
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(m, field)[:9], getattr(cur, field)[:9], err_msg=field)
        self.assertEqual(int(m.opt.integrator), int(cur.opt.integrator))
        self.assertAlmostEqual(float(m.opt.timestep), 1 / 600)
        self.assertEqual(int(m.neq), 1)

    def test_only_visual_geoms_are_drawn(self):
        self.assertFalse(np.any(self.model.geom_group == 0))
        self.assertFalse(np.any(self.model.site_group < 3))

    def test_finger_pads_report_on_the_finger_bodies(self):
        m = self.model
        colliding = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        for g in colliding:
            self.assertFalse(m.body(int(m.geom_bodyid[g])).name.endswith("_tip"), m.geom(g).name)
        for finger in ("left_finger", "right_finger"):
            pads = [g for g in colliding if m.geom_bodyid[g] == m.body(finger).id
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            self.assertEqual(len(pads), 1, finger)

    def test_each_prop_has_one_colliding_geom(self):
        m = self.model
        for name in ("cereal", "can"):
            body = m.body(f"prop_{name}").id
            colliding = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
            self.assertEqual(len(colliding), 1, name)


class ResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.props_scene.scene import build_model
        cls.env, cls.model = build_model(("cereal", "can"), "left")
        cls.poses = {"cereal": (0.30, -0.05, 45.0), "can": (0.25, 0.08, -60.0)}
        cls.jitter = [0.5, -0.5, 0.2, 0.0, 1.0, -1.0, 0.3]

    def _reset(self):
        from experiments.props_scene.scene import reset
        return reset(self.model, self.poses, self.jitter)

    def test_deterministic(self):
        a, b = self._reset(), self._reset()
        self.assertEqual(a["initial_state"].tobytes(), b["initial_state"].tobytes())
        self.assertEqual(a["qpos"].tobytes(), b["qpos"].tobytes())

    def test_props_rest_on_the_table_where_asked(self):
        from experiments.props_scene.geometry import bottom_z
        from experiments.props_scene.scene import object_pose
        scene = self._reset()
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, scene["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, data)
        for name, (x, y, yaw) in self.poses.items():
            self.assertLess(abs(bottom_z(self.model, data, name)), 0.002, name)
            pose = object_pose(self.model, data, name)
            self.assertLess(float(np.hypot(pose["xy"][0] - x, pose["xy"][1] - y)), 0.002, name)
            wrapped = (pose["yaw_deg"] - yaw + 180.0) % 360.0 - 180.0
            self.assertLess(abs(wrapped), 1.0, name)

    def test_arm_is_at_home_plus_jitter(self):
        from src.robots.franka_panda.config import Q_HOME
        scene = self._reset()
        expected = np.asarray(Q_HOME[:7]) + np.deg2rad(self.jitter)
        np.testing.assert_allclose(scene["qpos"][:7], expected, atol=5e-3)

    def test_reset_rejects_wrong_props_or_jitter(self):
        from experiments.props_scene.scene import reset
        with self.assertRaises(ValueError):
            reset(self.model, {"cereal": (0.3, 0.0, 0.0)}, self.jitter)
        with self.assertRaises(ValueError):
            reset(self.model, self.poses, [0.0] * 6)


class RenderTests(unittest.TestCase):
    def test_both_cameras_show_the_scene(self):
        from experiments.props_scene.scene import build_model, reset
        _, model = build_model(("cereal", "can"), "left")
        scene = reset(model, {"cereal": (0.30, -0.05, 45.0), "can": (0.25, 0.08, -60.0)}, [0.0] * 7)
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, scene["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        OUT.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            for camera in ("front_cam", "gripper_cam"):
                renderer.update_scene(data, camera=camera)
                image = renderer.render().copy()
                self.assertGreater(float(image.std()), 1.0, camera)
                Image.fromarray(image).save(OUT / f"reset_{camera}.png")
                if camera == "front_cam":
                    blue = np.all(np.abs(image.astype(float) - np.array([0.15, 0.35, 0.85]) * 255) < 70, axis=-1)
                    orange = np.all(np.abs(image.astype(float) - np.array([0.95, 0.55, 0.10]) * 255) < 70, axis=-1)
                    self.assertGreater(int(blue.sum()), 2000)
                    self.assertGreater(int(orange.sum()), 2000)


if __name__ == "__main__":
    unittest.main()
