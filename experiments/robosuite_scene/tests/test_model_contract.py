import unittest
import numpy as np
import mujoco

class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.robosuite_scene.scene import build_sim
        from src.robots.franka_panda.config import SCENE_PATH
        _, cls.model, cls.data = build_sim()
        cls.current = mujoco.MjModel.from_xml_path(SCENE_PATH)

    def test_names_the_shared_code_looks_up(self):
        for body in ("left_finger", "right_finger", "target_cube_red", "target_cube_green"):
            self.model.body(body)

    def test_qpos_layout_and_finger_sign(self):
        m = self.model
        self.assertEqual([int(m.jnt_qposadr[i]) for i in range(9)], list(range(9)))
        np.testing.assert_allclose(m.jnt_range[7:9], [[0, 0.04], [0, 0.04]])
        np.testing.assert_allclose(m.jnt_range[:7], self.current.jnt_range[:7])
        d = mujoco.MjData(m)
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        gap = d.xpos[m.body("right_finger").id] - d.xpos[m.body("left_finger").id]
        self.assertAlmostEqual(float(np.linalg.norm(gap)), 0.08, places=3)   # open = 8 cm apart

    def test_actuators_match_the_current_scene(self):
        m, c = self.model, self.current
        self.assertEqual(m.nu, 8)
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_allclose(getattr(m, field), getattr(c, field), err_msg=field)
        self.assertEqual(int(m.actuator_trntype[7]), int(mujoco.mjtTrn.mjTRN_TENDON))

    def test_joint_dynamics_and_integrator_match(self):
        m, c = self.model, self.current
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(m, field)[:9], getattr(c, field)[:9], err_msg=field)
        self.assertEqual(int(m.opt.integrator), int(c.opt.integrator))
        self.assertAlmostEqual(float(m.opt.timestep), 1 / 600)
        self.assertEqual(int(m.neq), 1)                                       # finger coupling

    def test_cube_contact_parameters(self):
        m, c = self.model, self.current
        for name in ("target_cube_red", "target_cube_green"):
            self.assertAlmostEqual(float(m.body_mass[m.body(name).id]), 0.05, places=6)
            ours = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(name).id and m.geom_contype[g]]
            theirs = c.geom(f"cube_geom_{name.rsplit('_', 1)[1]}").id
            self.assertEqual(len(ours), 1)
            for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"):
                np.testing.assert_allclose(getattr(m, field)[ours[0]], getattr(c, field)[theirs], err_msg=field)

    def test_only_visual_geoms_are_drawn(self):
        self.assertFalse(np.any(self.model.geom_group == 0))
        self.assertFalse(np.any(self.model.site_group < 3))
