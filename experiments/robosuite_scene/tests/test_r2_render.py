import os
import unittest
import numpy as np


class R2RenderTests(unittest.TestCase):
    def test_two_cameras_are_nonblank_and_opencv(self):
        self.assertEqual(os.environ.get("MUJOCO_GL"), "osmesa")
        from experiments.robosuite_scene.scene import build_sim
        import mujoco
        env, model, data = build_sim()
        self.assertIsInstance(model, mujoco.MjModel)
        data.qpos[:] = env.sim.data.qpos          # same joint order; compare like with like
        mujoco.mj_forward(model, data)
        link0 = data.xpos[model.body("robot0_link0").id]
        np.testing.assert_allclose(link0, [0, 0, 0], atol=1e-9)
        table = model.geom("table_collision")
        top = data.geom_xpos[table.id][2] + model.geom_size[table.id][2]
        self.assertAlmostEqual(float(top), 0.0, places=6)
        self.assertLess(float(model.geom_pos[model.geom("floor").id][2]), -0.04)
        import sys
        self.assertNotIn("experiments.robosuite_scene.mujoco_compat", sys.modules)
        names = ("front_cam", "gripper_cam")
        # Two OSMesa contexts cannot be interleaved (the second render reads garbage) and closing ours
        # would break robosuite's, so render everything robosuite owns first, then ours.
        theirs_by_name = {n: env.sim.render(camera_name=n, width=512, height=512) for n in names}
        option = mujoco.MjvOption()
        option.geomgroup[0] = 0   # robosuite hides group 0 (collision geoms) in its camera renders
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            ours_by_name = {}
            for n in names:
                renderer.update_scene(data, camera=n, scene_option=option)
                ours_by_name[n] = renderer.render().copy()
        for name in names:
            ours, theirs = ours_by_name[name], theirs_by_name[name]
            self.assertEqual(ours.shape, (512, 512, 3))
            self.assertEqual(theirs.shape, (512, 512, 3))
            self.assertGreater(float(ours.std()), 0.0)
            self.assertGreater(float(theirs.std()), 0.0)
            # env.sim.render is always bottom-up (OpenGL); robosuite flips it to top-down (OpenCV)
            # only when building observations. Ours must match the OpenCV form, not the raw one.
            unflipped = float(np.abs(ours.astype(np.int16) - theirs[::-1].astype(np.int16)).mean())
            flipped = float(np.abs(ours.astype(np.int16) - theirs.astype(np.int16)).mean())
            print(name, "opencv-vs-ours", unflipped, "opengl-vs-ours", flipped)
            self.assertLess(unflipped, flipped)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(env.action_dim))
