import unittest

import numpy as np


class R5ResetTests(unittest.TestCase):
    def test_reset_is_deterministic_and_settled(self):
        import mujoco
        from experiments.robosuite_scene.contract import FIXED_CUBES
        from experiments.robosuite_scene.scene import build_sim, overwrite_reset
        env, model, data = build_sim()
        first = overwrite_reset(model, data)
        second = overwrite_reset(model, data)
        np.testing.assert_array_equal(first["qpos"], second["qpos"])
        np.testing.assert_array_equal(first["initial_state"], second["initial_state"])
        for name, (x, y, _) in FIXED_CUBES.items():
            pos = data.xpos[model.body(name).id]
            np.testing.assert_allclose(pos[:2], [x, y], atol=1e-3)
            self.assertAlmostEqual(float(pos[2]), 0.02, delta=1e-3)
        floor = model.geom("floor").id
        cube_bodies = {model.body(n).id for n in FIXED_CUBES}
        for contact in data.contact[:data.ncon]:
            geoms = (contact.geom1, contact.geom2)
            has_cube = any(model.geom_bodyid[g] in cube_bodies for g in geoms)
            self.assertFalse(has_cube and floor in geoms)


if __name__ == "__main__":
    unittest.main()
