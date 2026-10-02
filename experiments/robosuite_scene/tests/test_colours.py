import unittest

import mujoco
import numpy as np


class ColourTests(unittest.TestCase):
    def test_four_cube_model_matches_the_current_scene_cubes(self):
        from experiments.robosuite_scene.scene import build_sim
        from src.robots.franka_panda.config import SCENE_PATH
        _, model, _ = build_sim(colours=("red", "green", "blue", "yellow"))
        current = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.assertEqual(model.nq, 9 + 4 * 7)
        for colour in ("red", "green", "blue", "yellow"):
            body = model.body(f"target_cube_{colour}")
            self.assertAlmostEqual(float(model.body_mass[body.id]), 0.05, places=6)
            ours = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id and model.geom_contype[g]]
            theirs = current.geom(f"cube_geom_{colour}").id
            self.assertEqual(len(ours), 1)
            for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"):
                np.testing.assert_allclose(getattr(model, field)[ours[0]], getattr(current, field)[theirs])
            visible = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id and model.geom_group[g] < 3]
            np.testing.assert_allclose(model.geom_rgba[visible[0]][:3], current.geom_rgba[theirs][:3], atol=1e-6)

    def test_only_requested_cubes_exist(self):
        from experiments.robosuite_scene.scene import build_sim
        _, model, _ = build_sim(colours=("blue", "yellow"))
        names = {model.body(i).name for i in range(model.nbody)}
        self.assertEqual({n for n in names if n.startswith("target_cube_")},
                         {"target_cube_blue", "target_cube_yellow"})
        self.assertEqual(model.nq, 9 + 2 * 7)

    def test_default_is_still_red_and_green(self):
        from experiments.robosuite_scene.scene import build_sim
        _, model, _ = build_sim()
        self.assertEqual(model.nq, 23)

    def test_bad_colour_sets_are_rejected(self):
        from experiments.robosuite_scene.scene import build_sim
        for bad in (("red",), ("red", "red"), ("red", "purple")):
            with self.assertRaises(ValueError):
                build_sim(colours=bad)


if __name__ == "__main__":
    unittest.main()
