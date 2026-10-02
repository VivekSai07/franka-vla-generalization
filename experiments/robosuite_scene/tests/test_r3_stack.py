import unittest

import numpy as np


class R3StackTests(unittest.TestCase):
    def test_one_scripted_stack(self):
        from experiments.robosuite_scene.expert import run_episode
        result = run_episode(0, sink=lambda frame, trace: None)
        self.assertTrue(result["success"], result["reason"])
        self.assertTrue(result["released_stack"])
        self.assertLess(result["final_xy_m"], 0.02)
        self.assertTrue(0.035 < result["final_z_m"] < 0.045)
        with self.assertRaises(RuntimeError):
            result["env"].step(np.zeros(result["env"].action_dim))


if __name__ == "__main__":
    unittest.main()
