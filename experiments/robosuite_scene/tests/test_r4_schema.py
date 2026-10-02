import hashlib
import unittest
from pathlib import Path

import numpy as np

BENCHMARK_SHA256 = "1c11e65b620da8d4aa3b89cba46d1fc832e718f5d6a5615104a4be05d03c10e4"


class R4SchemaTests(unittest.TestCase):
    def test_frame_matches_lerobot_schema(self):
        from experiments.paired_demonstrations.media import CAMERAS, FEATURES
        from experiments.robosuite_scene.scene import build_sim, overwrite_reset
        from experiments.robosuite_scene.validate import frame_from_sim
        env, model, data = build_sim()
        overwrite_reset(model, data)
        frame = frame_from_sim(model, data, np.zeros(8), "stack the red cube on the green cube")
        again = frame_from_sim(model, data, np.zeros(8), "stack the red cube on the green cube")
        state = frame["observation.state"]
        self.assertEqual((state.dtype, state.shape), (np.dtype("float32"), FEATURES["observation.state"]["shape"]))
        act = frame["action"]
        self.assertEqual((act.dtype, act.shape), (np.dtype("float32"), FEATURES["action"]["shape"]))
        self.assertIsInstance(frame["task"], str)
        for key in CAMERAS:
            image = frame[key]
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, FEATURES[key]["shape"])
            self.assertGreater(float(image.std()), 0.0)
            self.assertTrue(np.array_equal(image, again[key]), f"{key} unstable between renders")
        for finger in state[7:9]:
            self.assertAlmostEqual(float(finger), 0.04, delta=1e-3)
        with self.assertRaises(ValueError):
            frame_from_sim(model, data, np.zeros(7), "t")

    def test_benchmark_module_is_unmodified(self):
        path = Path(__file__).resolve().parents[3] / "data_collection" / "benchmark.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), BENCHMARK_SHA256)


if __name__ == "__main__":
    unittest.main()
