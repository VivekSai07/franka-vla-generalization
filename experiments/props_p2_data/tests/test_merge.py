import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.props_p2_data.merge import aggregate, verify
from experiments.props_p2_data.record import create_dataset

TASKS = ("put the can in the blue bin", "put the milk in the orange bin", "put the lemon in the blue bin")


def _shard(root, k, episodes, frames=3):
    dataset = create_dataset(root, f"props_p2_shard_{k:02d}")
    rng = np.random.RandomState(k)
    for e in range(episodes):
        task = TASKS[(k + e) % len(TASKS)]
        for _ in range(frames):
            dataset.add_frame({"observation.state": rng.rand(9).astype(np.float32),
                               "action": rng.rand(8).astype(np.float32), "task": task,
                               "observation.images.image": rng.randint(0, 255, (512, 512, 3), dtype=np.uint8),
                               "observation.images.image2": rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)})
        dataset.save_episode()
        dataset.finalize()
    return Path(root) / "dataset"


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.shards = [_shard(base / "s0", 0, 2), _shard(base / "s1", 1, 3)]
        cls.merged = base / "merged"
        aggregate(cls.shards, cls.merged)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_merge_equals_the_shards_in_order(self):
        # Review Focus 5.
        report = verify(self.shards, self.merged)
        self.assertEqual(report["episodes"], 5)
        self.assertEqual(report["frames"], 15)

    def test_verify_refuses_the_wrong_shard_order(self):
        with self.assertRaises(ValueError):
            verify(list(reversed(self.shards)), self.merged)

    def test_verify_refuses_a_missing_shard(self):
        with self.assertRaises(ValueError):
            verify(self.shards[:1], self.merged)


if __name__ == "__main__":
    unittest.main()
