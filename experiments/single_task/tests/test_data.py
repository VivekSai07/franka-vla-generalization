"""Read-only CPU integration: actual pilot gates, rows and own train-only statistics."""
import unittest

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from experiments.single_task import contract as c


class ActualDataTests(unittest.TestCase):
    def test_full_original_replay_binding_and_new_source_isolation(self):
        manifest = c.h.require_complete(c.DATA)
        gate = c.h.require_replay_pass(c.DATA, c.ROOT)
        self.assertTrue(gate["passed"])
        self.assertEqual(len(c.select_entries(manifest)), 13)
        self.assertTrue(all(not p.startswith("experiments/single_task/") for p in manifest["source_hashes"]))
        current = c.sources()
        for name in ("contract.py", "runner.py", "evaluate.py", "tests/test_data.py"):
            self.assertIn("experiments/single_task/"+name, current)
        self.assertIn("training/smolvla_pilot.py", current)
        self.assertIn("validation/precision.py", current)

    def test_actual_metadata_train_stats_and_two_camera_sample(self):
        ds = LeRobotDataset(repo_id=c.DATA.name, root=c.DATA, delta_timestamps={"action": [i/30 for i in range(50)]})
        rows, starts, probes = c.indices(ds.meta.episodes)
        ds._ensure_hf_dataset_loaded()
        self.assertEqual({int(ds.hf_dataset[i]["episode_index"]) for i in rows}, set(c.TRAIN))
        self.assertEqual({int(ds.hf_dataset[i]["episode_index"]) for i in probes["validation"]}, set(c.VAL))
        stats = c.h.compute_training_stats(ds.hf_dataset, rows)
        self.assertEqual(set(stats), {"observation.state", "action"})
        for key, size in (("observation.state", 9), ("action", 8)):
            self.assertEqual(stats[key]["mean"].shape, (size,))
            self.assertTrue(all(torch.isfinite(v).all() for v in stats[key].values()))
        selected = c.PolicyRows(ds)
        for index in (starts[0], probes["validation"][0]):
            frame = selected[index]
            self.assertEqual(frame["task"], c.TASK)
            self.assertEqual(frame["action"].shape, (50, 8))
            self.assertFalse(frame["action_is_pad"].any())
            self.assertEqual({k: tuple(frame[k].shape) for k in c.OBS}, c.OBS)
            self.assertEqual(set(c.h.observation_only(frame)), set(c.OBS)|{"task"})


if __name__ == "__main__":
    unittest.main()