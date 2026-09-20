import unittest

import torch

from validation.smolvla_smoke import compute_training_stats, valid_chunk_starts


class LearningSmokeTests(unittest.TestCase):
    def test_overfit_protocol_uses_only_one_training_episode(self):
        from validation.smolvla_smoke import experiment_protocol
        manifest = {"splits": {"train": [0, 1, 2, 3], "validation": [4, 5]}}
        protocol = experiment_protocol(manifest, True)
        self.assertEqual(protocol["train_episodes"], [0])
        self.assertEqual(protocol["eval_episodes"], [0])
        self.assertEqual(protocol["steps"], 2000)
        self.assertEqual(manifest["splits"]["validation"], [4, 5])
        normal = experiment_protocol(manifest, False)
        self.assertEqual(normal["steps"], 100)
        self.assertEqual(normal["train_episodes"], [0, 1, 2, 3])
        self.assertEqual(normal["eval_episodes"], [4, 5])

    def test_statistics_exclude_validation_rows(self):
        rows = [{"observation.state": torch.tensor([value] * 9),
                 "action": torch.tensor([value] * 8)} for value in (1., 3., 1000.)]
        stats = compute_training_stats(rows, [0, 1])
        self.assertEqual(stats["action"]["mean"][0].item(), 2.)
        self.assertEqual(stats["action"]["std"][0].item(), 1.)
        self.assertEqual(stats["observation.state"]["max"][0].item(), 3.)

    def test_chunk_starts_do_not_cross_episode_boundary(self):
        episodes = [{"dataset_from_index": 0, "dataset_to_index": 60},
                    {"dataset_from_index": 60, "dataset_to_index": 120}]
        starts = valid_chunk_starts(episodes, [0], 50)
        self.assertEqual(starts, list(range(11)))
        self.assertNotIn(59, starts)


if __name__ == "__main__":
    unittest.main()