import unittest

import torch


class PilotTrainingTests(unittest.TestCase):
    def helpers(self):
        from training import pilot_protocol
        return pilot_protocol

    def manifest(self):
        return {"splits": {"train": list(range(300)), "validation": list(range(300, 420))}}

    def test_exact_manifest_splits(self):
        train, val = self.helpers().checked_splits(self.manifest(), 420)
        self.assertEqual(train, list(range(300)))
        self.assertEqual(val, list(range(300, 420)))

    def test_duplicate_overlap_and_missing_ids_rejected(self):
        for bad in (0, 300, 999):
            manifest = self.manifest()
            manifest["splits"]["train"][1] = bad
            with self.assertRaises(ValueError):
                self.helpers().checked_splits(manifest, 420)
        with self.assertRaises(ValueError):
            self.helpers().checked_splits(self.manifest(), 421)

    def test_selected_indices_and_probes_respect_episodes(self):
        episodes = [{"dataset_from_index": 0, "dataset_to_index": 528},
                    {"dataset_from_index": 528, "dataset_to_index": 1056}]
        self.assertEqual(self.helpers().frame_indices(episodes, [1]), list(range(528, 1056)))
        self.assertEqual(self.helpers().probe_indices(episodes, [1]), [588, 635, 875])
        with self.assertRaises(ValueError):
            self.helpers().probe_indices([{"dataset_from_index": 0, "dataset_to_index": 350}], [0])

    def test_lr_warmup_peak_and_floor(self):
        helper = self.helpers()
        self.assertAlmostEqual(helper.learning_rate(1), 1e-7)
        self.assertAlmostEqual(helper.learning_rate(1000), 1e-4)
        self.assertAlmostEqual(helper.learning_rate(20000), 1e-5)
        self.assertGreater(helper.learning_rate(10000), helper.learning_rate(15000))
        for step in (0, 20001):
            with self.assertRaises(ValueError):
                helper.learning_rate(step)

    def test_batch_rejects_padding_nonfinite_and_wrong_shapes(self):
        batch = {"action": torch.zeros(8, 50, 8), "observation.state": torch.zeros(8, 9),
                 "action_is_pad": torch.zeros(8, 50, dtype=torch.bool),
                 "observation.images.image": torch.zeros(8, 3, 512, 512),
                 "observation.images.image2": torch.zeros(8, 3, 512, 512)}
        self.helpers().check_batch(batch)
        batch["action_is_pad"][0, -1] = True
        with self.assertRaises(ValueError):
            self.helpers().check_batch(batch)
        batch["action_is_pad"][0, -1] = False
        batch["action"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.helpers().check_batch(batch)
        batch["action"] = torch.zeros(7, 50, 8)
        with self.assertRaises(ValueError):
            self.helpers().check_batch(batch)

    def test_scaled_accumulation_matches_full_batch(self):
        helper = self.helpers()
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 1)
        other = torch.nn.Linear(3, 1)
        other.load_state_dict(model.state_dict())
        x, target = torch.randn(32, 3), torch.randn(32, 1)
        torch.nn.functional.mse_loss(model(x), target).backward()
        for start in range(0, 32, 8):
            helper.backward_microbatch(torch.nn.functional.mse_loss(other(x[start:start+8]), target[start:start+8]))
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad)


if __name__ == "__main__":
    unittest.main()