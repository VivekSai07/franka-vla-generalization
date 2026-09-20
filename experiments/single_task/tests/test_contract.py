import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.single_task import contract as c


def manifest():
    return {"splits": {"train": list(range(300)), "validation": list(range(300, 420))},
            "fps": 30, "camera_names": ["front_cam", "gripper_cam"], "camera_size": [512, 512],
            "control_dt_requested": 1/30, "validation_required": True,
            "episodes": [dict(episode_index=i, split="train" if i < 300 else "validation",
                              pick="red", place="green", colors=["red", "green"], frames=528,
                              instruction=c.TASK, file=f"episode_{i:06d}.npz", sha256="a"*64,
                              scene_hash=f"{i:064x}", seed=i, slot_id=str(i)) for i in range(420)]}


class ContractTests(unittest.TestCase):
    def test_exact_selection(self):
        entries = c.select_entries(manifest())
        self.assertEqual([e["episode_index"] for e in entries], list(c.TRAIN + c.VAL))

    def test_manifest_membership_not_position(self):
        m = manifest()
        m["episodes"].reverse()
        self.assertEqual(c.select_entries(m)[0]["episode_index"], 0)
        m["splits"]["train"][0], m["splits"]["validation"][0] = 300, 0
        with self.assertRaises(ValueError):
            c.select_entries(m)

    def test_subset_rejects_wrong_split_pair_count_task_frames(self):
        for key, value in (("split", "validation"), ("pick", "green"),
                           ("colors", ["red", "green", "blue"]), ("instruction", "other"),
                           ("frames", 527), ("sha256", "")):
            with self.subTest(key=key):
                m = manifest()
                m["episodes"][0][key] = value
                with self.assertRaises(ValueError):
                    c.select_entries(m)

    def test_duplicates_and_split_overlap_rejected(self):
        m = manifest()
        m["episodes"][1] = copy.deepcopy(m["episodes"][0])
        with self.assertRaises(ValueError):
            c.select_entries(m)
        m = manifest()
        m["splits"]["validation"][0] = 0
        with self.assertRaises(ValueError):
            c.select_entries(m)

    def test_schedule(self):
        self.assertAlmostEqual(c.learning_rate(1), 2e-7)
        self.assertEqual(c.learning_rate(500), 1e-4)
        self.assertEqual(c.learning_rate(5000), 1e-5)
        self.assertTrue(all(c.learning_rate(i) >= c.learning_rate(i+1) for i in range(500, 5000)))
        for invalid in (0, 5001, True, 1.5):
            with self.assertRaises(ValueError):
                c.learning_rate(invalid)

    def test_episode_rows_complete_starts_and_probes(self):
        episodes = [dict(episode_index=i, dataset_from_index=i*528,
                         dataset_to_index=(i+1)*528) for i in range(420)]
        rows, starts, probes = c.indices(episodes)
        self.assertEqual(len(rows), 4752)
        self.assertEqual(len(starts), 4311)
        self.assertEqual(probes["training"][:5], [0, 30, 60, 107, 347])
        self.assertEqual(len(probes["validation"]), 20)
        self.assertEqual(probes["validation"][0], 300*528)
        self.assertNotIn(479, starts)
        self.assertTrue(set(rows).isdisjoint(probes["validation"]))
        episodes[0]["dataset_to_index"] -= 1
        with self.assertRaises(ValueError):
            c.indices(episodes)

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp)/"full"
            c.claim_output(allowed, allowed)
            with self.assertRaises(FileExistsError):
                c.claim_output(allowed, allowed)
            with self.assertRaises(ValueError):
                c.claim_output(Path(tmp)/"elsewhere", allowed)

    def test_no_symlink_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"link").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                c.claim_output(root/"link"/"new", root/"link"/"new")

    def test_eval_count_and_actual_baseline_seed(self):
        trials = c.trials(manifest())
        self.assertEqual(len(trials), 13)
        for entry, ordinal, seed in trials:
            self.assertEqual(ordinal, entry["episode_index"]+1)
            self.assertEqual(seed, 20260917+entry["episode_index"])
            self.assertEqual(seed, 20260916+ordinal)
        self.assertEqual(trials[9][1], 301)

    def test_provisional_counts_not_generalization(self):
        rows = [dict(episode_index=i, task_success=i not in (8, 303)) for i in c.TRAIN+c.VAL]
        summary = c.summarize(rows)
        self.assertEqual(summary["seen"], {"trials": 9, "successes": 8, "goal": 8})
        self.assertEqual(summary["held_out"], {"trials": 4, "successes": 3, "goal": 3})
        self.assertTrue(summary["provisional_capability"])
        self.assertFalse(summary["statistical_generalization"])
        self.assertFalse(c.summarize(rows[:9])["provisional_capability"])
        with self.assertRaises(ValueError):
            c.summarize(rows+[rows[0]])

    def test_inventory_detects_training_byte_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights"
            path.write_bytes(b"original")
            before = c.inventory(Path(tmp))
            path.write_bytes(b"changed")
            self.assertNotEqual(before, c.inventory(Path(tmp)))

    def test_absolute_reload_gate_accepts_inclusive_boundary(self):
        for error in (0., 1e-5):
            with self.subTest(error=error):
                c.require_absolute_reload({"reload_max_action_error": error})

    def test_absolute_reload_gate_rejects_missing_nonfinite_and_out_of_range(self):
        for marker in ({}, *({"reload_max_action_error": error} for error in
                            (-1e-8, 1.000001e-5, float("nan"), float("inf"), -float("inf")))):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                c.require_absolute_reload(marker)

    def test_failure_record_survives_nonfinite_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            c.save_failure(Path(tmp), {"metric": float("nan")}, ValueError("original failure"))
            saved = json.loads((Path(tmp)/"failure.json").read_text())
            self.assertEqual(saved["error"], "original failure")
            self.assertEqual(saved["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()