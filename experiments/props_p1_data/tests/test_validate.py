import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.props_p1_data import plan as p
from experiments.props_p1_data.validate import admit, fingerprint, seal, validate_dataset


class Fixture:
    def __init__(self):
        self.plan = p.make_plan()
        self.episode = self.plan["episodes"][0]
        self.text = self.episode["instruction"]

    def good(self):
        e = self.episode
        n = p.EXPECTED_EPISODE_FRAMES
        result = {"success": True, "stable_ticks": 30, "frames": n}
        frames = [{"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32),
                   "observation.images.image": np.zeros((512, 512, 3), np.uint8),
                   "observation.images.image2": np.zeros((512, 512, 3), np.uint8), "task": self.text}
                  for _ in range(n)]
        return copy.deepcopy(e), result, frames, [{}] * n


def replay_ok(result, frames, traces):
    return {"ok": True, "max_policy_state_error": 1e-6}


def replay_bad(result, frames, traces):
    return {"ok": False, "max_policy_state_error": 5e-3}


class AdmitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()

    def run_admit(self, mutate=None, replay=replay_ok):
        episode, result, frames, traces = self.fx.good()
        if mutate:
            mutate(episode, result, frames, traces)
        return admit(episode, result, frames, traces, replay=replay)

    def test_a_clean_episode_is_admitted(self):
        verdict = self.run_admit()
        self.assertTrue(verdict["admitted"], verdict)
        self.assertEqual(verdict["reason"], "admitted")
        self.assertTrue(all(verdict["gates"].values()), verdict["gates"])

    def test_expert_failure_is_rejected_first(self):
        def mutate(episode, result, frames, traces):
            result["success"] = False
        verdict = self.run_admit(mutate)
        self.assertFalse(verdict["admitted"])
        self.assertEqual(verdict["reason"], "expert_success")

    def test_short_stable_ticks_fails_expert_success(self):
        def mutate(episode, result, frames, traces):
            result["stable_ticks"] = 29
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "expert_success")

    def test_wrong_frame_count_is_rejected(self):
        def mutate(episode, result, frames, traces):
            frames.pop()
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "frames")

    def test_replay_gate_is_skipped_not_run_when_expert_already_failed(self):
        calls = []

        def spy(*args, **kwargs):
            calls.append(1)
            return replay_bad(*args, **kwargs)

        def mutate(episode, result, frames, traces):
            result["success"] = False
        self.run_admit(mutate, replay=spy)
        self.assertEqual(calls, [])

    def test_a_failed_replay_is_rejected(self):
        verdict = self.run_admit(replay=replay_bad)
        self.assertEqual(verdict["reason"], "replay")

    def test_a_wrong_instruction_fails_labels(self):
        def mutate(episode, result, frames, traces):
            episode["instruction"] = "put the lemon in the orange bin"
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "labels")

    def test_a_frame_whose_task_disagrees_with_the_episode_fails_labels(self):
        def mutate(episode, result, frames, traces):
            frames[5]["task"] = "put the lemon in the orange bin"
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "labels")

    def test_a_wrong_dtype_fails_schema(self):
        def mutate(episode, result, frames, traces):
            frames[0]["action"] = frames[0]["action"].astype(np.float64)
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "schema")

    def test_a_wrong_image_shape_fails_schema(self):
        def mutate(episode, result, frames, traces):
            frames[0]["observation.images.image"] = np.zeros((256, 256, 3), np.uint8)
        verdict = self.run_admit(mutate)
        self.assertEqual(verdict["reason"], "schema")


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_ignores_seal_json_but_changes_on_any_other_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text(json.dumps({"a": 1}))
            before = fingerprint(root)
            (root / "seal.json").write_text(json.dumps({"ignored": True}))
            self.assertEqual(fingerprint(root), before)
            (root / "plan.json").write_text(json.dumps({"a": 2}))
            self.assertNotEqual(fingerprint(root), before)

    def test_fingerprint_is_stable_under_json_key_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text(json.dumps({"a": 1, "b": 2}))
            first = fingerprint(root)
            (root / "plan.json").write_text(json.dumps({"b": 2, "a": 1}))
            self.assertEqual(fingerprint(root), first)

    def test_fingerprint_refuses_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real.json").write_text("{}")
            (root / "link.json").symlink_to(root / "real.json")
            with self.assertRaises(ValueError):
                fingerprint(root)


class ValidateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.plan = p.make_plan()
        self.journal = [{"slot": e["episode_index"], "admitted": True, "r": 0} for e in self.plan["episodes"]]

    def _reader(self):
        episodes = self.plan["episodes"]
        return {
            "total_episodes": len(episodes), "total_frames": p.EXPECTED_EPISODE_FRAMES * len(episodes), "fps": 30,
            "task_table": sorted({e["instruction"] for e in episodes}),
            "tasks": [[e["instruction"]] for e in episodes],
            "lengths": [p.EXPECTED_EPISODE_FRAMES] * len(episodes),
            "indices": [e["episode_index"] for e in episodes],
        }

    def test_a_matching_dataset_validates(self):
        meta = self._reader()
        result = validate_dataset("root", self.plan, self.journal, reader=lambda root: meta)
        self.assertEqual(result["episodes"], 540)
        self.assertEqual(result["train"], 480)
        self.assertEqual(result["validation"], 60)
        self.assertEqual(result["held_out_pairs_collected"], 0)

    def test_a_missing_slot_is_rejected(self):
        meta = self._reader()
        journal = self.journal[:-1]
        with self.assertRaises(ValueError):
            validate_dataset("root", self.plan, journal, reader=lambda root: meta)

    def test_a_wrong_episode_count_is_rejected(self):
        meta = {**self._reader(), "total_episodes": 539}
        with self.assertRaises(ValueError):
            validate_dataset("root", self.plan, self.journal, reader=lambda root: meta)

    def test_a_cell_imbalance_is_rejected(self):
        meta = self._reader()
        meta["tasks"][0] = meta["tasks"][1]  # task table still matches, but episode-vs-slot does not
        with self.assertRaises(ValueError):
            validate_dataset("root", self.plan, self.journal, reader=lambda root: meta)

    def test_a_held_out_pair_sneaking_in_is_rejected(self):
        meta = self._reader()
        tampered_plan = copy.deepcopy(self.plan)
        tampered_plan["episodes"][0]["object"], tampered_plan["episodes"][0]["bin"] = "milk", "blue"
        with self.assertRaises(ValueError):
            validate_dataset("root", tampered_plan, self.journal, reader=lambda root: meta)


if __name__ == "__main__":
    unittest.main()
