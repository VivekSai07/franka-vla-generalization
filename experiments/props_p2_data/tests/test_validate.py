import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.props_p2_data import plan as p
from experiments.props_p2_data import validate as v

PLAN = p.make_plan()
EPISODE = next(e for e in PLAN["episodes"] if e["distractor_count"] == 2)


def _frames(episode, n=p.EXPECTED_EPISODE_FRAMES, images=True):
    frame = {"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32), "task": episode["instruction"]}
    if images:
        frame.update({k: np.zeros((512, 512, 3), np.uint8) for k in v.CAMERA_KEYS})
    return [frame] * n


def _result(episode, success=True):
    return {"success": success, "stable_ticks": 30 if success else 0, "frames": p.EXPECTED_EPISODE_FRAMES,
            "distractors": [d["object"] for d in episode["distractors"]],
            "distractor_motion": [{"object": d["object"], "displacement_m": 0.0, "tilt_deg": 0.0}
                                  for d in episode["distractors"]]}


OK_REPLAY = lambda result, frames, traces: {"ok": True, "max_prop_pose_error": 0.0}
BAD_REPLAY = lambda result, frames, traces: {"ok": False, "max_prop_pose_error": 0.5}
UNTOUCHED = lambda result, frames, traces: {"touched": []}


def _touched_by(partner):
    return lambda result, frames, traces: {"touched": [{"object": result["distractors"][0], "partner": partner, "tick": 80}]}


def _diverged(result, frames, traces):
    raise ValueError("contact re-run diverged from the recorded trace at tick 3")


class AdmitTests(unittest.TestCase):
    def admit(self, episode=EPISODE, images=True, contacts=UNTOUCHED, **kwargs):
        replay = kwargs.pop("replay", OK_REPLAY)
        frames = kwargs.pop("frames", _frames(episode, images=images))
        return v.admit(episode, _result(episode, **kwargs), frames, [{}] * len(frames), replay=replay,
                       contacts=contacts, images=images)

    def test_a_clean_episode_is_admitted(self):
        verdict = self.admit()
        self.assertTrue(verdict["admitted"], verdict)
        self.assertEqual(verdict["reason"], "admitted")

    def test_any_touch_of_a_distractor_rejects(self):
        for partner in ("left_finger", "robot0_link7", "prop_milk", "bin_blue"):
            verdict = self.admit(contacts=_touched_by(partner))
            self.assertEqual(verdict["reason"], "distractors", partner)
            self.assertEqual(verdict["details"]["distractor_touched"][0]["partner"], partner)

    def test_a_diverging_contact_rerun_rejects(self):
        verdict = self.admit(contacts=_diverged)
        self.assertEqual(verdict["reason"], "distractors")
        self.assertIn("diverged", verdict["details"]["distractor_error"])

    def test_results_for_the_wrong_objects_reject(self):
        verdict = v.admit(EPISODE, {**_result(EPISODE), "distractors": []}, _frames(EPISODE),
                          [{}] * p.EXPECTED_EPISODE_FRAMES, replay=OK_REPLAY, contacts=UNTOUCHED)
        self.assertEqual(verdict["reason"], "distractors")

    def test_p1_gates_still_reject(self):
        self.assertEqual(self.admit(success=False)["reason"], "expert_success")
        self.assertEqual(self.admit(replay=BAD_REPLAY)["reason"], "replay")
        self.assertEqual(self.admit(frames=_frames(EPISODE, n=451))["reason"], "frames")
        wrong = [{**f, "task": "put the can in the blue bin"} for f in _frames(EPISODE)]
        if EPISODE["instruction"] != "put the can in the blue bin":
            self.assertEqual(self.admit(frames=wrong)["reason"], "labels")
        bad = [{**f, "observation.state": np.zeros(9, np.float64)} for f in _frames(EPISODE)]
        self.assertEqual(self.admit(frames=bad)["reason"], "schema")

    def test_unrendered_gate_runs_skip_only_the_image_check(self):
        self.assertTrue(self.admit(images=False)["admitted"])
        self.assertEqual(v.admit(EPISODE, _result(EPISODE), _frames(EPISODE, images=False),
                                 [{}] * p.EXPECTED_EPISODE_FRAMES, replay=OK_REPLAY, contacts=UNTOUCHED)["reason"], "schema")


def _journal(r_for=None):
    r_for = r_for or {}
    return [{"slot": s, "admitted": True, "r": r_for.get(s, 0)} for s in range(p.ADMIT_COUNT)]


def _meta(episodes):
    return {"total_episodes": len(episodes), "total_frames": p.EXPECTED_EPISODE_FRAMES * len(episodes), "fps": 30,
            "task_table": sorted({e["instruction"] for e in PLAN["episodes"]}),
            "tasks": [[e["instruction"]] for e in episodes], "lengths": [p.EXPECTED_EPISODE_FRAMES] * len(episodes),
            "indices": list(range(len(episodes)))}


class DatasetTests(unittest.TestCase):
    def test_a_complete_dataset_validates(self):
        summary = v.validate_dataset("x", PLAN, _journal(), reader=lambda root: _meta(PLAN["episodes"]))
        self.assertEqual((summary["episodes"], summary["train"], summary["validation"]), (540, 480, 60))

    def test_out_of_order_journal_or_dataset_is_refused(self):
        journal = _journal()
        journal[3], journal[4] = journal[4], journal[3]
        with self.assertRaises(ValueError):
            v.validate_dataset("x", PLAN, journal, reader=lambda root: _meta(PLAN["episodes"]))
        meta = _meta(PLAN["episodes"])
        meta["tasks"][0], meta["tasks"][1] = meta["tasks"][1], meta["tasks"][0]
        if meta["tasks"][0] != meta["tasks"][1]:
            with self.assertRaises(ValueError):
                v.validate_dataset("x", PLAN, _journal(), reader=lambda root: meta)
        meta = {**_meta(PLAN["episodes"]), "indices": [1, 0] + list(range(2, 540))}
        with self.assertRaises(ValueError):
            v.validate_dataset("x", PLAN, _journal(), reader=lambda root: meta)

    def test_a_missing_episode_is_refused(self):
        with self.assertRaises(ValueError):
            v.validate_dataset("x", PLAN, _journal()[:-1], reader=lambda root: _meta(PLAN["episodes"][:-1]))


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_ignores_seal_and_json_key_order_and_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps({"x": 1, "y": 2}))
            (root / "b.bin").write_bytes(b"abc")
            first = v.fingerprint(root)
            (root / "seal.json").write_text("{}")
            (root / "a.json").write_text(json.dumps({"y": 2, "x": 1}))
            self.assertEqual(v.fingerprint(root), first)
            (root / "b.bin").write_bytes(b"abd")
            self.assertNotEqual(v.fingerprint(root), first)
            os.symlink(root / "b.bin", root / "c.bin")
            with self.assertRaises(ValueError):
                v.fingerprint(root)


if __name__ == "__main__":
    unittest.main()
