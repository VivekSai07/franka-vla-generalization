import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.props_p3_data import plan as p
from experiments.props_p3_data import validate as v

PLAN = p.make_plan()
EPISODE = PLAN["episodes"][0]
EXPERT_TEXT = lambda e: f"put the {e['object']} in the {e['bin']} bin"   # what the frozen expert writes into a frame


def _frames(n=p.EXPECTED_EPISODE_FRAMES, images=True, task=p.PROMPT):
    frame = {"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32), "task": task}
    if images:
        frame.update({k: np.zeros((512, 512, 3), np.uint8) for k in v.CAMERA_KEYS})
    return [frame] * n


def _result(success=True):
    return {"success": success, "stable_ticks": 30 if success else 0, "frames": p.EXPECTED_EPISODE_FRAMES}


OK_REPLAY = lambda result, frames, traces: {"ok": True, "max_prop_pose_error": 0.0}
BAD_REPLAY = lambda result, frames, traces: {"ok": False, "max_prop_pose_error": 0.5}


def _diverged(result, frames, traces):
    raise ValueError("replay clock drift at tick 3")


class AdmitTests(unittest.TestCase):
    def admit(self, episode=EPISODE, images=True, replay=OK_REPLAY, frames=None, success=True):
        frames = _frames(images=images) if frames is None else frames
        return v.admit(episode, _result(success), frames, [{}] * len(frames), replay=replay, images=images)

    def test_a_clean_episode_is_admitted(self):
        verdict = self.admit()
        self.assertTrue(verdict["admitted"], verdict)
        self.assertEqual(verdict["reason"], "admitted")

    def test_the_expert_wording_is_rejected_by_the_labels_gate(self):
        # Review Focus 3: the expert's own "put the X in the Y bin" must never be accepted as a label.
        verdict = self.admit(frames=_frames(task=EXPERT_TEXT(EPISODE)))
        self.assertEqual(verdict["reason"], "labels")

    def test_a_single_wrong_frame_or_episode_label_rejects(self):
        frames = _frames()
        frames[200] = {**frames[200], "task": "sort the items"}
        self.assertEqual(self.admit(frames=frames)["reason"], "labels")
        self.assertEqual(self.admit(episode={**EPISODE, "instruction": EXPERT_TEXT(EPISODE)})["reason"], "labels")

    def test_a_bin_that_is_not_the_objects_container_rejects(self):
        other = "blue" if EPISODE["bin"] == "orange" else "orange"
        self.assertEqual(self.admit(episode={**EPISODE, "bin": other})["reason"], "labels")

    def test_p1_gates_still_reject(self):
        self.assertEqual(self.admit(success=False)["reason"], "expert_success")
        self.assertEqual(self.admit(replay=BAD_REPLAY)["reason"], "replay")
        self.assertEqual(self.admit(frames=_frames(n=451))["reason"], "frames")
        bad = [{**f, "observation.state": np.zeros(9, np.float64)} for f in _frames()]
        self.assertEqual(self.admit(frames=bad)["reason"], "schema")

    def test_success_needs_the_full_stable_streak(self):
        short = {**_result(), "stable_ticks": v.STABLE_TICKS - 1}
        frames = _frames()
        self.assertEqual(v.admit(EPISODE, short, frames, [{}] * len(frames), replay=OK_REPLAY)["reason"], "expert_success")
        enough = {**_result(), "stable_ticks": v.STABLE_TICKS}
        self.assertTrue(v.admit(EPISODE, enough, frames, [{}] * len(frames), replay=OK_REPLAY)["admitted"])

    def test_frames_traces_and_the_expert_count_must_all_agree(self):
        frames = _frames()
        self.assertEqual(v.admit(EPISODE, _result(), frames, [{}] * 451, replay=OK_REPLAY)["reason"], "frames")
        self.assertEqual(v.admit(EPISODE, {**_result(), "frames": 451}, frames, [{}] * 452, replay=OK_REPLAY)["reason"],
                         "frames")

    def test_camera_images_must_be_uint8_512_squares(self):
        for bad in (np.zeros((512, 512, 3), np.float32), np.zeros((256, 256, 3), np.uint8), np.zeros((512, 512), np.uint8), None):
            frames = [{**f, "observation.images.image2": bad} for f in _frames()]
            self.assertEqual(self.admit(frames=frames)["reason"], "schema")

    def test_a_diverging_replay_rejects_and_says_why(self):
        verdict = self.admit(replay=_diverged)
        self.assertEqual(verdict["reason"], "replay")
        self.assertIn("drift", verdict["details"]["replay_error"])

    def test_unrendered_gate_runs_skip_only_the_image_check(self):
        self.assertTrue(self.admit(images=False)["admitted"])
        self.assertEqual(v.admit(EPISODE, _result(), _frames(images=False), [{}] * p.EXPECTED_EPISODE_FRAMES,
                                 replay=OK_REPLAY)["reason"], "schema")


def _journal(r_for=None):
    r_for = r_for or {}
    return [{"slot": s, "admitted": True, "r": r_for.get(s, 0)} for s in range(p.ADMIT_COUNT)]


def _meta(episodes=None, **override):
    episodes = PLAN["episodes"] if episodes is None else episodes
    meta = {"total_episodes": len(episodes), "total_frames": p.EXPECTED_EPISODE_FRAMES * len(episodes), "fps": 30,
            "task_table": [p.PROMPT], "tasks": [[p.PROMPT]] * len(episodes),
            "lengths": [p.EXPECTED_EPISODE_FRAMES] * len(episodes), "indices": list(range(len(episodes)))}
    meta.update(override)
    return meta


def _plan_with(mutate):
    plan = copy.deepcopy(PLAN)
    mutate(plan["episodes"])
    return plan


class DatasetTests(unittest.TestCase):
    def validate(self, plan=PLAN, journal=None, meta=None):
        meta = _meta() if meta is None else meta
        return v.validate_dataset("x", plan, _journal() if journal is None else journal, reader=lambda root: meta)

    def test_a_complete_dataset_validates(self):
        summary = self.validate()
        self.assertEqual((summary["episodes"], summary["train"], summary["validation"]), (544, 480, 64))
        self.assertEqual(summary["objects"], {"can": 136, "milk": 136, "bread": 136, "lemon": 136})
        self.assertEqual(summary["held_out_objects_collected"], 0)

    def test_the_task_table_must_be_exactly_the_fixed_prompt(self):
        # Review Focus 4, re-derived: P2 matched ten instructions by text; here there is one string and no more.
        with self.assertRaisesRegex(ValueError, "task table"):
            self.validate(meta=_meta(task_table=[p.PROMPT, "put the can in the orange bin"]))
        with self.assertRaisesRegex(ValueError, "task table"):
            self.validate(meta=_meta(task_table=["put the can in the orange bin"]))
        with self.assertRaisesRegex(ValueError, "task table"):
            self.validate(meta=_meta(task_table=[]))

    def test_every_episode_must_carry_the_fixed_prompt(self):
        tasks = [[p.PROMPT]] * 544
        tasks[300] = ["put the milk in the orange bin"]
        with self.assertRaisesRegex(ValueError, "Episode 300"):
            self.validate(meta=_meta(tasks=tasks))
        tasks = [[p.PROMPT]] * 544
        tasks[7] = [p.PROMPT, p.PROMPT]
        with self.assertRaisesRegex(ValueError, "Episode 7"):
            self.validate(meta=_meta(tasks=tasks))

    def test_a_held_out_object_collected_is_refused(self):
        def swap(episodes):
            episodes[0].update(object="bottle", bin="orange", cell=["bottle", "orange", episodes[0]["blue_slot"]])
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.validate(plan=_plan_with(swap))

    def test_a_bin_that_is_not_the_objects_container_is_refused(self):
        def flip(episodes):
            episodes[5]["bin"] = "blue" if episodes[5]["bin"] == "orange" else "orange"
        with self.assertRaisesRegex(ValueError, "container"):
            self.validate(plan=_plan_with(flip))

    def test_cell_balance_is_enforced_for_train_and_validation(self):
        def move_train(episodes):
            a, b = [e for e in episodes if e["split"] == "train" and e["cell"][0] == "can"][:2]
            b.update(cell=["milk", "orange", b["blue_slot"]], object="milk")   # still a trained object and a valid bin
        with self.assertRaisesRegex(ValueError, "balance"):
            self.validate(plan=_plan_with(move_train))

        def move_validation(episodes):
            e = next(e for e in episodes if e["split"] == "validation" and e["object"] == "lemon")
            e.update(cell=["bread", "blue", e["blue_slot"]], object="bread")
        with self.assertRaisesRegex(ValueError, "balance"):
            self.validate(plan=_plan_with(move_validation))

    def test_out_of_order_journal_or_dataset_is_refused(self):
        journal = _journal()
        journal[3], journal[4] = journal[4], journal[3]
        with self.assertRaises(ValueError):
            self.validate(journal=journal)
        with self.assertRaises(ValueError):
            self.validate(meta=_meta(indices=[1, 0] + list(range(2, 544))))

    def test_a_missing_episode_is_refused(self):
        with self.assertRaises(ValueError):
            self.validate(journal=_journal()[:-1], meta=_meta(PLAN["episodes"][:-1]))

    def test_counts_frames_and_fps_are_enforced(self):
        for override in ({"total_frames": 1}, {"fps": 24}, {"lengths": [451] + [452] * 543}, {"total_episodes": 543}):
            with self.assertRaisesRegex(ValueError, "counts", msg=str(override)):
                self.validate(meta=_meta(**override))

    def test_a_substituted_slot_uses_its_substitute_layout(self):
        episodes = v.admitted_episodes(PLAN, _journal({3: 2}))
        self.assertEqual(episodes[3]["layout_id"], f"{PLAN['episodes'][3]['layout_id']}_r2")
        self.assertEqual(episodes[3]["start"], p.resample_start(PLAN["episodes"][3], 2)["start"])
        self.assertNotEqual(episodes[3]["start"], PLAN["episodes"][3]["start"])
        self.assertEqual(episodes[4], PLAN["episodes"][4])

    def test_substituted_slots_are_counted(self):
        self.assertEqual(self.validate(journal=_journal({3: 1, 9: 2}))["substituted_slots"], 2)


class FingerprintAndSealTests(unittest.TestCase):
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

    def test_seal_binds_the_fingerprint_plan_provenance_and_substitutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.bin").write_bytes(b"abc")
            (root / "substitutions.json").write_text(json.dumps({"3": 1}))
            sealed = v.seal(root, PLAN, {"f": "h"})
            self.assertEqual(json.loads((root / "seal.json").read_text()), sealed)
            self.assertEqual(sealed["plan_sha256"], p.digest(PLAN))
            self.assertEqual(sealed["provenance_sha256"], p.digest({"f": "h"}))
            self.assertEqual(sealed["substitutions_sha256"], p.digest({"3": 1}))
            self.assertEqual(sealed["episodes"], 544)
            self.assertEqual(sealed["image"], "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e")
            self.assertNotEqual(sealed["substitutions_sha256"], p.digest({}))
            self.assertEqual(sealed["fingerprint"], v.fingerprint(root))


if __name__ == "__main__":
    unittest.main()
