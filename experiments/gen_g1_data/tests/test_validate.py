import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import plan as p
from experiments.gen_g1_data import validate as v
from experiments.vla_inputs_signals.sidecar import KEYS

PLAN = p.make_plan()
MIXED = next(e for e in PLAN["episodes"] if e["prop_distractors"] >= 1 and e["primitive_count"] >= 1)
FRAMES = p.EXPECTED_EPISODE_FRAMES


def _frames(episode, n=FRAMES, images=True):
    frame = {"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32), "task": episode["instruction"]}
    if images:
        frame.update({k: np.zeros((512, 512, 3), np.uint8) for k in v.CAMERA_KEYS})
    return [frame] * n


def _names(episode):
    return [d["object"] for d in episode["distractors"]] + [f"{s}_{c}" for s, c, *_ in p.condition_of(episode).primitives]


def _result(episode, success=True):
    names = _names(episode)
    return {"success": success, "stable_ticks": 30 if success else 0, "frames": FRAMES, "reason": "success" if success else "grasp",
            "distractors": names,
            "distractor_motion": [{"object": n, "displacement_m": 0.0, "tilt_deg": 0.0} for n in names]}


def _arrays(episode, frames=FRAMES, props=3):
    """Arrays that pass S0's `check_arrays`: the right keys, shapes, dtypes and a monotone segment."""
    arrays = {"episode_index": np.array(episode["episode_index"]), "frame_index": np.arange(frames),
              "wrist_wrench": np.zeros((frames, 6)), "finger_force": np.zeros((frames, 2)), "target_box": np.zeros((frames, 2, 5)),
              "props_box": np.zeros((frames, props, 2, 5)), "prop_names": np.array([episode["object"], "x", "y"][:props]),
              "segment": np.repeat(np.arange(4), frames // 4).astype(np.int8), "phase_index": np.zeros(frames, np.int64),
              "object": np.array(episode["object"]), "split": np.array(episode["split"])}
    assert set(arrays) == set(KEYS)
    return arrays


def replay_ok(episode, arrays=None):
    return lambda ep, result, frames, traces: {"ok": True, "arrays": _arrays(episode) if arrays is None else arrays, "ticks": FRAMES}


BAD_REPLAY = lambda ep, result, frames, traces: {"ok": False, "arrays": None}
UNTOUCHED = lambda result, frames, traces: {"touched": []}


def _touched_by(partner):
    return lambda result, frames, traces: {"touched": [{"object": result["distractors"][0], "partner": partner, "tick": 80}]}


def _diverged(result, frames, traces):
    raise ValueError("contact re-run diverged from the recorded trace at tick 3")


def _mismatch(ep, result, frames, traces):
    raise ValueError("qpos differs at tick 3")


class AdmitTests(unittest.TestCase):
    def admit(self, episode=MIXED, images=True, contacts=UNTOUCHED, replay=None, **kwargs):
        frames = kwargs.pop("frames", _frames(episode, images=images))
        result = kwargs.pop("result", None) or _result(episode, **kwargs)
        return v.admit(episode, result, frames, [{}] * len(frames), replay=replay or replay_ok(episode), contacts=contacts,
                       images=images), result

    def test_a_clean_episode_is_admitted_and_carries_its_signals(self):
        verdict, result = self.admit()
        self.assertTrue(verdict["admitted"], verdict)
        self.assertEqual(verdict["reason"], "admitted")
        self.assertEqual(set(result["signals"]), set(KEYS))

    def test_the_episode_is_a_mix_of_props_and_primitives(self):
        self.assertGreaterEqual(len(MIXED["distractors"]), 1)
        self.assertGreaterEqual(len(p.condition_of(MIXED).primitives), 1)

    def test_any_touch_of_a_distractor_rejects_and_is_reported(self):
        for partner in ("left_finger", "robot0_link7", "prop_milk", "bin_blue"):
            verdict, _ = self.admit(contacts=_touched_by(partner))
            self.assertEqual(verdict["reason"], "distractors", partner)
            self.assertEqual(verdict["details"]["distractor_touched"][0]["partner"], partner)

    def test_a_diverging_contact_rerun_rejects(self):
        verdict, _ = self.admit(contacts=_diverged)
        self.assertEqual(verdict["reason"], "distractors")
        self.assertIn("diverged", verdict["details"]["distractor_error"])

    def test_a_missing_prop_or_primitive_rejects(self):
        for drop in (0, -1):
            result = _result(MIXED)
            result["distractors"] = [n for i, n in enumerate(result["distractors"]) if i != drop % len(result["distractors"])]
            verdict, _ = self.admit(result=result)
            self.assertEqual(verdict["reason"], "distractors", drop)

    def test_a_success_needs_the_full_stable_streak(self):
        result = _result(MIXED)
        result["stable_ticks"] = v.STABLE_TICKS - 1
        self.assertEqual(self.admit(result=result)[0]["reason"], "expert_success")
        result["stable_ticks"] = v.STABLE_TICKS
        self.assertTrue(self.admit(result=result)[0]["admitted"])

    def test_the_frame_count_is_the_plans_and_the_experts_and_the_traces(self):
        short = _frames(MIXED, n=FRAMES - 1)
        result = {**_result(MIXED), "frames": FRAMES - 1}
        self.assertEqual(self.admit(frames=short, result=result)[0]["reason"], "frames")
        self.assertEqual(self.admit(result={**_result(MIXED), "frames": FRAMES - 1})[0]["reason"], "frames")
        frames = _frames(MIXED)
        verdict = v.admit(MIXED, _result(MIXED), frames, [{}] * (FRAMES - 1), replay=replay_ok(MIXED), contacts=UNTOUCHED)
        self.assertEqual(verdict["reason"], "frames")

    def test_images_must_be_uint8_512_by_512_rgb(self):
        for key, image in ("observation.images.image", np.zeros((256, 256, 3), np.uint8)), ("observation.images.image2", np.zeros((512, 512, 3), np.float32)):
            frames = [{**f, key: image} for f in _frames(MIXED)]
            self.assertEqual(self.admit(frames=frames)[0]["reason"], "schema", key)

    def test_every_other_gate_rejects_with_its_own_reason(self):
        self.assertEqual(self.admit(success=False)[0]["reason"], "expert_success")
        self.assertEqual(self.admit(replay=BAD_REPLAY)[0]["reason"], "replay")
        self.assertEqual(self.admit(replay=_mismatch)[0]["reason"], "replay")
        self.assertEqual(self.admit(frames=_frames(MIXED, n=FRAMES - 1))[0]["reason"], "frames")
        wrong = [{**f, "task": "put the can in the blue bin"} for f in _frames(MIXED)]
        self.assertNotEqual(MIXED["instruction"], "put the can in the blue bin")
        self.assertEqual(self.admit(frames=wrong)[0]["reason"], "labels")
        bad = [{**f, "observation.state": np.zeros(9, np.float64)} for f in _frames(MIXED)]
        self.assertEqual(self.admit(frames=bad)[0]["reason"], "schema")

    def test_a_replay_that_is_skipped_does_not_run(self):
        calls = []
        verdict, _ = self.admit(success=False, replay=lambda *a: calls.append(a) or {"ok": True, "arrays": _arrays(MIXED)})
        self.assertEqual(calls, [])
        self.assertFalse(verdict["gates"]["replay"])
        self.assertFalse(verdict["gates"]["signals"])

    def test_broken_signals_reject_even_when_the_replay_matched(self):
        for name, break_it in (("nan", lambda a: a["wrist_wrench"].__setitem__((5, 2), np.nan)),
                               ("segments", lambda a: a.__setitem__("segment", a["segment"][::-1].copy())),
                               ("shape", lambda a: a.__setitem__("finger_force", a["finger_force"][:-1])),
                               ("other episode", lambda a: a.__setitem__("episode_index", np.array(MIXED["episode_index"] + 1)))):
            arrays = _arrays(MIXED)
            break_it(arrays)
            verdict, result = self.admit(replay=replay_ok(MIXED, arrays))
            self.assertEqual(verdict["reason"], "signals", name)
            self.assertNotIn("signals", result, name)

    def test_unrendered_gate_runs_skip_only_the_image_check(self):
        self.assertTrue(self.admit(images=False)[0]["admitted"])
        frames = _frames(MIXED, images=False)
        verdict = v.admit(MIXED, _result(MIXED), frames, [{}] * FRAMES, replay=replay_ok(MIXED), contacts=UNTOUCHED)
        self.assertEqual(verdict["reason"], "schema")

    def test_the_first_failing_gate_names_the_reason_in_declared_order(self):
        verdict, _ = self.admit(success=False, contacts=_touched_by("left_finger"))
        self.assertEqual(verdict["reason"], "expert_success")
        self.assertEqual(v.GATES, ("expert_success", "distractors", "frames", "replay", "labels", "schema", "signals"))


def _journal(r_for=None):
    r_for = r_for or {}
    return [{"slot": s, "admitted": True, "r": r_for.get(s, 0)} for s in range(p.ADMIT_COUNT)]


def _meta(episodes):
    return {"total_episodes": len(episodes), "total_frames": FRAMES * len(episodes), "fps": 30,
            "task_table": sorted({e["instruction"] for e in PLAN["episodes"]}),
            "tasks": [[e["instruction"]] for e in episodes], "lengths": [FRAMES] * len(episodes),
            "indices": list(range(len(episodes)))}


def _validate(journal=None, meta=None, plan=PLAN):
    return v.validate_dataset("x", plan, journal or _journal(), reader=lambda root: meta or _meta(plan["episodes"]), files=False)


class DatasetTests(unittest.TestCase):
    def test_a_complete_dataset_validates(self):
        summary = _validate()
        self.assertEqual((summary["episodes"], summary["train"], summary["validation"]), (3000, 2700, 300))
        self.assertEqual(summary["distractor_counts"], {"0": 150, "1": 450, "2": 750, "3": 750, "4": 600, "5": 300})
        self.assertEqual(summary["height_strata"], {str(q): 600 for q in range(5)})
        self.assertEqual(summary["start_regions"], {"outer": 900, "p2": 2100})
        self.assertFalse(summary["signals_checked"])

    def test_out_of_order_journal_or_dataset_is_refused(self):
        journal = _journal()
        journal[3], journal[4] = journal[4], journal[3]
        with self.assertRaises(ValueError):
            v.admitted_episodes(PLAN, journal)
        with self.assertRaises(ValueError):
            v.admitted_episodes(PLAN, _journal()[1:])
        with self.assertRaises(ValueError):
            v.admitted_episodes(PLAN, _journal() + [{"slot": 0, "admitted": True, "r": 0}])
        with self.assertRaises(ValueError):
            _validate(journal)
        meta = _meta(PLAN["episodes"])
        meta["tasks"][0], meta["tasks"][1] = meta["tasks"][1], meta["tasks"][0]
        self.assertNotEqual(meta["tasks"][0], meta["tasks"][1])
        with self.assertRaises(ValueError):
            _validate(meta=meta)
        with self.assertRaises(ValueError):
            _validate(meta={**_meta(PLAN["episodes"]), "indices": [1, 0] + list(range(2, 3000))})

    def test_a_missing_episode_or_wrong_frame_count_is_refused(self):
        with self.assertRaises(ValueError):
            v.validate_dataset("x", PLAN, _journal()[:-1], reader=lambda root: _meta(PLAN["episodes"][:-1]), files=False)
        with self.assertRaises(ValueError):
            _validate(meta={**_meta(PLAN["episodes"]), "lengths": [FRAMES] * 2999 + [FRAMES - 1]})

    def test_a_substitute_scene_is_what_gets_validated_and_counted(self):
        summary = _validate(_journal({5: 1, 6: 2}))
        self.assertEqual(summary["substituted_slots"], 2)
        self.assertEqual(v.admitted_episodes(PLAN, _journal({5: 1}))[5], p.resample_start(PLAN["episodes"][5], 1))

    def test_a_balance_that_differs_from_the_plan_is_refused(self):
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][0]["distractor_count"] = (plan["episodes"][0]["distractor_count"] + 1) % 6
        with self.assertRaises(ValueError):
            _validate(plan=plan)
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][0]["quintile"] = (plan["episodes"][0]["quintile"] + 1) % 5
        with self.assertRaises(ValueError):
            _validate(plan=plan)

    def test_the_validation_split_is_balanced_too(self):
        plan = json.loads(json.dumps(PLAN))
        first = next(e for e in plan["episodes"] if e["split"] == "validation")
        first["distractor_count"] = (first["distractor_count"] + 1) % 6
        with self.assertRaises(ValueError):
            _validate(plan=plan)
        plan = json.loads(json.dumps(PLAN))
        first = next(e for e in plan["episodes"] if e["split"] == "validation")
        first["quintile"] = (first["quintile"] + 1) % 5
        with self.assertRaises(ValueError):
            _validate(plan=plan)

    def test_the_task_table_is_the_trained_instructions(self):
        meta = _meta(PLAN["episodes"])
        with self.assertRaises(ValueError):
            _validate(meta={**meta, "task_table": meta["task_table"] + ["put the milk in the blue bin"]})
        with self.assertRaises(ValueError):
            _validate(meta={**meta, "task_table": meta["task_table"][1:]})

    def test_a_held_out_pair_is_refused(self):
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][0].update(object="milk", bin="blue")
        with self.assertRaises(ValueError):
            _validate(plan=plan)

    def test_a_condition_outside_the_train_pool_is_refused(self):
        plan = json.loads(json.dumps(PLAN))
        bare = plan["episodes"][0]["condition"]
        bare.update(pool="test", table_texture=None, floor_texture=None, wall_texture=None, primitives=[])
        with self.assertRaises(ValueError):  # well formed, and only the pool says it is not a training scene
            _validate(plan=plan)
        bare["pool"] = "train"
        _validate(plan=plan)
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][1]["condition"]["pool"] = "none"
        with self.assertRaises(ValueError):
            _validate(plan=plan)

    def test_a_test_pool_texture_or_a_pool_b_primitive_in_a_train_condition_is_refused(self):
        from experiments.gen_scene import pools
        both = pools.pools()
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][0]["condition"]["table_texture"] = both["table"]["test"][0]
        with self.assertRaises(ValueError):
            _validate(plan=plan)
        plan = json.loads(json.dumps(PLAN))
        plan["episodes"][0]["condition"]["primitives"] = [[sorted(pools.PRIMITIVE_SHAPES["B"])[0], sorted(pools.PRIMITIVE_COLOURS["B"])[0], 0.3, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            _validate(plan=plan)


def _trace(episode, **change):
    draw = {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    fields = dict(qpos=np.zeros((FRAMES, 18)), object=np.asarray(episode["object"]), bin=np.asarray(episode["bin"]),
                  blue_slot=np.asarray(episode["blue_slot"]), start=np.asarray(episode["start"]),
                  distractors=np.asarray([d["object"] for d in episode["distractors"]]),
                  condition_sha256=np.asarray(episode["condition_sha256"]), condition=np.asarray(json.dumps(episode["condition"])),
                  layout_id=np.asarray(episode["layout_id"]), home_pose_jitter_deg=np.asarray(episode["home_pose_jitter_deg"]),
                  draw=np.asarray(json.dumps(draw)))
    fields.update(change)
    return fields


class RecordedFilesTests(unittest.TestCase):
    EPISODES = PLAN["episodes"][:2]

    def build(self, root, trace_change=None, arrays_change=None):
        (root / "episodes").mkdir()
        (root / "signals").mkdir()
        for e in self.EPISODES:
            name = f"episode_{e['episode_index']:04d}.npz"
            np.savez(root / "episodes" / name, **_trace(e, **(trace_change or {}).get(e["episode_index"], {})))
            np.savez(root / "signals" / name, **{**_arrays(e), **(arrays_change or {}).get(e["episode_index"], {})})

    def check(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, **kwargs)
            v.check_recorded_files(root, self.EPISODES)

    def test_files_that_say_what_the_plan_says_pass(self):
        self.check()

    def test_a_trace_of_another_scene_is_refused(self):
        other = PLAN["episodes"][2]
        for key in ("condition_sha256", "layout_id", "start", "home_pose_jitter_deg", "object", "distractors"):
            with self.assertRaises(ValueError, msg=key):
                self.check(trace_change={0: {key: _trace(other)[key]}})
        with self.assertRaises(ValueError):
            self.check(trace_change={0: {"condition": np.asarray(json.dumps(other["condition"]))}})

    def test_signals_of_another_episode_or_length_are_refused(self):
        with self.assertRaises(ValueError):
            self.check(arrays_change={1: {"episode_index": np.array(0)}})
        with self.assertRaises(ValueError):
            self.check(arrays_change={1: {"split": np.array("validation")}})
        with self.assertRaises(ValueError):
            self.check(trace_change={0: {"qpos": np.zeros((FRAMES - 1, 18))}})

    def test_a_missing_or_extra_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root)
            os.remove(root / "signals" / "episode_0001.npz")
            with self.assertRaises(ValueError):
                v.check_recorded_files(root, self.EPISODES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root)
            np.savez(root / "episodes" / "episode_0009.npz", x=np.zeros(1))
            with self.assertRaises(ValueError):
                v.check_recorded_files(root, self.EPISODES)


class RejectionReportTests(unittest.TestCase):
    def test_rates_are_per_stratum_over_all_attempts(self):
        slot = lambda **kw: next(e for e in PLAN["episodes"] if all(e[k] == val for k, val in kw.items()))
        a, b = slot(object="can", quintile=0), slot(object="lemon", quintile=4)
        journal = [{"slot": a["episode_index"], "admitted": False, "reason": "distractors"},
                   {"slot": a["episode_index"], "admitted": True, "reason": "admitted"},
                   {"slot": b["episode_index"], "admitted": True, "reason": "admitted"}]
        report = v.rejection_report(PLAN, journal)
        self.assertEqual((report["attempts"], report["rejected"], report["reasons"]), (3, 1, {"distractors": 1}))
        self.assertEqual(report["object"]["can"], {"attempts": 2, "rejected": 1, "rate": 0.5})
        self.assertEqual(report["object"]["lemon"]["rate"], 0.0)
        self.assertEqual(report["height_stratum"]["0"]["rejected"], 1)
        self.assertEqual(report["distractor_count"][str(a["distractor_count"])]["attempts"] >= 2, True)


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
