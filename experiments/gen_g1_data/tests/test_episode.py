import gc
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np

from experiments.gen_g1_data import episode as ep
from experiments.gen_g1_data import plan as p
from experiments.gen_g1_data.validate import admit
from experiments.gen_scene import compose, gates
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.props_scene import constants as c
from experiments.props_scene.validate import trace_digest
from experiments.vla_inputs_signals import sidecar

PLAN = p.make_plan()
TRACE_KEYS = ("qpos", "ctrl", "sim_time", "props_pose", "phase_index", "substeps", "success_tick")
S0_ROOT = Path(gates.ROOT) / "outputs/vla_inputs_signals/p2_signals_20261030/episodes"


def _run(episode, draw=None, render=False):
    frames, traces = [], []

    def sink(frame, trace):
        frames.append({k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in frame.items()})
        traces.append({k: np.copy(trace[k]) for k in TRACE_KEYS})

    result = ep.run_attempt(episode, draw or p.attempt_draw(episode, 0), sink, render=render)
    return result, frames, traces


def _nominal(index):
    """Sealed P2 episode `index` as a G1-shaped episode under the nominal condition, and its draw."""
    row, _ = gates.sealed_episode(index)
    episode = {**row, "condition": NOMINAL.to_dict(), "condition_sha256": NOMINAL.digest()}
    return episode, {"seed": row["seed"], "home_pose_jitter_deg": list(row["home_pose_jitter_deg"])}


def _actions(frames):
    return [f["action"] for f in frames]


class ParityWithS0Tests(unittest.TestCase):
    """The recorder's signals equal S0's sealed sidecar byte for byte, for the sealed P2 episodes, under the nominal condition."""

    def test_signals_equal_the_sealed_sidecar_byte_for_byte(self):
        for index in gates.GATE_EPISODES:
            episode, draw = _nominal(index)
            result, frames, traces = _run(episode, draw)
            self.assertTrue(result["success"], (index, result["reason"]))
            arrays, details = ep.replay_signals(episode, result, _actions(frames), traces)
            with np.load(S0_ROOT / f"episode_{index:03d}.npz", allow_pickle=False) as stored:
                sealed = {key: stored[key] for key in stored.files}
            self.assertEqual(set(arrays), set(sealed), index)
            for key in sealed:
                self.assertEqual(arrays[key].dtype, sealed[key].dtype, (index, key))
                self.assertTrue(np.array_equal(arrays[key], sealed[key]), (index, key))
            self.assertEqual(details["ticks"], p.EXPECTED_EPISODE_FRAMES)
            sidecar.check_arrays(arrays, index, p.EXPECTED_EPISODE_FRAMES)
            compose.clear_cache()
            gc.collect()

    def test_the_signals_are_not_trivially_zero(self):
        episode, draw = _nominal(gates.GATE_EPISODES[0])
        result, frames, traces = _run(episode, draw)
        arrays, _ = ep.replay_signals(episode, result, _actions(frames), traces)
        self.assertGreater(float(arrays["finger_force"].max()), 1.0)
        self.assertGreater(float(np.abs(arrays["wrist_wrench"]).max()), 1.0)
        self.assertEqual(set(np.unique(arrays["segment"])), {0, 1, 2, 3})


class ReplayExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode, cls.draw = _nominal(gates.GATE_EPISODES[1])
        cls.result, cls.frames, cls.traces = _run(cls.episode, cls.draw)
        cls.actions = _actions(cls.frames)

    def replay(self, actions=None, traces=None, result=None):
        return ep.replay_signals(self.episode, result or self.result, actions or self.actions, traces or self.traces)

    def test_the_untouched_attempt_replays(self):
        self.replay()

    def test_a_different_action_is_a_mismatch(self):
        actions = [a.copy() for a in self.actions]
        actions[40] = actions[40] + np.float32(1e-3)
        with self.assertRaises(ep.ReplayMismatch) as caught:
            self.replay(actions=actions)
        self.assertIn("tick", str(caught.exception))

    def test_a_different_recorded_qpos_prop_pose_or_substep_count_is_a_mismatch(self):
        for key, tick, change in (("qpos", 10, lambda x: x + 1e-9), ("props_pose", 200, lambda x: x + 1e-9),
                                  ("substeps", 7, lambda x: x + 1)):
            traces = [dict(t) for t in self.traces]
            traces[tick][key] = change(traces[tick][key])
            with self.assertRaises(ep.ReplayMismatch, msg=key):
                self.replay(traces=traces)

    def test_a_final_state_or_streak_that_differs_is_a_mismatch(self):
        with self.assertRaises(ep.ReplayMismatch):
            self.replay(result={**self.result, "final_qpos": self.result["final_qpos"] + 1e-9})
        with self.assertRaises(ep.ReplayMismatch):
            self.replay(result={**self.result, "stable_ticks": self.result["stable_ticks"] + 1})

    def test_the_success_tick_flag_is_checked(self):
        traces = [dict(t) for t in self.traces]
        traces[-1]["success_tick"] = not bool(traces[-1]["success_tick"])
        with self.assertRaises(ep.ReplayMismatch):
            self.replay(traces=traces)

    def test_a_tick_that_was_never_stepped_is_a_mismatch(self):
        traces = [dict(t) for t in self.traces]
        traces[-1]["substeps"] = 0
        with self.assertRaises(ep.ReplayMismatch) as caught:
            self.replay(traces=traces)
        self.assertIn("never stepped", str(caught.exception))

    def test_the_signals_are_read_at_the_observation_instant_before_the_action(self):
        order = []
        real_act, real_wrench = ep.apply_action, ep.signals.wrist_wrench

        def act(*args, **kwargs):
            order.append("act")
            return real_act(*args, **kwargs)

        def wrench(*args, **kwargs):
            order.append("signal")
            return real_wrench(*args, **kwargs)

        ep.apply_action, ep.signals.wrist_wrench = act, wrench
        try:
            self.replay()
        finally:
            ep.apply_action, ep.signals.wrist_wrench = real_act, real_wrench
        self.assertEqual(order[:6], ["signal", "act", "signal", "act", "signal", "act"])
        self.assertEqual(order.count("signal"), p.EXPECTED_EPISODE_FRAMES)

    def test_the_arrays_have_the_sidecar_schema(self):
        arrays, _ = self.replay()
        self.assertEqual(set(arrays), set(sidecar.KEYS))
        self.assertEqual(arrays["target_box"].shape, (p.EXPECTED_EPISODE_FRAMES, 2, 5))
        self.assertEqual(str(arrays["object"]), self.episode["object"])
        self.assertEqual(list(arrays["prop_names"]), list(self.result["props"]))


def _tall_height_with_primitive():
    return next(e for e in PLAN["gate_attempts"] if e["primitive_count"] >= 1 and e["quintile"] == 4)


class ContactTests(unittest.TestCase):
    def test_the_real_contact_path_reports_a_distractor_in_a_bin(self):
        episode, draw = _nominal(0)
        episode = dict(episode)
        side = episode["blue_slot"] if episode["bin"] == "blue" else ("right" if episode["blue_slot"] == "left" else "left")
        x, y = c.SLOT_XY[side]
        first = episode["distractors"][0]
        episode["distractors"] = [{**first, "pose": [x, y, 0.0]}, *episode["distractors"][1:]]
        result, frames, traces = _run(episode, draw)
        touched = ep.distractor_contacts(result, _actions(frames), traces)["touched"]
        self.assertEqual([t["object"] for t in touched][:1], [first["object"]])
        self.assertTrue(touched[0]["partner"].startswith("bin_"), touched)

    def test_a_primitive_distractor_in_a_bin_is_reported_by_its_name(self):
        episode, draw = _nominal(0)
        side = episode["blue_slot"] if episode["bin"] == "blue" else ("right" if episode["blue_slot"] == "left" else "left")
        x, y = c.SLOT_XY[side]
        condition = Condition(axis="randomized", level="train", pool="train", primitives=(("cube", "red", x, y, 0.0),)).check()
        episode = {**episode, "condition": condition.to_dict(), "condition_sha256": condition.digest()}
        result, frames, traces = _run(episode, draw)
        self.assertEqual(result["primitives"], ["cube_red"])
        self.assertIn("cube_red", result["distractors"])
        touched = ep.distractor_contacts(result, _actions(frames), traces)["touched"]
        self.assertIn("cube_red", [t["object"] for t in touched])

    def test_the_lemon_rocking_alone_is_not_a_touch(self):
        episode, draw = _nominal(0)
        self.assertIn("lemon", [d["object"] for d in episode["distractors"]])
        result, frames, traces = _run(episode, draw)
        self.assertEqual(ep.distractor_contacts(result, _actions(frames), traces)["touched"], [])

    def test_a_diverging_rerun_raises(self):
        episode, draw = _nominal(1)
        result, frames, traces = _run(episode, draw)
        traces[30] = {**traces[30], "qpos": traces[30]["qpos"] + 1e-9}
        with self.assertRaises(ValueError):
            ep.distractor_contacts(result, _actions(frames), traces)


class RandomizedSceneTests(unittest.TestCase):
    def test_a_randomized_slot_runs_replays_and_is_admitted(self):
        # The first gate attempt (in plan order) whose expert succeeds; every one is admitted or rejected by a named gate.
        verdicts = []
        for attempt in PLAN["gate_attempts"][:6]:
            result, frames, traces = _run(attempt)
            verdict = admit(attempt, result, frames, traces, images=False)
            verdicts.append((attempt["layout_id"], verdict["reason"], result["reason"]))
            compose.clear_cache()
            gc.collect()
        self.assertTrue(any(reason == "admitted" for _, reason, _ in verdicts), verdicts)
        self.assertTrue(all(reason in ("admitted", "expert_success", "distractors") for _, reason, _ in verdicts), verdicts)

    def test_the_scene_is_the_plans_condition(self):
        attempt = _tall_height_with_primitive()
        model, resolved = ep.scene_of(attempt)
        condition = p.condition_of(attempt)
        self.assertEqual(resolved["primitive_names"], [f"{s}_{col}" for s, col, *_ in condition.primitives])
        nominal_top = compose.table_top_z(compose.model_for(attempt, NOMINAL)[0])
        self.assertGreater(condition.table_offset_m, 0.07)
        self.assertAlmostEqual(compose.table_top_z(model) - nominal_top, condition.table_offset_m, places=9)

    def test_the_model_cache_stays_bounded_over_many_scenes(self):
        for attempt in PLAN["gate_attempts"][:6]:
            ep.scene_of(attempt)
            self.assertLessEqual(len(compose._MODELS), compose.MAX_CACHED_MODELS)

    def test_a_rebuilt_model_gives_the_same_attempt(self):
        attempt = _tall_height_with_primitive()
        first = _run(attempt)
        compose.clear_cache()
        gc.collect()
        second = _run(attempt)
        digest = lambda run: trace_digest(_actions(run[1]), run[2])
        self.assertEqual(digest(first), digest(second))
        self.assertEqual(first[0]["reason"], second[0]["reason"])

    def test_the_distractor_motion_names_every_distractor(self):
        attempt = _tall_height_with_primitive()
        result, _, _ = _run(attempt)
        self.assertEqual(sorted(m["object"] for m in result["distractor_motion"]), sorted(result["distractors"]))
        self.assertEqual(len(result["distractors"]), attempt["distractor_count"])


class RenderedFramesTests(unittest.TestCase):
    def test_rendered_frames_have_the_dataset_schema_and_admit(self):
        episode, draw = _nominal(0)
        result, frames, traces = _run(episode, draw, render=True)
        self.assertEqual(len(frames), p.EXPECTED_EPISODE_FRAMES)
        for key in ("observation.images.image", "observation.images.image2"):
            self.assertEqual(frames[5][key].shape, (512, 512, 3))
            self.assertEqual(frames[5][key].dtype, np.uint8)
            self.assertGreater(frames[5][key].std(), 0)
        episode = {**episode, "episode_index": 0, "split": "train"}
        verdict = admit(episode, result, frames, traces)
        self.assertTrue(verdict["admitted"], {**verdict, "expert_reason": result["reason"]})
        self.assertEqual(set(result["signals"]), set(sidecar.KEYS))


if __name__ == "__main__":
    unittest.main()
