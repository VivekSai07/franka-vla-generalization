import unittest

import mujoco
import numpy as np

from experiments.props_p3_data import episode as ep
from experiments.props_p3_data import plan as p
from experiments.props_p3_data.validate import admit
from experiments.props_scene.expert import task_text
from experiments.props_scene.validate import trace_digest

PLAN = p.make_plan()
EPISODE = PLAN["episodes"][0]


def _run(episode, render=False):
    frames, traces = [], []

    def sink(frame, trace):
        frames.append(frame)
        traces.append(trace)

    result = ep.run_attempt(episode, p.attempt_draw(episode, 0), sink, render=render)
    return result, frames, traces


def _episode(obj, slot="left"):
    return next(e for e in PLAN["episodes"] if e["object"] == obj and e["blue_slot"] == slot)


class FixedPromptTests(unittest.TestCase):
    def test_the_expert_itself_writes_its_own_wording(self):
        # The premise of the sink fix: without it, the frozen expert puts "put the X in the Y bin" in every frame.
        episode = EPISODE
        frames = []
        expert = ep.expert_for(episode["object"], episode["blue_slot"])
        from experiments.props_scene.scene import reset
        scene = reset(expert.model, {episode["object"]: tuple(episode["start"])}, episode["home_pose_jitter_deg"])
        expert.run(scene["initial_state"], {"object": episode["object"], "bin": episode["bin"], "seed": episode["seed"],
                                            "home_pose_jitter_deg": episode["home_pose_jitter_deg"]},
                   lambda frame, trace: frames.append(frame["task"]), render=False)
        self.assertEqual(set(frames), {task_text(episode["object"], episode["bin"])})
        self.assertNotEqual(frames[0], p.PROMPT)

    def test_every_recorded_frame_carries_the_fixed_prompt_and_nothing_else(self):
        # Review Focus 3.
        result, frames, traces = _run(EPISODE)
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(len(frames), p.EXPECTED_EPISODE_FRAMES)
        self.assertEqual({f["task"] for f in frames}, {p.PROMPT})
        self.assertFalse(any("put the" in f["task"] for f in frames))

    def test_the_wrapper_accepts_the_expert_wording_for_every_object_and_raises_on_anything_else(self):
        for obj in p.OBJECTS:
            episode = {"object": obj, "bin": p.container_of(obj)}
            seen = []
            wrapped = ep.relabelled(lambda frame, trace: seen.append(frame["task"]), episode)
            wrapped({"task": f"put the {obj} in the {p.container_of(obj)} bin"}, {})
            self.assertEqual(seen, [p.PROMPT])
            for other in (p.PROMPT, f"put the {obj} in the {'blue' if p.container_of(obj) == 'orange' else 'orange'} bin", ""):
                with self.assertRaises(ValueError):
                    wrapped({"task": other}, {})
        self.assertEqual(seen, [p.PROMPT])


class EpisodeTests(unittest.TestCase):
    def test_a_trained_episode_succeeds_and_is_admitted_through_the_real_replay(self):
        result, frames, traces = _run(EPISODE)
        verdict = admit(EPISODE, result, frames, traces, images=False)
        self.assertTrue(verdict["admitted"], {**verdict, "expert_reason": result["reason"]})
        self.assertEqual(verdict["details"]["replay"]["max_prop_pose_error"], 0.0)

    def test_every_object_in_both_arrangements_is_admitted(self):
        # The held-out objects are never collected, but the expert must handle them for the reserved layouts.
        for attempt in (_episode(o, s) if o in p.TRAINED_OBJECTS else next(
                g for g in PLAN["gate_attempts"] if g["object"] == o and g["blue_slot"] == s)
                for o in p.OBJECTS for s in p.SLOTS):
            result, frames, traces = _run(attempt)
            verdict = admit(attempt, result, frames, traces, images=False)
            self.assertTrue(verdict["admitted"], (attempt["object"], attempt["blue_slot"], result["reason"], verdict["reason"]))

    def test_props_pose_logs_the_one_object(self):
        _, _, traces = _run(EPISODE)
        self.assertEqual(traces[0]["props_pose"].shape, (1, 7))

    def test_the_planned_start_is_where_the_object_begins(self):
        episode = _episode("bread", "right")
        _, _, traces = _run(episode)
        np.testing.assert_allclose(traces[0]["props_pose"][0][:2], episode["start"][:2], atol=0.005)
        yaw = np.degrees(2 * np.arctan2(traces[0]["props_pose"][0][6], traces[0]["props_pose"][0][3]))
        self.assertAlmostEqual(yaw, episode["start"][2], delta=2.0)

    def test_the_draws_home_jitter_moves_the_arm_by_the_jitter(self):
        # The settle under gravity shifts the arm by a few mrad, so compare two draws: the sag cancels in the difference.
        a, b = p.attempt_draw(EPISODE, 0), p.attempt_draw(EPISODE, 1)
        jitter_difference = np.deg2rad(np.asarray(a["home_pose_jitter_deg"]) - np.asarray(b["home_pose_jitter_deg"]))
        self.assertGreater(np.abs(jitter_difference).max(), 0.01)

        def first_qpos(draw):
            traces = []
            ep.run_attempt(EPISODE, draw, lambda f, t: traces.append(t), render=False)
            return traces[0]["qpos"][:7]

        np.testing.assert_allclose(first_qpos(a) - first_qpos(b), jitter_difference, atol=0.004)

    def test_the_draws_seed_drives_the_expert(self):
        digest = lambda run: trace_digest([f["action"] for f in run[1]], run[2])
        draw = p.attempt_draw(EPISODE, 0)
        frames, traces = [], []
        ep.run_attempt(EPISODE, {**draw, "seed": draw["seed"] + 1}, lambda f, t: (frames.append(f), traces.append(t)),
                       render=False)
        self.assertNotEqual(trace_digest([f["action"] for f in frames], traces), digest(_run(EPISODE)))

    def test_the_expert_cache_is_bounded_and_a_rebuild_gives_the_same_episode(self):
        ep._EXPERTS.clear()
        ep.props_scene._built.cache_clear()
        first = _run(EPISODE)
        fillers = [(o, s) for o in p.OBJECTS for s in p.SLOTS if (o, s) != (EPISODE["object"], EPISODE["blue_slot"])]
        for obj, slot in fillers[:ep.MAX_CACHED_MODELS + 1]:
            ep.expert_for(obj, slot)
        self.assertLessEqual(len(ep._EXPERTS), ep.MAX_CACHED_MODELS)
        self.assertLessEqual(ep.props_scene._built.cache_info().currsize, ep.MAX_CACHED_MODELS)
        self.assertNotIn((EPISODE["object"], EPISODE["blue_slot"]), ep._EXPERTS)
        second = _run(EPISODE)
        digest = lambda run: trace_digest([f["action"] for f in run[1]], run[2])
        self.assertEqual(digest(first), digest(second))

    def test_the_expert_for_a_key_is_cached(self):
        self.assertIs(ep.expert_for("can", "left"), ep.expert_for("can", "left"))

    def test_the_blue_bin_is_on_the_planned_side(self):
        for slot, sign in ("left", 1), ("right", -1):
            model = ep.expert_for("can", slot).model
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertGreater(sign * data.xpos[model.body("bin_blue").id][1], 0.3, slot)
            self.assertLess(sign * data.xpos[model.body("bin_orange").id][1], -0.3, slot)

    def test_rendered_frames_have_the_dataset_schema(self):
        result, frames, traces = _run(EPISODE, render=True)
        self.assertEqual(len(frames), p.EXPECTED_EPISODE_FRAMES)
        for key in ("observation.images.image", "observation.images.image2"):
            self.assertEqual(frames[5][key].shape, (512, 512, 3))
            self.assertEqual(frames[5][key].dtype, np.uint8)
            self.assertGreater(frames[5][key].std(), 0)
        verdict = admit(EPISODE, result, frames, traces)
        self.assertTrue(verdict["admitted"], {**verdict, "expert_reason": result["reason"]})


if __name__ == "__main__":
    unittest.main()
