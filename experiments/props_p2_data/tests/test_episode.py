import math
import unittest

import mujoco
import numpy as np

from experiments.props_p2_data import episode as ep
from experiments.props_p2_data import plan as p
from experiments.props_p2_data.validate import admit
from experiments.props_scene import geometry, scene
from experiments.props_scene.expert import task_text
from experiments.props_scene.validate import trace_digest

PLAN = p.make_plan()
MOVE = 4  # phase index of "move" in props_scene.constants.PHASE_NAMES


def _run(episode, render=False):
    frames, traces = [], []

    def sink(frame, trace):
        frames.append(frame)
        traces.append(trace)

    result = ep.run_attempt(episode, p.attempt_draw(episode, 0), sink, render=render)
    return result, frames, traces


def _tall_two_distractor_episode():
    return next(e for e in PLAN["episodes"] if e["distractor_count"] == 2
                and any(d["object"] in ("bottle", "cereal", "milk") for d in e["distractors"]))


class FootprintTests(unittest.TestCase):
    def test_plan_footprints_and_heights_cover_the_compiled_props(self):
        _, model = scene.build_model(p.OBJECTS, "left")
        for name in p.OBJECTS:
            half = geometry.extents(model, name)
            self.assertLessEqual(math.hypot(half[0], half[1]), p.FOOTPRINT_RADIUS_M[name], name)
            self.assertAlmostEqual(2 * half[2], p.HEIGHT_M[name], delta=0.001, msg=name)
        self.assertGreaterEqual(p.CARRY_BOTTOM_M, max(p.HEIGHT_M.values()) + 0.03 - 1e-9)

    def test_instruction_matches_the_expert_wording(self):
        for obj in p.OBJECTS:
            for colour in p.BIN_COLOURS:
                self.assertEqual(p.instruction(obj, colour), task_text(obj, colour))


class EpisodeTests(unittest.TestCase):
    def test_two_distractor_episode_succeeds_carries_high_and_leaves_distractors_alone(self):
        episode = _tall_two_distractor_episode()
        result, frames, traces = _run(episode)
        self.assertTrue(result["success"], result["reason"])
        report = ep.distractor_contacts(result, [f["action"] for f in frames], traces)
        self.assertEqual(report["touched"], [])
        model, data = result["model"], mujoco.MjData(result["model"])
        bottoms = []
        for trace in traces:
            if trace["phase_index"] == MOVE:
                data.qpos[:] = trace["qpos"]
                mujoco.mj_forward(model, data)
                bottoms.append(geometry.bottom_z(model, data, episode["object"]))
        self.assertGreater(max(bottoms), p.CARRY_BOTTOM_M - 0.01, "carry override not active")
        verdict = admit(episode, result, frames, traces, images=False)
        self.assertTrue(verdict["admitted"], verdict)

    def test_a_distractor_that_is_touched_is_reported(self):
        # The real contact path, not a fake: a distractor standing in the target bin touches the bin from tick 0.
        episode = dict(_tall_two_distractor_episode())
        from experiments.props_scene import constants as c
        side = episode["blue_slot"] if episode["bin"] == "blue" else ("right" if episode["blue_slot"] == "left" else "left")
        x, y = c.SLOT_XY[side]
        first = episode["distractors"][0]
        episode["distractors"] = [{**first, "pose": [x, y, 0.0]}, *episode["distractors"][1:]]
        result, frames, traces = _run(episode)
        touched = ep.distractor_contacts(result, [f["action"] for f in frames], traces)["touched"]
        self.assertEqual([t["object"] for t in touched][:1], [first["object"]])
        self.assertTrue(touched[0]["partner"].startswith("bin_"), touched)
        self.assertEqual(admit(episode, result, frames, traces, images=False)["reason"],
                         "expert_success" if not result["success"] else "distractors")

    def test_the_lemon_rocking_alone_is_not_a_touch(self):
        # Measured 2026-10-13: episode 0's lemon rocks to ~22 deg untouched; P1's rule would have rejected it.
        episode = PLAN["episodes"][0]
        self.assertIn("lemon", [d["object"] for d in episode["distractors"]])
        result, frames, traces = _run(episode)
        self.assertEqual(ep.distractor_contacts(result, [f["action"] for f in frames], traces)["touched"], [])

    def test_props_pose_logs_every_object(self):
        episode = _tall_two_distractor_episode()
        result, frames, traces = _run(episode)
        self.assertEqual(traces[0]["props_pose"].shape, (3, 7))

    def test_eviction_and_rebuild_give_the_same_episode(self):
        episode = PLAN["episodes"][1]
        first = _run(episode)
        for e in PLAN["episodes"][2:2 + 2 * ep.MAX_CACHED_MODELS]:
            ep.expert_for(ep.names_of(e), e["blue_slot"])
        self.assertLessEqual(len(ep._EXPERTS), ep.MAX_CACHED_MODELS)
        second = _run(episode)
        digest = lambda run: trace_digest([f["action"] for f in run[1]], run[2])
        self.assertEqual(digest(first), digest(second))

    def test_rendered_frames_have_the_dataset_schema(self):
        episode = PLAN["episodes"][0]
        result, frames, traces = _run(episode, render=True)
        self.assertEqual(len(frames), p.EXPECTED_EPISODE_FRAMES)
        for key in ("observation.images.image", "observation.images.image2"):
            self.assertEqual(frames[5][key].shape, (512, 512, 3))
            self.assertEqual(frames[5][key].dtype, np.uint8)
            self.assertGreater(frames[5][key].std(), 0)
        verdict = admit(episode, result, frames, traces)
        self.assertTrue(verdict["admitted"], {**verdict, "expert_reason": result["reason"]})


if __name__ == "__main__":
    unittest.main()
