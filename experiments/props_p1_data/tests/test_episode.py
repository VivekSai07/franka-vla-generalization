import unittest

import numpy as np

from experiments.props_p1_data import episode as ep
from experiments.props_p1_data import plan as p
from experiments.props_scene import constants as c
from experiments.props_scene.expert import task_text


def _run(episode, k=0):
    frames, traces = [], []

    def sink(frame, trace):
        frames.append(frame)
        traces.append(trace)

    draw = p.attempt_draw(episode, k)
    result = ep.run_attempt(episode, draw, sink)
    return result, frames, traces


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = p.make_plan()

    def _find(self, obj, bin_colour, blue_slot):
        return next(e for e in self.plan["episodes"]
                    if e["object"] == obj and e["bin"] == bin_colour and e["blue_slot"] == blue_slot)

    def test_instruction_matches_the_props_scene_expert_wording(self):
        # Closes the cross-check deferred from test_plan.py (admission image cannot import props_scene.expert).
        for obj in p.OBJECTS:
            for colour in p.BIN_COLOURS:
                self.assertEqual(p.instruction(obj, colour), task_text(obj, colour))

    def test_one_episode_per_bin_colour_in_both_arrangements_succeeds(self):
        # Review Focus 1, at episode level: the expert fills the bin the instruction names.
        for colour in p.BIN_COLOURS:
            for slot in ("left", "right"):
                obj = next(e["object"] for e in self.plan["episodes"] if e["bin"] == colour and e["blue_slot"] == slot)
                episode = self._find(obj, colour, slot)
                with self.subTest(object=obj, bin=colour, blue_slot=slot):
                    result, frames, traces = _run(episode)
                    self.assertTrue(result["success"], (obj, colour, slot, result["reason"]))
                    self.assertEqual(frames[0]["task"], episode["instruction"])

    def test_frames_have_the_dataset_schema(self):
        episode = self.plan["episodes"][0]
        result, frames, traces = _run(episode)
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(len(frames), result["frames"])
        frame = frames[10]
        self.assertEqual(frame["observation.state"].shape, (9,))
        self.assertEqual(frame["observation.state"].dtype, np.float32)
        self.assertEqual(frame["action"].shape, (8,))
        self.assertEqual(frame["action"].dtype, np.float32)
        self.assertEqual(frame["task"], episode["instruction"])
        for key in ("observation.images.image", "observation.images.image2"):
            self.assertEqual(frame[key].shape, (c.CAMERA_SIZE, c.CAMERA_SIZE, 3))
            self.assertEqual(frame[key].dtype, np.uint8)
        self.assertEqual(len(frames), p.EXPECTED_EPISODE_FRAMES)
        self.assertIn("qpos", traces[0])
        self.assertIn("reset", result)
        self.assertIn("initial_state", result)
        self.assertIn("model", result)

    def test_a_resampled_start_still_runs(self):
        episode = self.plan["episodes"][0]
        resampled = p.resample_start(episode, 1)
        result, frames, traces = _run(resampled)
        self.assertEqual(frames[0]["task"], resampled["instruction"])


if __name__ == "__main__":
    unittest.main()
