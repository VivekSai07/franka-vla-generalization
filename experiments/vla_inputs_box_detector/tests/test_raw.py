"""The raw-render path of the gate (robosuite image, CPU, OSMesa): replay, render at the cached frames, predict, score; and how far a raw render
is from the video-decoded frame the detector trained on (the codec shift the gate's measurement (b) exists to expose)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import data, gate, inputs

EPISODE = 480


def mounted():
    try:
        import mujoco  # noqa: F401  (this module runs in the robosuite image)
        from experiments.xvla_p2_force_box import boxes, contract
        return (contract.DATA / "meta/info.json").is_file() and (boxes.forces.SIDECAR / "seal.json").is_file()
    except Exception:
        return False


@unittest.skipUnless(mounted(), "the sealed data is not mounted")
class RawPath(unittest.TestCase):
    def test_a_predictor_that_returns_s0s_box_scores_one_on_every_replayed_frame_and_is_asked_in_tick_order(self):
        from experiments.xvla_p2_force_box import boxes, contract
        table, asked = boxes.table(), []

        class Oracle:
            def predict(self, images, instruction):
                self.last = len(asked)
                asked.append((instruction, [i.shape for i in images], [i.dtype for i in images]))
                return table[EPISODE, k.CACHE_FRAMES[self.last]]

        records = gate.raw_report(Oracle(), [EPISODE], log=lambda *a, **kw: None)
        self.assertEqual(len(asked), len(k.CACHE_FRAMES))
        self.assertEqual({a[0] for a in asked}, {contract.planned_instructions()[EPISODE]})
        self.assertEqual(asked[0][1], [(512, 512, 3), (512, 512, 3)])
        self.assertEqual(asked[0][2], [np.uint8, np.uint8])
        self.assertEqual(records[0]["episode"], EPISODE)
        self.assertEqual(records[0]["front"], [1.0] * 57)
        self.assertEqual(records[0]["gripper"], [1.0] * 57)
        self.assertEqual(gate.merge_raw(records)["front"]["mean_iou"], 1.0)

    def test_the_raw_render_is_close_to_the_decoded_frame_of_the_same_tick(self):
        """Alignment and codec shift: the replayed render at tick t against the cached decoded frame t; a frame off would differ far more."""
        import mujoco
        from experiments.vla_inputs_eval.rollout import CAMERA_NAMES
        from experiments.vla_inputs_signals import replay
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            data.extract(cache_dir, episodes=(EPISODE,), workers=1, free_bytes=lambda p: 10**15)
            cache = data.open_cache(cache_dir)
            renders, holder = {}, {}

            def on_frame(tick, model, sim):
                if tick in (0, 120, 240, 448):
                    if "r" not in holder:
                        holder["r"] = mujoco.Renderer(model, height=512, width=512)
                    images = []
                    for camera in CAMERA_NAMES:
                        holder["r"].update_scene(sim, camera=camera)
                        images.append(holder["r"].render().copy())
                    renders[tick] = inputs.preprocess(images)

            replay.frames(EPISODE, on_frame)
            holder["r"].close()
            errors = {}
            for tick, (front, gripper) in renders.items():
                row = k.CACHE_FRAMES.index(tick)
                errors[tick] = (float(np.abs(front.astype(float) - cache.front[row]).mean()), float(np.abs(gripper.astype(float) - cache.gripper[row]).mean()))
                print("CODEC_SHIFT tick", tick, "mean abs grey-level difference front/gripper", errors[tick], flush=True)
            for tick, (front_error, gripper_error) in errors.items():
                self.assertLess(front_error, 6.0, tick)
                self.assertLess(gripper_error, 6.0, tick)


if __name__ == "__main__":
    unittest.main()
