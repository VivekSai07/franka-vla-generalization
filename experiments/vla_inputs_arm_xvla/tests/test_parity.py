"""GPU parity of the X-VLA adapter with P2's own inline code path. Robosuite image, GPU, the P2 bundle mounted at /bundle."""
import json
import os
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    raise unittest.SkipTest("needs a GPU")

import mujoco
from lerobot.policies.factory import make_pre_post_processors

from experiments.paired_demonstrations.media import CAMERAS
from experiments.vla_inputs_arm_xvla.adapter import XVLAAdapter
from experiments.vla_inputs_eval import adapters, scenes
from experiments.xvla_props_p2_eval import protocol
from experiments.xvla_props_p2_eval.evaluate import load_policy
from experiments.xvla_single_task.contract import observation

BUNDLE = Path(os.environ.get("BUNDLE", "/bundle"))
CHECKPOINT = BUNDLE / "checkpoints/step_101520"
IMAGE_KEYS = tuple(CAMERAS)


def fixed_observation(layout_key="report_000"):
    layout = json.loads((BUNDLE / "dataset_provenance/plan.json").read_text())["evaluation_holdout"][layout_key]
    model, data, names = scenes.restore(layout)
    mujoco.mj_forward(model, data)
    images = []
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        for camera in CAMERAS.values():
            renderer.update_scene(data, camera=camera)
            images.append(renderer.render().copy())
    return dict(images=images, state=adapters.state_p2(model, data, names[0]), instruction=layout["instruction"])


class P2Path:
    """What `xvla_props_p2_eval.rollout.rollout` does per tick, inline, on its own policy instance."""

    def __init__(self):
        self.policy = load_policy(CHECKPOINT)
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(CHECKPOINT))

    def act(self, obs):
        frame = {"observation.state": torch.tensor(obs["state"], dtype=torch.float32), "task": obs["instruction"]}
        for key, image in zip(IMAGE_KEYS, obs["images"]):
            frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        with protocol.eval_instruction(obs["instruction"]):
            processed = self.pre(observation(frame))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.post(self.policy.select_action(processed)).squeeze(0).float().cpu().numpy()


class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.use_deterministic_algorithms(True)
        cls.obs = fixed_observation()
        cls.adapter = XVLAAdapter(CHECKPOINT)
        cls.reference = P2Path()

    def run_both(self, steps):
        self.adapter.reset()
        self.reference.policy.reset()
        torch.manual_seed(1234)
        mine = [self.adapter.act(self.obs) for _ in range(steps)]
        torch.manual_seed(1234)
        theirs = [self.reference.act(self.obs) for _ in range(steps)]
        return mine, theirs

    def test_the_adapter_action_equals_p2s_code_path_byte_for_byte(self):
        mine, theirs = self.run_both(12)  # 12 > 10 queued actions: the second chunk is inferred too
        for step, (a, b) in enumerate(zip(mine, theirs)):
            self.assertEqual(a.tobytes(), b.tobytes(), step)

    def test_act_returns_eight_finite_floats(self):
        self.adapter.reset()
        action = self.adapter.act(self.obs)
        self.assertEqual(action.shape, (8,))
        self.assertTrue(np.isfinite(action).all())

    def test_the_adapter_is_loadable_by_identifier(self):
        adapter = adapters.load_adapter("experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter", CHECKPOINT)
        self.assertEqual(adapter.name, "xvla")

    def test_a_different_observation_gives_a_different_action(self):
        other = fixed_observation("report_059")
        self.adapter.reset()
        a = self.adapter.act(self.obs)
        self.adapter.reset()
        b = self.adapter.act(other)
        self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_the_image_order_matters(self):
        swapped = dict(self.obs, images=self.obs["images"][::-1])
        self.adapter.reset()
        a = self.adapter.act(self.obs)
        self.adapter.reset()
        b = self.adapter.act(swapped)
        self.assertNotEqual(a.tobytes(), b.tobytes())


if __name__ == "__main__":
    unittest.main()
