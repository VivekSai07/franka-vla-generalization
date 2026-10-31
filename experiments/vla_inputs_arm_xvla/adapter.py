"""X-VLA policy adapter for the M0 evaluator (the control arm A0, and any arm that only changes the state or the loss).

Loads a checkpoint exactly as `xvla_props_p2_eval.evaluate.load_policy` does, builds the pre/post processors from the
checkpoint directory, and per call does what P2's rollout does inline: `pre(observation(frame))`, `select_action` under
`torch.inference_mode` and bf16 autocast, `post(...)`. Robosuite image with a GPU. Separately bound from the shared core
(`experiments/vla_inputs_eval`): the core loads this module lazily by its identifier and never imports it.
"""
from pathlib import Path

import torch

from experiments.paired_demonstrations.media import CAMERAS
from experiments.vla_inputs_eval.adapters import check_action, check_observation, state_p2
from experiments.xvla_props_p2_eval import protocol
from experiments.xvla_props_p2_eval.evaluate import load_policy
from experiments.xvla_single_task.contract import observation

IMAGE_KEYS = tuple(CAMERAS)


class XVLAAdapter:
    name = "xvla"

    def __init__(self, checkpoint, state_builder=state_p2):
        from lerobot.policies.factory import make_pre_post_processors
        self.checkpoint = Path(checkpoint)
        self.state_builder = state_builder
        self.policy = load_policy(self.checkpoint)
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(self.checkpoint))

    def reset(self):
        self.policy.reset()

    def act(self, obs):
        check_observation(obs)
        frame = {"observation.state": torch.tensor(obs["state"], dtype=torch.float32), "task": obs["instruction"]}
        for key, image in zip(IMAGE_KEYS, obs["images"]):
            frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        with protocol.eval_instruction(obs["instruction"]):
            processed = self.pre(observation(frame))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.post(self.policy.select_action(processed)).squeeze(0).float().cpu().numpy()
        check_action(action)
        return action
