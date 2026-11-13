"""Arm A1 of the G1 evaluation: X-VLA trained on G1's randomized data with M2's 17-value state (joints, wrench, finger forces).

Both evaluators (G0's `gen_eval` and M0's `vla_inputs_eval`) load an arm only from a module named `experiments.vla_inputs_arm_*`, by the
identifier `experiments.vla_inputs_arm_xvla_g1.adapter:XVLAG1Adapter`. The state is what the policy saw in training, built live: the
first nine joint positions as P2's rollout reads them, then the wrist wrench and the finger forces from S0's own functions, clipped and
standardised with **G1's** constants (`experiments.xvla_g1_rand.forces`, derived from G1's 2,700 training episodes, never A2's).
`INPUT_DIRS` binds the base bundle and G1's sealed signals, which `forces` reads and verifies against the pinned seal to derive the
constants; `EXTRA_SOURCE` binds that seal's file. The training package is imported read-only and is not edited.
"""
import numpy as np
import torch

from experiments.vla_inputs_arm_xvla.adapter import IMAGE_KEYS, XVLAAdapter
from experiments.vla_inputs_eval.adapters import check_action, check_observation
from experiments.vla_inputs_signals import signals
from experiments.xvla_g1_rand import contract, forces
from experiments.xvla_props_p2_eval import protocol

INPUT_DIRS = (".cache/xvla_trial/bundle", (forces.SEALED / "signals").relative_to(forces.ROOT).as_posix())
EXTRA_SOURCE = ((forces.SEALED / "seal.json").relative_to(forces.ROOT).as_posix(),)


def state_g1(model, data, target):
    """(17,) float64 at the observation instant: `qpos[:9]`, then the standardised wrist wrench (6) and finger forces (2).

    Reads `data` without writing it (S0 computes on a scratch copy). `target` is part of the state-builder signature.
    """
    raw = np.concatenate([signals.wrist_wrench(model, data), signals.finger_force(model, data)])
    return np.concatenate([data.qpos[:9], forces.standardise(raw, forces.constants()).astype(np.float64)])


class XVLAG1Adapter(XVLAAdapter):
    """A0's adapter with the 17-value state: loading, processors and action handling are A0's; `act` validates 17 values."""
    name = "xvla_g1"

    def __init__(self, checkpoint, state_builder=state_g1):
        super().__init__(checkpoint, state_builder)

    def act(self, obs):
        check_observation(obs)
        frame = {"observation.state": torch.tensor(obs["state"], dtype=torch.float32), "task": obs["instruction"]}
        for key, image in zip(IMAGE_KEYS, obs["images"]):
            frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        with protocol.eval_instruction(obs["instruction"]):
            processed = self.pre(contract.observation(frame))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.post(self.policy.select_action(processed)).squeeze(0).float().cpu().numpy()
        check_action(action)
        return action
