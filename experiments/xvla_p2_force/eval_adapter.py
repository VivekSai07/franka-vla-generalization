"""The rollout side of M2: the 17-value state builder and the X-VLA adapter for the M0 evaluator (arm A2).

Not listed in `contract.SOURCE_FILES`: fixing an adapter after training must not invalidate the training binding. M0's arm
manifest binds it (through `experiments/vla_inputs_arm_xvla_force`, the module M0 loads by identifier, which only re-exports
this file), together with every repository module it loads and the inputs it reads (`INPUT_DIRS`).

`state_force` is what the policy saw in training, built live: the first nine joint positions as P2's rollout reads them, then
the wrist wrench and the finger forces from the very S0 functions that wrote the sidecar, clipped and standardised with the
very constants (`forces`) the training loader applied. `eval_tests/test_force_parity.py` proves the two are byte-equal on
replayed sealed episodes.
"""
import numpy as np
import torch

from experiments.vla_inputs_arm_xvla.adapter import IMAGE_KEYS, XVLAAdapter
from experiments.vla_inputs_eval.adapters import check_action, check_observation
from experiments.vla_inputs_signals import signals
from experiments.xvla_p2_force import contract, forces
from experiments.xvla_props_p2_eval import protocol

# The base bundle (tokenizer files, base config) the processors read, and the sealed sidecar `forces` reads and verifies.
INPUT_DIRS = (".cache/xvla_trial/bundle", forces.SIDECAR.relative_to(forces.ROOT).as_posix())


def state_force(model, data, target):
    """(17,) float64 at the observation instant: `qpos[:9]`, then the standardised wrist wrench (6) and finger forces (2).

    Reads `data` without writing it (S0 computes on a scratch copy). `target` is part of the state-builder signature.
    """
    raw = np.concatenate([signals.wrist_wrench(model, data), signals.finger_force(model, data)])
    return np.concatenate([data.qpos[:9], forces.standardise(raw, forces.constants()).astype(np.float64)])


class XVLAForceAdapter(XVLAAdapter):
    """A0's adapter with the 17-value state: loading, processors and action handling are A0's; `act` validates 17 values."""
    name = "xvla_force"

    def __init__(self, checkpoint, state_builder=state_force):
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
