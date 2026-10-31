"""The rollout side of the X-VLA continuation pilots: the 19-value state builders and the adapter for the M0 evaluator.

Not listed in `contract.SOURCE_FILES`: fixing an adapter after training must not invalidate the training binding. M0's arm
manifest binds it (through `experiments/vla_inputs_arm_xvla_pilot`, the module M0 loads by identifier, which only re-exports
this file), together with every repository module it loads and the inputs it reads (`INPUT_DIRS`).

`LIVE[builder]` is what the policy saw in training for that builder, built live: the first nine joint positions as P2's rollout
reads them, then the builder's extras from the very S0 functions that wrote the sidecar (`force`: M2's own `state_force`,
imported, then two zeros; `box`: S0's `target_box`), then zeros up to 19 values. `eval_tests/test_pilot_parity.py` proves each
equal, byte for byte, to the training row (`contract.joined_state`) on replayed sealed episodes.

**The variant comes from the checkpoint, not from the identifier:** a pilot checkpoint's marker names its variant and the sha256
of its variant file; `XVLAPilotAdapter` loads that file, refuses a changed one, and picks the state builder from it. One adapter
class therefore serves every variant (a new variant is a JSON file, as in training), and a checkpoint cannot be evaluated
with the wrong builder. The evaluator binds the marker (it is one of the checkpoint's hashed files).
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.vla_inputs_arm_xvla.adapter import IMAGE_KEYS, XVLAAdapter
from experiments.vla_inputs_eval.adapters import check_action, check_observation, state_p2
from experiments.vla_inputs_signals import signals
from experiments.xvla_p2_force import eval_adapter as m2_adapter
from experiments.xvla_p2_pilot import contract
from experiments.xvla_props_p2_eval import protocol

# The base bundle (tokenizer files, base config) the processors read, and the sealed sidecar `forces` reads and verifies.
INPUT_DIRS = m2_adapter.INPUT_DIRS
state_force_m2 = m2_adapter.state_force


def _padded(joints, extras):
    state = np.zeros(contract.STATE_WIDTH)
    state[:contract.JOINT_DIM] = joints
    state[contract.JOINT_DIM:contract.JOINT_DIM+len(extras)] = extras
    return state


def state_control(model, data, target):
    """(19,) float64: the nine joint positions, then zeros: P2's input."""
    return _padded(state_p2(model, data, target), ())


def state_force(model, data, target):
    """(19,) float64: M2's 17-value state (joints, standardised wrist wrench and finger forces), then two zeros."""
    state = state_force_m2(model, data, target)
    return _padded(state[:contract.JOINT_DIM], state[contract.JOINT_DIM:])


def state_box(model, data, target):
    """(19,) float64: the nine joint positions, then the target's box in the front and the gripper camera, raw."""
    return _padded(state_p2(model, data, target), signals.target_box(model, data, target).reshape(-1))


LIVE = {'control': state_control, 'force': state_force, 'box': state_box}


def checkpoint_variant(checkpoint):
    """The variant a pilot checkpoint was trained with, from its marker; a changed variant file is refused."""
    path = Path(checkpoint)/'checkpoint_result.json'
    if not path.is_file():
        raise ValueError(f'{checkpoint}: no checkpoint_result.json; not a pilot checkpoint')
    marker = json.loads(path.read_text())
    variant = contract.load_variant(marker.get('variant'))
    if marker.get('variant_sha256') != variant['sha256']:
        raise ValueError(f'{checkpoint}: variant file {variant["path"]} changed since this checkpoint was trained')
    return variant


class XVLAPilotAdapter(XVLAAdapter):
    """A0's adapter with the pilots' 19-value state: loading, processors and action handling are A0's; `act` validates 19
    values. The state builder is the one of the checkpoint's variant (`state_builders` can be injected in tests)."""
    name = "xvla_pilot"

    def __init__(self, checkpoint, state_builders=LIVE):
        variant = checkpoint_variant(checkpoint)
        builders = dict(state_builders)
        builder = variant['spec']['state_builder']
        if builder not in builders:
            raise ValueError(f'No live state builder for {builder!r}')
        super().__init__(checkpoint, builders[builder])
        self.variant = variant['name']

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
