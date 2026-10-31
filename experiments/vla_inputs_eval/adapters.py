"""The frozen policy-adapter interface of the M0 shared core. Pure; imports no model stack.

An adapter wraps one policy for the rollout: `reset()` before a trial, `act(obs) -> (8,)` physical commands. Implementations
live in separately bound arm packages (`experiments/vla_inputs_arm_*`) and are loaded lazily by an explicit identifier
`module:Class`, so an environment without an arm's dependencies never imports it and adding an arm never edits this file.
Do not change this module after A0 is prepared: a change is a new protocol and a new control.

`obs` keys: `images` (two uint8 512 x 512 x 3 arrays, `front_cam` then `gripper_cam`), `state` (float array built by the
arm's state builder), `instruction` (str).
"""
import importlib
import re

import numpy as np

ARM_PREFIX = "experiments.vla_inputs_arm_"
IMAGE_SIZE = 512
ACTION_DIM = 8
_IDENTIFIER = re.compile(r"^(experiments\.vla_inputs_arm_[a-z0-9_]+(?:\.[a-z0-9_]+)*):([A-Za-z_][A-Za-z0-9_]*)$")


def state_p2(model, data, target):
    """The control arm's state: the first nine joint positions (float64 copy), as P2's rollout reads them."""
    return data.qpos[:9].copy()


def check_adapter(adapter):
    if not (isinstance(getattr(adapter, "name", None), str) and callable(getattr(adapter, "reset", None))
            and callable(getattr(adapter, "act", None))):
        raise TypeError(f"{adapter!r} does not implement the adapter interface (name, reset(), act(obs))")


def load_adapter(identifier, *args, **kwargs):
    """Import `module:Class` (module under `experiments.vla_inputs_arm_*`), construct it, check the interface."""
    match = _IDENTIFIER.match(identifier)
    if not match:
        raise ValueError(f"adapter identifier {identifier!r} is not `{ARM_PREFIX}<arm>[.module]:<Class>`")
    adapter = getattr(importlib.import_module(match.group(1)), match.group(2))(*args, **kwargs)
    check_adapter(adapter)
    return adapter


def check_observation(obs):
    missing = {"images", "state", "instruction"} - set(obs)
    if missing:
        raise ValueError(f"observation lacks {sorted(missing)}")
    images = obs["images"]
    if len(images) != 2 or any(not isinstance(i, np.ndarray) or i.dtype != np.uint8
                               or i.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) for i in images):
        raise ValueError("images must be two uint8 512 x 512 x 3 arrays")
    if not isinstance(obs["state"], np.ndarray) or obs["state"].ndim != 1:
        raise ValueError("state must be a one-dimensional numpy array")
    if not isinstance(obs["instruction"], str):
        raise ValueError("instruction must be a string")


def check_action(action):
    if not isinstance(action, np.ndarray) or action.shape != (ACTION_DIM,):
        raise ValueError(f"action must be a numpy array of shape ({ACTION_DIM},)")


def is_finite_action(action):
    return bool(np.isfinite(action).all())
