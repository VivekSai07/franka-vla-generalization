"""Restore an M0 layout (target plus distractors) in the robosuite scene. Robosuite image (MuJoCo 3.4.0).

Forked from `experiments/xvla_props_p2_eval/scenes.py` (P2's is unchanged): the layout is passed in rather than looked up
in the sealed P2 plan. `props_scene.scene.build_model`/`reset` are deterministic from the pose parameters, so this builds the
scene the collector built and asserts every prop's settled pose lands within 1 mm / 1 degree of the plan. A compiled model
is large, so the cache is bounded; a rebuild is bit-identical (measured by P2).
"""
import gc

import mujoco
import numpy as np

from data_collection.benchmark import restore_state
from experiments.props_scene import scene as props_scene
from experiments.props_scene.scene import build_model, object_pose, present_objects, reset

POSITION_TOLERANCE_M = 1e-3
YAW_TOLERANCE_DEG = 1.0
MAX_CACHED_MODELS = 4
_KEYS = []


def _yaw_difference_deg(a, b):
    return abs(((a - b + 180) % 360) - 180)


def names_of(layout):
    """Target first, then distractors in plan order."""
    return [layout["object"], *(d["object"] for d in layout["distractors"])]


def poses_of(layout):
    return {layout["object"]: tuple(layout["start"]), **{d["object"]: tuple(d["pose"]) for d in layout["distractors"]}}


def _model(objects, blue_slot):
    key = (tuple(sorted(objects)), blue_slot)
    if key not in _KEYS:
        if len(_KEYS) >= MAX_CACHED_MODELS:
            _KEYS.clear()
            props_scene._built.cache_clear()
            gc.collect()
        _KEYS.append(key)
    return build_model(objects, blue_slot)[1]


def restore(layout):
    """(model, data, names) for a layout; raises if any prop drifts from the plan. `names` is target first."""
    names = names_of(layout)
    model = _model(names, layout["blue_slot"])
    if sorted(present_objects(model)) != sorted(names):
        raise ValueError(f"{layout.get('layout_id')}: model holds other props than the plan")
    scene = reset(model, poses_of(layout), layout["home_pose_jitter_deg"])
    data = mujoco.MjData(model)
    restore_state(model, data, scene["initial_state"])
    mujoco.mj_forward(model, data)
    for name, (x, y, yaw) in poses_of(layout).items():
        pose = object_pose(model, data, name)
        if float(np.hypot(pose["xy"][0] - x, pose["xy"][1] - y)) > POSITION_TOLERANCE_M:
            raise ValueError(f"{layout.get('layout_id')}: {name} restored position differs from the plan by more than 1 mm")
        if _yaw_difference_deg(pose["yaw_deg"], yaw) > YAW_TOLERANCE_DEG:
            raise ValueError(f"{layout.get('layout_id')}: {name} restored yaw differs from the plan by more than 1 degree")
    return model, data, names
