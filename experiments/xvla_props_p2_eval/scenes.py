"""Restore a reserved props P2 layout (target plus distractors) from the sealed plan. Robosuite image (MuJoCo 3.4.0).

The plan stores raw pose parameters per reserved layout (`start`, each distractor's `pose`, home joint jitter);
`props_scene.scene.build_model`/`reset` are deterministic from them, so this reproduces the construction the P2
collector used (`props_p2_data.episode.run_attempt`) and asserts every prop's settled pose lands within 1 mm / 1 deg
of the plan. A model is a compiled robosuite environment; they are large, so the cache is bounded as the collector
does (a rebuild is bit-identical, measured by P2).
"""
import gc

import numpy as np

from experiments.props_scene import scene as props_scene
from experiments.props_scene.scene import build_model, object_pose, present_objects, reset
from experiments.xvla_props_p2_eval import protocol

POSITION_TOLERANCE_M = 1e-3
YAW_TOLERANCE_DEG = 1.0
MAX_CACHED_MODELS = 4
_KEYS = []


def _yaw_difference_deg(a, b):
    return abs(((a - b + 180) % 360) - 180)


def names_of(layout):
    """Target first, then distractors in plan order."""
    return [layout['object'], *(d['object'] for d in layout['distractors'])]


def poses_of(layout):
    return {layout['object']: tuple(layout['start']), **{d['object']: tuple(d['pose']) for d in layout['distractors']}}


def _model(objects, blue_slot):
    key = (tuple(sorted(objects)), blue_slot)
    if key not in _KEYS:
        if len(_KEYS) >= MAX_CACHED_MODELS:
            _KEYS.clear()
            props_scene._built.cache_clear()
            gc.collect()
        _KEYS.append(key)
    return build_model(objects, blue_slot)[1]


def restore(layout_id):
    """(model, data, names) for a reserved layout; raises if any prop drifts from the plan. `names` is the plan's
    target-first order, which is the column order of every recorded prop trace."""
    holdout = protocol._plan()['evaluation_holdout']
    if layout_id not in holdout:
        raise ValueError(f'{layout_id!r} is not a reserved evaluation layout')
    layout = holdout[layout_id]
    names = names_of(layout)
    model = _model(names, layout['blue_slot'])
    if sorted(present_objects(model)) != sorted(names):
        raise ValueError(f'{layout_id}: model holds other props than the plan')
    scene = reset(model, poses_of(layout), layout['home_pose_jitter_deg'])
    import mujoco
    from data_collection.benchmark import restore_state
    data = mujoco.MjData(model)
    restore_state(model, data, scene['initial_state'])
    mujoco.mj_forward(model, data)
    for name, (x, y, yaw) in poses_of(layout).items():
        pose = object_pose(model, data, name)
        if float(np.hypot(pose['xy'][0] - x, pose['xy'][1] - y)) > POSITION_TOLERANCE_M:
            raise ValueError(f'{layout_id}: {name} restored position differs from the plan by more than 1 mm')
        if _yaw_difference_deg(pose['yaw_deg'], yaw) > YAW_TOLERANCE_DEG:
            raise ValueError(f'{layout_id}: {name} restored yaw differs from the plan by more than 1 degree')
    return model, data, names
