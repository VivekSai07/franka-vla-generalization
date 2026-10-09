"""Restore a reserved props P1 layout from the sealed plan. Robosuite image (MuJoCo 3.4.0).

Unlike B3 (a precomputed bank of physics states), the P1 plan stores raw pose parameters (start
x/y/yaw, home joint jitter) for each reserved layout: `props_scene.scene.build_model`/`reset` are
deterministic from those parameters alone, so there is nothing to bank. This restores by
reproducing the same construction the collector used for training episodes (`props_p1_data.episode`)
and asserts the settled pose lands within 1 mm / 1 deg of the plan's recorded start.
"""
import numpy as np

from experiments.props_scene.scene import build_model, object_pose, present_objects, reset
from experiments.xvla_props_p1_eval import protocol

POSITION_TOLERANCE_M = 1e-3
YAW_TOLERANCE_DEG = 1.0


def _yaw_difference_deg(a, b):
    return abs(((a - b + 180) % 360) - 180)


def restore(key):
    """(model, data) for a reserved layout `key` (e.g. 'screen_000'); raises if it drifts from the plan."""
    plan = protocol._plan()
    holdout = plan['evaluation_holdout']
    if key not in holdout:
        raise ValueError(f'{key!r} is not a reserved evaluation layout')
    layout = holdout[key]
    obj, blue_slot = layout['object'], layout['blue_slot']
    x, y, yaw = layout['start']
    _, model = build_model((obj,), blue_slot)
    if present_objects(model) != [obj]:
        raise ValueError(f'{key}: model holds other props than the plan')
    scene = reset(model, {obj: (x, y, yaw)}, layout['home_pose_jitter_deg'])
    import mujoco
    from data_collection.benchmark import restore_state
    data = mujoco.MjData(model)
    restore_state(model, data, scene['initial_state'])
    pose = object_pose(model, data, obj)
    if float(np.hypot(pose['xy'][0]-x, pose['xy'][1]-y)) > POSITION_TOLERANCE_M:
        raise ValueError(f'{key}: restored position differs from the plan by more than 1 mm')
    if _yaw_difference_deg(pose['yaw_deg'], yaw) > YAW_TOLERANCE_DEG:
        raise ValueError(f'{key}: restored yaw differs from the plan by more than 1 degree')
    return model, data
