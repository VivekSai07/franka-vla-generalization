"""Restore a G0 trial's scene (layout under a condition). Robosuite image (MuJoCo 3.4.0).

Forked from `experiments/vla_inputs_eval/scenes.py` (frozen, unchanged): the scene comes from the randomization engine
(`gen_scene.compose`), which is P2's scene under the nominal condition (gate D9 i), and the layout is passed with its condition.
`compose.reset` is deterministic from the pose parameters, so this builds the scene the feasibility pass ran and asserts every
object's settled pose lands within 1 mm / 1 degree of the plan. A compiled model is large, so the engine's cache is bounded.
"""
import mujoco
import numpy as np

from data_collection.benchmark import restore_state
from experiments.gen_scene import compose
from experiments.props_scene.scene import object_pose

POSITION_TOLERANCE_M = 1e-3
SETTLE_TOLERANCE_M = {"lemon": 5e-3}   # a lemon rocks after reset (P2 measured up to 4.6 mm untouched); measured 1.7 mm on a raised table
YAW_TOLERANCE_DEG = 1.0


def _yaw_difference_deg(a, b):
    return abs(((a - b + 180) % 360) - 180)


def restore(layout, condition):
    """(model, data, resolved) for a layout under a condition; raises if any object drifts from the plan.
    `resolved["names"]` lists the target first, then the layout's distractors, the condition's extra props and its primitives."""
    model, resolved = compose.model_for(layout, condition)
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    data = mujoco.MjData(model)
    restore_state(model, data, scene["initial_state"])
    mujoco.mj_forward(model, data)
    for name, (x, y, yaw) in resolved["poses"].items():
        pose = object_pose(model, data, name)
        if float(np.hypot(pose["xy"][0] - x, pose["xy"][1] - y)) > SETTLE_TOLERANCE_M.get(name, POSITION_TOLERANCE_M):
            raise ValueError(f"{layout.get('layout_id')}: {name} restored position differs from the plan by more than {SETTLE_TOLERANCE_M.get(name, POSITION_TOLERANCE_M) * 1000:g} mm")
        if _yaw_difference_deg(pose["yaw_deg"], yaw) > YAW_TOLERANCE_DEG:
            raise ValueError(f"{layout.get('layout_id')}: {name} restored yaw differs from the plan by more than 1 degree")
    return model, data, resolved
