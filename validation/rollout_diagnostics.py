"""Explicit physical-command adapters and diagnostic-only task measurements."""
import numpy as np

from data_collection.benchmark import validate_action


def bound_action(action, arm_ranges):
    action = validate_action(action)
    bounded = action.copy()
    bounded[:7] = np.clip(action[:7], arm_ranges[:, 0], arm_ranges[:, 1])
    bounded[7] = np.clip(action[7], 0, 255)
    metrics = {"joint_clip_mask": (bounded[:7] != action[:7]).tolist(),
               "gripper_clipped": bool(bounded[7] != action[7]),
               "gripper_thresholded": bool(bounded[7] != (255 if bounded[7] > 127 else 0)),
               "max_joint_clip_rad": float(np.max(np.abs(bounded[:7] - action[:7])))}
    bounded[7] = 255 if bounded[7] > 127 else 0
    return bounded, metrics


def target_diagnostics(model, data, pick, place, grasp_controller, initial_pick_z):
    pick_pos = data.xpos[model.body(f"target_cube_{pick}").id]
    place_pos = data.xpos[model.body(f"target_cube_{place}").id]
    left = data.xpos[model.body("left_finger").id]
    right = data.xpos[model.body("right_finger").id]
    midpoint = (left + right) / 2
    return {"finger_midpoint_to_pick_m": float(np.linalg.norm(midpoint - pick_pos)),
            "pick_lift_m": float(pick_pos[2] - initial_pick_z),
            "bilateral_target_contact": bool(grasp_controller.is_grasped()),
            "pick_position": pick_pos.copy(), "place_position": place_pos.copy()}