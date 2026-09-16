"""Plan/physical integrity checks shared by collection and final validation."""
import hashlib

import numpy as np

from data_collection.pilot_plan import validate_plan


def scene_digest(qpos):
    values = np.asarray(qpos, dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid scene geometry")
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_expert_commands(actions, arm_ranges):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise ValueError("Invalid expert action tensor")
    if (np.any(actions[:, :7] < arm_ranges[:, 0] - 1e-6)
            or np.any(actions[:, :7] > arm_ranges[:, 1] + 1e-6)
            or not np.isin(actions[:, 7], [0., 255.]).all()):
        raise ValueError("Expert action outside controller bounds or binary gripper contract")


def validate_accepted_slots(plan, episodes):
    validate_plan(plan)
    if len(episodes) != len(plan["slots"]):
        raise ValueError("Missing or extra accepted episodes")
    hashes = set()
    for index, (slot, entry) in enumerate(zip(plan["slots"], episodes)):
        if entry["episode_index"] != index:
            raise ValueError("Out of order episode")
        for key in ("slot_id", "split", "pick", "place", "colors"):
            if slot[key] != entry.get(key):
                raise ValueError(f"Accepted slot changed {key}")
        if entry.get("seed") not in slot["retry_seeds"]:
            raise ValueError("Accepted episode uses an unplanned seed")
        if not entry.get("scene_hash") or entry["scene_hash"] in hashes:
            raise ValueError("Duplicate or missing initialized scene")
        hashes.add(entry["scene_hash"])


def validate_scene_geometry(model, data, colors):
    positions = np.array([data.xpos[model.body(f"target_cube_{c}").id] for c in colors])
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or not np.isfinite(positions).all()):
        raise ValueError("Nonfinite scene initialization")
    if (np.any(positions[:, 0] < .20 - 1e-5) or np.any(positions[:, 0] > .60 + 1e-5)
            or np.any(positions[:, 1] < -.40 - 1e-5) or np.any(positions[:, 1] > .40 + 1e-5)
            or np.any(positions[:, 2] < 0) or np.any(positions[:, 2] > .10)):
        raise ValueError("Scene outside workspace")
    for i in range(len(positions)):
        for j in range(i):
            if np.linalg.norm(positions[i, :2] - positions[j, :2]) < .15 - 1e-5:
                raise ValueError("Scene violates cube clearance")