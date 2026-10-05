"""Frame extraction, replay checks and the seal fingerprint. Must import without robosuite installed."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from data_collection.benchmark import synchronize_observation
from experiments.paired_demonstrations.media import CAMERAS, FEATURES


def frame_from_sim(model, data, action, task):
    """One LeRobot-schema frame from the simulator's current state."""
    action = np.asarray(action, dtype=np.float64)
    expected = FEATURES["action"]["shape"]
    if action.shape != expected or not np.isfinite(action).all():
        raise ValueError(f"action must be finite with shape {expected}, got {action.shape}")
    synchronize_observation(model, data)
    frame = {"observation.state": np.asarray(data.qpos[:9], dtype=np.float32).copy(),
             "action": action.astype(np.float32),
             "task": task}
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        for key, camera in CAMERAS.items():
            renderer.update_scene(data, camera=camera)
            frame[key] = renderer.render().copy()
    return frame


def _replay(result, frames, traces, use_recorded_ctrl, pick="red", place="green"):
    """Restore the initial state and re-apply controls; mirrors paired_demonstrations replay."""
    from data_collection.benchmark import apply_action, restore_state, stack_metrics, step_control
    from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
    model = result["model"]
    data = mujoco.MjData(model)
    restore_state(model, data, result["initial_state"])
    max_policy, max_full, stable = 0., 0., 0
    for tick, (frame, trace) in enumerate(zip(frames, traces)):
        synchronize_observation(model, data)
        max_policy = max(max_policy, float(np.max(np.abs(data.qpos[:9] - trace["qpos"][:9]))))
        max_full = max(max_full, float(np.max(np.abs(data.qpos - trace["qpos"]))))
        if abs(data.time - trace["sim_time"]) > 1e-9:
            raise ValueError(f"Replay clock drift at tick {tick}")
        if use_recorded_ctrl:
            data.ctrl[:] = trace["ctrl"]
        else:
            apply_action(model, data, frame["action"], ACTIVE_JOINTS)
        if int(trace["substeps"]) == 0:
            continue  # failed Verify Lift: the final tick was recorded but never stepped
        if step_control(model, data, 1 / 30, ARM_DOF) != int(trace["substeps"]):
            raise ValueError(f"Replay substeps changed at tick {tick}")
        metrics = stack_metrics(model, data, pick, place)
        stable = stable + 1 if metrics["released_stack"] else 0
        if use_recorded_ctrl and metrics["released_stack"] != bool(trace["released_stack"]):
            raise ValueError(f"Reference success transcript mismatch at tick {tick}")
    max_policy = max(max_policy, float(np.max(np.abs(data.qpos[:9] - result["final_qpos"][:9]))))
    max_full = max(max_full, float(np.max(np.abs(data.qpos - result["final_qpos"]))))
    return {"max_policy_state_error": max_policy, "max_full_qpos_error": max_full, "stable_ticks": stable}


def replay_recorded(result, frames, traces, pick="red", place="green"):
    """Replay the recorded policy actions through apply_action."""
    return _replay(result, frames, traces, use_recorded_ctrl=False, pick=pick, place=place)


def replay_reference(result, frames, traces, pick="red", place="green"):
    """Replay the exact recorded ctrl vectors (the true-expert reference)."""
    out = _replay(result, frames, traces, use_recorded_ctrl=True, pick=pick, place=place)
    return {"max_full_qpos_error": out["max_full_qpos_error"], "stable_ticks": out["stable_ticks"]}


def kinematic_mismatch(result, traces):
    """Max distance (m) between the robosuite hand body and the URDF panda_hand, every 10th tick."""
    from src.robots.franka_panda.robot import FrankaPanda
    model = result["model"]
    robot = FrankaPanda()
    data = mujoco.MjData(model)
    hand = model.body("robot0_right_hand").id
    worst, worst_tick = 0., 0
    for tick in range(0, len(traces), 10):
        data.qpos[:] = traces[tick]["qpos"]
        mujoco.mj_forward(model, data)
        gap = float(np.linalg.norm(data.xpos[hand] - robot.forward_kinematics(data.qpos[:9].copy()).translation))
        if gap > worst:
            worst, worst_tick = gap, tick
    kinematic_mismatch.worst_tick = worst_tick
    return worst


def _file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(root):
    """sha256 over the sorted {relative path: sha256(file)} JSON of every file except root/seal.json."""
    root, files = Path(root), {}
    if root.is_symlink():
        raise ValueError("Symlink root")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed root: {path.relative_to(root)}")
        if path.is_file() and path != root / "seal.json":
            files[str(path.relative_to(root))] = _file_hash(path)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
