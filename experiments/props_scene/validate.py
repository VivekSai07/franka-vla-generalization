"""Replay and trace hashing, and `execute`: one attempt end to end (reset, expert, replay, digest).

Runs in the robosuite image. The sink keeps actions and traces only, never images, so an attempt
holds a few hundred kilobytes however long it is.
"""
import hashlib
import json

import mujoco
import numpy as np

from data_collection.benchmark import apply_action, restore_state, step_control, synchronize_observation
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import in_bin, success_streak
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

REPLAY_LIMIT = 1e-3
_EXPERTS = {}


def _expert(obj, blue_slot):
    from experiments.props_scene.expert import PropsExpert
    key = (obj, blue_slot)
    if key not in _EXPERTS:
        _EXPERTS[key] = PropsExpert((obj,), blue_slot)
    return _EXPERTS[key]


def trace_digest(actions, traces):
    """sha256 over every action and every trace array, in order; byte-equal runs give equal digests."""
    hasher = hashlib.sha256()
    for action, trace in zip(actions, traces):
        hasher.update(np.ascontiguousarray(action, dtype=np.float32).tobytes())
        for key in sorted(trace):
            value = np.asarray(trace[key])
            hasher.update(key.encode())
            hasher.update(np.ascontiguousarray(value if value.dtype.kind in "US" else value.astype(np.float64)).tobytes())
    return hasher.hexdigest()


def _prop_poses(model, data, names):
    return np.array([np.concatenate((data.xpos[model.body(f"prop_{n}").id], data.xquat[model.body(f"prop_{n}").id]))
                     for n in names])


def replay(result, actions, traces):
    """Restore the initial state and re-apply the recorded actions; compare against the recorded traces."""
    model, episode = result["model"], result["episode"]
    names = [n for n in c.OBJECTS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{n}") >= 0]
    data = mujoco.MjData(model)
    restore_state(model, data, result["initial_state"])
    max_state, max_full, max_prop, stable = 0.0, 0.0, 0.0, 0
    transcript_mismatch = None
    for tick, (action, trace) in enumerate(zip(actions, traces)):
        synchronize_observation(model, data)
        max_state = max(max_state, float(np.max(np.abs(data.qpos[:9] - trace["qpos"][:9]))))
        max_full = max(max_full, float(np.max(np.abs(data.qpos - trace["qpos"]))))
        max_prop = max(max_prop, float(np.max(np.abs(_prop_poses(model, data, names) - trace["props_pose"]))))
        if abs(data.time - trace["sim_time"]) > 1e-9:
            raise ValueError(f"replay clock drift at tick {tick}")
        apply_action(model, data, action, ACTIVE_JOINTS)
        if int(trace["substeps"]) == 0:
            continue  # the final tick of a missed grasp was recorded but never stepped
        if step_control(model, data, 1 / c.FPS, ARM_DOF) != int(trace["substeps"]):
            raise ValueError(f"replay substeps changed at tick {tick}")
        tick_metrics = in_bin(model, data, episode["object"], episode["bin"])
        if tick_metrics["success_tick"] != bool(trace["success_tick"]) and transcript_mismatch is None:
            transcript_mismatch = tick
        stable, _ = success_streak(stable, tick_metrics)
    max_state = max(max_state, float(np.max(np.abs(data.qpos[:9] - result["final_qpos"][:9]))))
    max_full = max(max_full, float(np.max(np.abs(data.qpos - result["final_qpos"]))))
    return {"max_policy_state_error": max_state, "max_full_qpos_error": max_full,
            "max_prop_pose_error": max_prop, "stable_ticks": stable, "transcript_mismatch_tick": transcript_mismatch,
            "ok": bool(max(max_state, max_prop) < REPLAY_LIMIT and stable == result["stable_ticks"]
                       and transcript_mismatch is None)}


def execute(attempt, render=True):
    """One attempt: reset, expert, replay. Returns (result, actions, traces, replay_summary, digest, start_error_m)."""
    from experiments.props_scene.scene import object_pose, reset
    obj = attempt["object"]
    expert = _expert(obj, attempt["blue_slot"])
    x, y, yaw = attempt["pose"]
    scene = reset(expert.model, {obj: (x, y, yaw)}, attempt["home_pose_jitter_deg"])
    data = mujoco.MjData(expert.model)
    mujoco.mj_setState(expert.model, data, scene["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(expert.model, data)
    start_error = float(np.hypot(*(object_pose(expert.model, data, obj)["xy"] - np.array([x, y]))))
    actions, traces = [], []

    def sink(frame, trace):
        actions.append(frame["action"].copy())
        traces.append(trace)

    episode = {"object": obj, "bin": attempt["bin"], "seed": attempt["seed"],
               "home_pose_jitter_deg": list(attempt["home_pose_jitter_deg"])}
    result = expert.run(scene["initial_state"], episode, sink, render=render)
    return result, actions, traces, replay(result, actions, traces), trace_digest(actions, traces), start_error


def attempt_digest(attempt):
    """Digest of one attempt, for comparing two processes."""
    return execute(attempt)[4]


if __name__ == "__main__":
    import sys
    print(attempt_digest(json.loads(sys.argv[1])))
