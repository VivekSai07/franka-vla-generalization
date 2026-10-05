"""P0 fallback comparison: the same attempts driven by robosuite's OSC_POSE controller instead of our IK.

A separate script, not a replacement for `expert.py`. It builds the same two-bin scene with robosuite's
own physics and controller (`env.step`, torque motors), places the prop at the attempt's start pose, and
walks a scripted hand through the same phases with proportional OSC deltas. The grasp yaw and grasp
point come from `PropsExpert.plan_grasp`, so the comparison isolates the controller. Success uses the same
definition as `metrics.in_bin` (centre of mass inside the target bin's inner box and below the wall top,
gripper open, no finger contact) held for 30 control steps at the end of the episode.

Differences that are not the controller's fault, read them before comparing numbers: robosuite's
actuators, joint dynamics and contact parameters are its own, not the repo's robot contract, and the
gripper is robosuite's. Runs in the robosuite image.

    python -m experiments.props_scene.osc_comparison --out /out/osc_gate --workers 20 [--objects milk cereal]
"""
import argparse
import functools
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.props_scene import constants as c

GAIN = 4.0           # proportional gain on the OSC delta (action 1.0 = 5 cm or 0.5 rad per step)
POS_TOL_M = 0.004
ROT_TOL_RAD = 0.03
MAX_STEPS_PER_LEG = 150
GRIPPER_STEPS = 30
RETRACT_M = c.RETRACT_HEIGHT_M


def _env_class():
    from experiments.props_scene.scene import PropsEnv

    class OscPropsEnv(PropsEnv):
        def reward(self, action=None):
            return 0.0

    return OscPropsEnv


@functools.lru_cache(maxsize=None)
def _env(obj, blue_slot):
    import robosuite
    _env_class()
    return robosuite.make("OscPropsEnv", objects=(obj,), blue_slot=blue_slot, robots="Panda",
                          base_types="RethinkMinimalMount", has_renderer=False, has_offscreen_renderer=False,
                          use_camera_obs=False, control_freq=c.FPS, hard_reset=False,
                          initialization_noise={"magnitude": 0.0, "type": "uniform"})


def _rot_error(current, desired):
    import mujoco
    delta = desired @ current.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, delta.reshape(9))
    axis_angle = np.zeros(3)
    mujoco.mju_quat2Vel(axis_angle, quat, 1.0)
    return axis_angle


def run_attempt(attempt):
    import mujoco
    from robosuite.utils import transform_utils as T
    from src.robots.franka_panda.config import Q_HOME
    from experiments.props_scene import geometry as geo
    from experiments.props_scene.expert import PropsExpert, _rotation_z
    from experiments.props_scene.scene import reset as scene_reset

    obj, bin_colour, slot = attempt["object"], attempt["bin"], attempt["blue_slot"]
    x, y, yaw = attempt["pose"]
    expert = PropsExpert((obj,), slot)
    start = scene_reset(expert.model, {obj: (x, y, yaw)}, [0.0] * 7)
    plan_data = mujoco.MjData(expert.model)
    mujoco.mj_setState(expert.model, plan_data, start["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(expert.model, plan_data)
    plan = expert.plan_grasp(plan_data, obj)
    centre0, half0 = geo.bbox(expert.model, obj)
    height = float(2 * half0[2])
    grasp_pad = plan["hand_point"].copy()
    grasp_pad[2] -= c.HAND_OFFSET_M  # the OSC end effector is the pad centre, not the URDF hand frame
    bin_xy = np.array(c.SLOT_XY["left" if (bin_colour == "blue") == (slot == "left") else "right"])

    env = _env(obj, slot)
    robot = env.robots[0]
    robot.init_qpos = np.asarray(Q_HOME[:7], dtype=float)
    obs = env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    joint = model.joint(f"{obj}_joint0")
    adr = int(joint.qposadr[0])
    data.qpos[adr:adr + 3] = (x, y, half0[2] - centre0[2] + 0.001)
    data.qpos[adr + 3:adr + 7] = (np.cos(np.deg2rad(yaw) / 2), 0.0, 0.0, np.sin(np.deg2rad(yaw) / 2))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    zero = np.zeros(env.action_dim)
    zero[6] = -1.0
    for _ in range(15):
        obs, *_ = env.step(zero)

    prop = model.body(f"{obj}_main").id
    bin_body = model.body(f"bin_{bin_colour}").id
    eef_rot0 = T.quat2mat(obs["robot0_eef_quat_site"])
    target_rot = _rotation_z(plan["hand_yaw"]) @ eef_rot0
    state = {"obs": obs, "steps": 0, "stable": 0, "grasp_lost": False}

    def contact_with_finger():
        for i in range(data.ncon):
            con = data.contact[i]
            names = {model.body(int(model.geom_bodyid[con.geom1])).name, model.body(int(model.geom_bodyid[con.geom2])).name}
            if f"{obj}_main" in names and any("finger" in n for n in names):
                return True
        return False

    def tick(gripper):
        obs, *_ = env.step(np.concatenate((state["act"], [gripper])))
        state["obs"] = obs
        state["steps"] += 1
        delta = data.xpos[prop] - data.xpos[bin_body]
        margin = c.BIN_INNER_M / 2 - max(abs(delta[0]), abs(delta[1]))
        inside = margin > 0 and delta[2] < c.BIN_WALL_HEIGHT_M
        gap = float(abs(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]))
        opened = gap > 0.07
        touching = contact_with_finger()
        ok = bool(inside and opened and not touching)
        state["last"] = {"xy_margin_m": float(margin), "gripper_open": bool(opened), "finger_contact": bool(touching),
                         "finger_gap_m": gap}
        state["stable"] = state["stable"] + 1 if ok else 0

    def leg(target, gripper):
        for _ in range(MAX_STEPS_PER_LEG):
            obs = state["obs"]
            pos = obs["robot0_eef_pos"]
            rot = T.quat2mat(obs["robot0_eef_quat_site"])
            err = target - pos
            rerr = _rot_error(rot, target_rot)
            act = np.zeros(6)
            act[:3] = np.clip(GAIN * err / 0.05, -1, 1)
            act[3:] = np.clip(GAIN * rerr / 0.5, -1, 1)
            state["act"] = act
            tick(gripper)
            if np.linalg.norm(err) < POS_TOL_M and np.linalg.norm(rerr) < ROT_TOL_RAD:
                return True
        return False

    def hold(gripper, steps):
        state["act"] = np.zeros(6)
        for _ in range(steps):
            tick(gripper)

    def object_bottom():
        return float(data.xpos[prop][2] + (centre0[2] - half0[2]))

    started = time.time()
    reason = "success"
    hover = grasp_pad + np.array([0, 0, c.APPROACH_HEIGHT_M])
    reached = leg(hover, -1.0)
    reached &= leg(grasp_pad, -1.0)
    hand_err_at_grasp = float(np.linalg.norm(grasp_pad - state["obs"]["robot0_eef_pos"]))
    hold(1.0, GRIPPER_STEPS)
    pad = state["obs"]["robot0_eef_pos"]
    d = float(pad[2] - object_bottom())
    off = (data.xpos[prop][:2] + (data.xmat[prop].reshape(3, 3) @ centre0)[:2]) - pad[:2]
    carry_z = c.BIN_WALL_HEIGHT_M + c.LIFT_CLEARANCE_M
    leg(np.array([pad[0], pad[1], carry_z + d]), 1.0)
    lifted = object_bottom() - 0.0
    if lifted < 0.5 * carry_z:
        reason = "grasp"
    else:
        pad = state["obs"]["robot0_eef_pos"]
        off = (data.xpos[prop][:2] + (data.xmat[prop].reshape(3, 3) @ centre0)[:2]) - pad[:2]
        xy = bin_xy - off
        leg(np.array([xy[0], xy[1], carry_z + d]), 1.0)
        pad = state["obs"]["robot0_eef_pos"]
        off = (data.xpos[prop][:2] + (data.xmat[prop].reshape(3, 3) @ centre0)[:2]) - pad[:2]
        xy = bin_xy - off
        leg(np.array([xy[0], xy[1], c.BIN_WALL_HEIGHT_M + c.RELEASE_ABOVE_WALL_M + d]), 1.0)
        hold(-1.0, GRIPPER_STEPS)
        pad = state["obs"]["robot0_eef_pos"]
        leg(pad + np.array([0, 0, RETRACT_M]), -1.0)
        hold(-1.0, c.SUCCESS_TICKS)
    success = state["stable"] >= c.SUCCESS_TICKS
    if not success and reason == "success":
        reason = "not_in_bin"
    return {"id": attempt["id"], "object": obj, "pose": attempt["pose"], "target_slot": attempt["target_slot"],
            "success": bool(success), "reason": reason, "steps": state["steps"], "reached_grasp": bool(reached),
            "hand_error_at_grasp_m": hand_err_at_grasp, "seconds": round(time.time() - started, 1),
            "centre_z_m": float(data.xpos[prop][2] - data.xpos[bin_body][2]), "height_m": height,
            "last_tick": state["last"]}


def _work_chunk(attempts):
    rows = []
    for attempt in attempts:
        try:
            rows.append(run_attempt(attempt))
        except Exception as error:  # an infrastructure fault is a failed attempt, never a skipped one
            rows.append({"id": attempt["id"], "object": attempt["object"], "pose": attempt["pose"],
                         "target_slot": attempt["target_slot"], "success": False,
                         "reason": f"exception:{type(error).__name__}:{error}"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--objects", nargs="*", default=list(c.OBJECTS))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from experiments.props_scene.benchmark import build_plan
    attempts = [a for a in build_plan() if a["object"] in args.objects]
    attempts.sort(key=lambda a: (a["object"], a["blue_slot"], a["index"]))
    if args.limit:
        attempts = attempts[:args.limit]
    size = max(1, -(-len(attempts) // args.workers))
    chunks = [attempts[i:i + size] for i in range(0, len(attempts), size)]
    with get_context("spawn").Pool(min(args.workers, len(chunks)), maxtasksperchild=1) as pool:
        rows = [row for part in pool.map(_work_chunk, chunks, chunksize=1) for row in part]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "osc_results.json").write_text(json.dumps(rows, indent=1))
    by_object = {o: (sum(r["success"] for r in rows if r["object"] == o), sum(1 for r in rows if r["object"] == o))
                 for o in sorted({r["object"] for r in rows})}
    print("OSC comparison:", sum(r["success"] for r in rows), "/", len(rows), by_object)


if __name__ == "__main__":
    main()
