"""One policy rollout on a G0 trial (a layout under one condition). Robosuite image (MuJoCo 3.4.0); a GPU only for a real policy.

Forked from `experiments/vla_inputs_eval/rollout.py` (frozen, unchanged). Differences, and only these: the scene arrives already
composed under the trial's condition; the tick cap is the trial's (900, or 1,200 for a disturbance); `drop` and `move` act at the
start of a tick (`gen_eval.disturbance` decides when, here they are applied); the bin the target rests in is measured against the
bin's own height (`gen_scene.metrics.placement`; `props_scene`'s is absolute and wrong off table height 0); and the episode record
adds `grasp_lift`, `max_lift_m`, `disturbance_fired`, `disturbance_tick` and `recovered`. The trace holds exactly M0's arrays:
under the nominal condition a trial is M0's, bit for bit (gate D9 iii). The success rule, lift rule and failure classifier are
P2's, imported unchanged from `analysis`.
"""
import json
import random
import time

import av
import mujoco
import numpy as np
import torch

from data_collection.benchmark import apply_action, step_control, synchronize_observation
from experiments.gen_eval.disturbance import FORCED_OPEN_COMMAND, Disturbance
from experiments.gen_scene import levels
from experiments.gen_scene.conditions import Condition
from experiments.gen_scene.metrics import placement
from experiments.paired_demonstrations.media import CAMERAS
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import in_bin, success_streak
from experiments.vla_inputs_eval import adapters
from experiments.vla_inputs_signals import signals
from experiments.xvla_props_p2_eval import analysis
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from validation.rollout_diagnostics import bound_action

CAMERA_NAMES = tuple(CAMERAS.values())
TRACE_KEYS = ("state", "raw_action", "applied_action", "props_xyz", "xy_margin_m", "z_m", "lowest_z_m",
              "gripper_open", "finger_contact")
SIGNAL_KEYS = ("wrist_wrench", "finger_force", "target_box")
FINGERS = ("left_finger", "right_finger")


def seed_trial(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def placed_bin_of(model, data, obj):
    """Whichever bin the prop rests stably inside at the moment this is called, or None."""
    for colour in c.BINS:
        if placement(model, data, obj, colour)["resting_in_bin"]:
            return colour
    return None


def grasp_xy(model, data):
    """Horizontal position of the grasp centre: the midpoint of the two fingers."""
    left, right = (data.xpos[model.body(name).id] for name in FINGERS)
    return ((left + right) / 2)[:2]


def teleport(model, data, name, xy):
    """Put the prop at `xy` at its current height and orientation, at rest."""
    body = model.body(f"prop_{name}")
    joint = model.joint(int(body.jntadr[0]))
    data.qpos[joint.qposadr[0]:joint.qposadr[0] + 2] = xy
    data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0
    mujoco.mj_forward(model, data)


def lift_of(props_z):
    """Greatest rise of the target (column 0) above where it started, in metres."""
    z = np.asarray(props_z, dtype=float)
    return float(z[:, 0].max() - z[0, 0]) if len(z) else 0.0


def run(adapter, state_builder, model, data, resolved, renderer, trial, folder, condition=None):
    """Run one trial from the restored `data`; write cameras.mp4, trace.npz, episode.json in `folder`."""
    condition = condition if condition is not None else Condition.from_dict(trial["condition"])
    names = resolved["names"]
    obj, bin_colour = trial["object"], trial["bin"]
    if names[0] != obj:
        raise ValueError("names must list the target first")
    max_ticks = trial["max_ticks"]
    adapter.reset()
    seed_trial(trial["seed"])
    synchronize_observation(model, data)
    prop_ids = [model.body(f"prop_{name}").id for name in names]
    disturbance = Disturbance(condition)
    start_z = None
    trace = {key: [] for key in TRACE_KEYS + SIGNAL_KEYS}
    stable, success, outcome, started = 0, False, "timeout", time.monotonic()
    metrics = {}
    with av.open(str(folder / "cameras.mp4"), "w") as video:
        stream = video.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}

        def observe():
            images = []
            for camera in CAMERA_NAMES:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")):
                video.mux(packet)
            return images

        for tick in range(max_ticks):
            synchronize_observation(model, data)
            if disturbance.kind != "none":
                target = data.xpos[prop_ids[0]]
                if start_z is None:
                    start_z = float(target[2])
                if disturbance.move_due(tick, grasp_xy(model, data), target[:2]):
                    teleport(model, data, obj, disturbance.move_to)
                    synchronize_observation(model, data)
                    disturbance.fire(tick)
                elif disturbance.drop_due(tick, float(target[2]), start_z):
                    disturbance.fire(tick)
            signal_rows = (signals.wrist_wrench(model, data), signals.finger_force(model, data),
                           signals.target_box(model, data, obj))
            state = state_builder(model, data, obj)
            action = adapter.act(dict(images=observe(), state=state, instruction=trial["instruction"]))
            adapters.check_action(action)
            trace["state"].append(state)
            trace["raw_action"].append(action.copy())
            trace["props_xyz"].append(np.stack([data.xpos[i] for i in prop_ids]))
            for key, row in zip(SIGNAL_KEYS, signal_rows):
                trace[key].append(row)
            if not adapters.is_finite_action(action):
                outcome = "nonfinite_action"
                break
            bounded, _ = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
            if disturbance.forced_open(tick):
                bounded = np.array(bounded, copy=True)
                bounded[7] = FORCED_OPEN_COMMAND
            apply_action(model, data, bounded, ACTIVE_JOINTS)
            disturbance.note_applied_gripper(bounded[7])
            if step_control(model, data, 1 / 30, ARM_DOF) != 20:
                raise ValueError("Physics substep count changed")
            metrics = in_bin(model, data, obj, bin_colour)
            trace["applied_action"].append(bounded.copy())
            for key in ("xy_margin_m", "z_m", "lowest_z_m", "gripper_open", "finger_contact"):
                trace[key].append(metrics[key])
            stable, success = success_streak(stable, metrics)
            if success:
                outcome = "success"
                break
        observe()
        for packet in stream.encode():
            video.mux(packet)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays["applied_action"] = arrays["applied_action"].reshape(-1, 8)
    arrays["props_xyz"] = arrays["props_xyz"].reshape(-1, len(names), 3)
    arrays["wrist_wrench"] = arrays["wrist_wrench"].reshape(-1, 6)
    arrays["finger_force"] = arrays["finger_force"].reshape(-1, 2)
    arrays["target_box"] = arrays["target_box"].reshape(-1, 2, 5)
    np.savez_compressed(folder / "trace.npz", **arrays)
    placed_bin = placed_bin_of(model, data, obj)
    final_gripper_open = bool(metrics.get("gripper_open", True))
    props_z = arrays["props_xyz"][:, :, 2] if len(arrays["props_xyz"]) else np.zeros((1, len(names)))
    cause = analysis.classify(dict(props_z=props_z, names=names, placed_bin=placed_bin,
                                   final_gripper_open=final_gripper_open), obj, bin_colour, success)
    first = analysis.first_lifted(props_z, names)
    distractor_displacement = {n: float(np.linalg.norm(arrays["props_xyz"][-1, i, :2] - arrays["props_xyz"][0, i, :2]))
                               for i, n in enumerate(names) if i and len(arrays["props_xyz"])}
    lift = lift_of(props_z)
    result = dict(trial, outcome=outcome, task_success=bool(success), ticks=len(arrays["raw_action"]),
                  placed_bin=placed_bin, final_gripper_open=final_gripper_open, cause=cause,
                  first_lifted=first, wrong_object=cause == "wrong_object",
                  distractor_displacement_m=distractor_displacement, names=list(names),
                  wall_seconds=time.monotonic() - started, max_lift_m=lift, grasp_lift=bool(lift >= levels.LIFT_SUCCESS_M),
                  recovered=bool(success and disturbance.fired), **disturbance.summary())
    (folder / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
