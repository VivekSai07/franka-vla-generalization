"""One policy rollout on an M0 layout. Robosuite image (MuJoCo 3.4.0); a GPU only for a real policy.

Forked from `experiments/xvla_props_p2_eval/rollout.py` (P2's is unchanged). Differences, and only these: the policy is an
adapter (`adapters.PolicyAdapter`) instead of an inline X-VLA call, the observation's state comes from the arm's state
builder, and right after `synchronize_observation` the S0 signals (`experiments/vla_inputs_signals/signals.py`, computed on
a scratch copy, sealed) are appended to the trace for every arm: `wrist_wrench` (ticks, 6), `finger_force` (ticks, 2),
`target_box` (ticks, 2, 5). P2's success rule, lift rule and failure classifier are imported unchanged from `analysis`.
"""
import json
import random
import time

import av
import numpy as np
import torch

from data_collection.benchmark import apply_action, step_control, synchronize_observation
from experiments.paired_demonstrations.media import CAMERAS
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import in_bin, placement, success_streak
from experiments.vla_inputs_eval import adapters
from experiments.vla_inputs_signals import signals
from experiments.xvla_props_p2_eval import analysis
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from validation.rollout_diagnostics import bound_action

MAX_TICKS = c.MAX_TICKS
CAMERA_NAMES = tuple(CAMERAS.values())
TRACE_KEYS = ("state", "raw_action", "applied_action", "props_xyz", "xy_margin_m", "z_m", "lowest_z_m",
              "gripper_open", "finger_contact")
SIGNAL_KEYS = ("wrist_wrench", "finger_force", "target_box")


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


def run(adapter, state_builder, model, data, names, renderer, trial, folder, max_ticks=MAX_TICKS):
    """Run one trial from the restored `data`; write cameras.mp4, trace.npz, episode.json in `folder`."""
    obj, bin_colour = trial["object"], trial["bin"]
    if names[0] != obj:
        raise ValueError("names must list the target first")
    adapter.reset()
    seed_trial(trial["seed"])
    synchronize_observation(model, data)
    prop_ids = [model.body(f"prop_{name}").id for name in names]
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

        for _ in range(max_ticks):
            synchronize_observation(model, data)
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
            apply_action(model, data, bounded, ACTIVE_JOINTS)
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
    result = dict(trial, outcome=outcome, task_success=bool(success), ticks=len(arrays["raw_action"]),
                  placed_bin=placed_bin, final_gripper_open=final_gripper_open, cause=cause,
                  first_lifted=first, wrong_object=cause == "wrong_object",
                  distractor_displacement_m=distractor_displacement, names=list(names),
                  wall_seconds=time.monotonic() - started)
    (folder / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
