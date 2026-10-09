"""One policy rollout on a reserved props P1 layout. Robosuite image (MuJoCo 3.4.0), GPU for a real policy.

The loop is the sim2sim/B3 one, reproduced rather than imported (neither is hash-bound, and a later
edit there must not change this protocol). Differences from B3: one object and two bins instead of
pick/place cubes, the P1 success rule (`props_scene.metrics`), a 900-tick cap, and `placed_bin` /
`wrong_bin` tracking instead of cube-stacking causes.
"""
import json
import random
import time

import av
import mujoco
import numpy as np
import torch

from data_collection.benchmark import apply_action, step_control, synchronize_observation
from experiments.paired_demonstrations.media import CAMERAS
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import in_bin, placement, success_streak
from experiments.xvla_props_p1_eval import analysis, protocol
from experiments.xvla_single_task.contract import observation
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from validation.rollout_diagnostics import bound_action

MAX_TICKS = c.MAX_TICKS
IMAGE_KEYS = tuple(CAMERAS)
CAMERA_NAMES = tuple(CAMERAS.values())
TRACE_KEYS = ('state', 'raw_action', 'applied_action', 'prop_xyz', 'xy_margin_m', 'z_m',
              'lowest_z_m', 'gripper_open', 'finger_contact')


def seed_trial(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def placed_bin_of(model, data, obj):
    """Whichever bin the prop rests stably inside at the moment this is called, or None."""
    for colour in c.BINS:
        if placement(model, data, obj, colour)['resting_in_bin']:
            return colour
    return None


def rollout(policy, pre, post, model, data, renderer, trial, folder):
    """Run one trial from the restored `data`; write cameras.mp4, trace.npz, episode.json in `folder`."""
    obj, bin_colour = trial['object'], trial['bin']
    policy.reset()
    seed_trial(trial['seed'])
    synchronize_observation(model, data)
    prop_id = model.body(f'prop_{obj}').id
    trace = {key: [] for key in TRACE_KEYS}
    stable, success, outcome, started = 0, False, 'timeout', time.monotonic()
    metrics = {}
    with av.open(str(folder / 'cameras.mp4'), 'w') as video:
        stream = video.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'fast'}

        def observe():
            images = []
            for camera in CAMERA_NAMES:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format='rgb24')):
                video.mux(packet)
            return images

        for _ in range(MAX_TICKS):
            synchronize_observation(model, data)
            state = data.qpos[:9].copy()
            frame = {'observation.state': torch.tensor(state, dtype=torch.float32), 'task': trial['instruction']}
            for key, image in zip(IMAGE_KEYS, observe()):
                frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
            with protocol.eval_instruction(trial['instruction']):
                processed = pre(observation(frame))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                action = post(policy.select_action(processed)).squeeze(0).float().cpu().numpy()
            if action.shape != (8,):
                raise ValueError('Policy action shape is not eight physical commands')
            trace['state'].append(state)
            trace['raw_action'].append(action.copy())
            trace['prop_xyz'].append(data.xpos[prop_id].copy())
            if not np.isfinite(action).all():
                outcome = 'nonfinite_action'
                break
            bounded, _ = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
            apply_action(model, data, bounded, ACTIVE_JOINTS)
            if step_control(model, data, 1 / 30, ARM_DOF) != 20:
                raise ValueError('Physics substep count changed')
            metrics = in_bin(model, data, obj, bin_colour)
            trace['applied_action'].append(bounded.copy())
            for key in ('xy_margin_m', 'z_m', 'lowest_z_m', 'gripper_open', 'finger_contact'):
                trace[key].append(metrics[key])
            stable, success = success_streak(stable, metrics)
            if success:
                outcome = 'success'
                break
        observe()
        for packet in stream.encode():
            video.mux(packet)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays['applied_action'] = arrays['applied_action'].reshape(-1, 8)
    np.savez_compressed(folder / 'trace.npz', **arrays)
    placed_bin = placed_bin_of(model, data, obj)
    final_gripper_open = bool(metrics.get('gripper_open', True))
    cause = analysis.classify({'prop_z': arrays['prop_xyz'][:, 2] if len(arrays['prop_xyz']) else np.zeros(1),
                               'placed_bin': placed_bin, 'final_gripper_open': final_gripper_open},
                              bin_colour, success)
    result = dict(trial, outcome=outcome, task_success=bool(success), ticks=len(arrays['raw_action']),
                  placed_bin=placed_bin, final_gripper_open=final_gripper_open, cause=cause,
                  wall_seconds=time.monotonic() - started)
    (folder / 'episode.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
