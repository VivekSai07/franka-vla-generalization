"""One policy rollout on a banked B3 scene. Robosuite image (MuJoCo 3.4.0), GPU for a real policy.

The loop is the sim2sim one (`experiments/robosuite_sim2sim/rollout.py`), copied rather than
imported because that package is not hash-bound and a later edit there must not change this
protocol. Differences: the 12-instruction approval, per-cube tracking so analysis can tell which
cube moved first, and the failure cause in `episode.json`.
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
from experiments.xvla_b3_robosuite_eval import analysis, protocol
from experiments.xvla_b3_robosuite_eval.scenes import present_colours
from experiments.xvla_single_task.contract import observation
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from validation.evaluate_pilot import advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics

MAX_TICKS = 1200
IMAGE_KEYS = tuple(CAMERAS)
CAMERA_NAMES = tuple(CAMERAS.values())
TRACE_KEYS = ('state', 'raw_action', 'applied_action', 'pick_position', 'place_position', 'pick_lift_m',
              'bilateral_target_contact', 'released_stack', 'xy_m', 'z_m', 'cube_xyz')


def seed_trial(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout(policy, pre, post, model, data, renderer, trial, folder):
    """Run one trial from the restored `data`; write cameras.mp4, trace.npz, episode.json in `folder`."""
    pick, place = trial['pick'], trial['place']
    colours = present_colours(model)
    if colours != sorted(trial['colours']):
        raise ValueError('restored scene holds other cubes than the trial plans')
    cube_ids = [model.body(f'target_cube_{c}').id for c in colours]
    scratch = mujoco.MjData(model)
    policy.reset()
    seed_trial(trial['seed'])
    grasp = GraspController(model, data, target=f'target_cube_{pick}')
    synchronize_observation(model, data)
    initial_z = float(data.xpos[grasp.target_id, 2])
    trace = {key: [] for key in TRACE_KEYS}
    stable, success, outcome, started = 0, False, 'timeout', time.monotonic()
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
            trace['cube_xyz'].append(np.stack([data.xpos[i] for i in cube_ids]).copy())
            if not np.isfinite(action).all():
                outcome = 'nonfinite_action'
                break
            bounded, _ = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
            apply_action(model, data, bounded, ACTIVE_JOINTS)
            if step_control(model, data, 1 / 30, ARM_DOF) != 20:
                raise ValueError('Physics substep count changed')
            metrics = fresh_stack_metrics(model, data, scratch, pick, place)
            diagnostics = target_diagnostics(model, data, pick, place, grasp, initial_z)
            trace['applied_action'].append(bounded.copy())
            for key in ('pick_position', 'place_position', 'pick_lift_m', 'bilateral_target_contact'):
                trace[key].append(diagnostics[key])
            for key in ('released_stack', 'xy_m', 'z_m'):
                trace[key].append(metrics[key])
            stable, success = advance_streak(stable, metrics)
            if success:
                outcome = 'success'
                break
        observe()
        for packet in stream.encode():
            video.mux(packet)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays['applied_action'] = arrays['applied_action'].reshape(-1, 8)
    np.savez_compressed(folder / 'trace.npz', **arrays, colours=np.asarray(colours))
    analysed = dict(cube_xyz=arrays['cube_xyz'], colours=np.asarray(colours), applied_action=arrays['applied_action'])
    lift = arrays['pick_lift_m'] if len(arrays['pick_lift_m']) else np.zeros(1)
    result = dict(trial, outcome=outcome, task_success=bool(success), ticks=len(arrays['raw_action']),
                  max_pick_lift_m=float(lift.max()),
                  ever_bilateral_contact=bool(np.any(arrays['bilateral_target_contact'])),
                  final_xy_m=float(arrays['xy_m'][-1]) if len(arrays['xy_m']) else None,
                  final_z_m=float(arrays['z_m'][-1]) if len(arrays['z_m']) else None,
                  first_lifted=analysis.first_lifted(analysed) if len(arrays['cube_xyz']) else None,
                  cause=analysis.classify(analysed, pick, success) if len(arrays['cube_xyz']) else 'never_lifted',
                  wall_seconds=time.monotonic() - started)
    (folder / 'episode.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
