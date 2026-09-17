"""Explicit minimal XVLA port of the bound baseline; no config spoofing or new score."""
from contextlib import ExitStack
import json
import random
import time

import av
import mujoco
import numpy as np
import torch

from data_collection.benchmark import (apply_action, capture_state, require_complete, restore_timing,
                                      step_control, synchronize_observation, write_json)
from validation.evaluate_pilot import restore_scene, advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from experiments.single_task.evaluate import derive_initial, training_bytes
from experiments.single_task.contract import summarize as base_summary
from . import contract as c


def summarize(rows):
    result = base_summary(rows)
    for name in ('seen','held_out'):
        row = result[name]
        row['success_rate'] = row['successes']/row['trials'] if row['trials'] else None
    return result


def run_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal):
    if (policy.config.chunk_size, policy.config.n_action_steps) != (30, 10):
        raise ValueError("Expected chunk 30/action steps 10")
    data, scratch = restore_scene(model, bank, entry), mujoco.MjData(model)
    policy.reset()
    random.seed(c.SEED + entry['episode_index'])
    np.random.seed(c.SEED + entry['episode_index'])
    torch.manual_seed(c.SEED + entry['episode_index'])
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
    trace = {k: [] for k in ("state", "sim_time", "raw_action", "applied_action", "post_state", "post_sim_time")}
    stable = calls = 0
    success, outcome = False, "timeout"
    original = policy._get_action_chunk
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    started = time.monotonic()
    with ExitStack() as resources:
        renderer = mujoco.Renderer(model, height=512, width=512)
        resources.callback(renderer.close)
        record = resources.enter_context(av.open(str(output / "cameras.mp4"), "w"))
        stream = record.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        def images_and_video():
            images = []
            for camera in manifest["camera_names"]:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")):
                record.mux(packet)
            return images
        policy._get_action_chunk = counted
        try:
            for _ in range(1200):
                synchronize_observation(model, data)
                state = data.qpos[:9].copy()
                frame = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": instruction}
                for key, image in zip(("observation.images.image", "observation.images.image2"), images_and_video()):
                    frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
                processed = pre(c.observation(frame))
                if _ == 0:
                    write_json(output/'input_precheck.json',c.check_model_inputs(policy,processed,frame['observation.state'][None]))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    action = post(policy.select_action(processed)).squeeze(0).cpu().numpy()
                if action.shape != (8,):
                    raise ValueError("Policy action shape is not eight physical commands")
                for key, value in (("state", state), ("sim_time", float(data.time)), ("raw_action", action.copy())):
                    trace[key].append(value)
                if not np.isfinite(action).all():
                    outcome = "nonfinite_action"
                    break
                bounded, clips = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
                apply_action(model, data, bounded, ACTIVE_JOINTS)
                if step_control(model, data, manifest["control_dt_requested"], ARM_DOF) != 20:
                    raise ValueError("Physics substep count changed")
                metrics = fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"])
                diagnostics = target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)
                for key, value in {"applied_action": bounded.copy(), "post_state": data.qpos[:9].copy(),
                        "post_sim_time": float(data.time), **clips, **metrics, **diagnostics}.items():
                    trace.setdefault(key, []).append(value)
                if not np.isfinite(capture_state(model, data)).all():
                    raise ValueError("Nonfinite simulator state")
                stable, success = advance_streak(stable, metrics)
                if success:
                    outcome = "success"
                    break
            images_and_video()  # Include terminal scene, not only pre-command observations.
        finally:
            policy._get_action_chunk = original
            for packet in stream.encode():
                record.mux(packet)
    attempts, ticks = len(trace["raw_action"]), len(trace["applied_action"])
    if calls != (attempts + 9)//10:
        raise ValueError("Actual chunk predictions differ from ceil(action_attempts/10)")
    if not np.isclose(data.time - initial_time, ticks/30, rtol=0, atol=1e-8):
        raise ValueError("Simulator duration differs from completed control ticks/30")
    masks = np.asarray(trace.get("joint_clip_mask", []), dtype=bool).reshape(-1, 7)
    grip = np.asarray(trace.get("gripper_clipped", []), dtype=bool)
    interventions = int(np.sum(np.any(masks, axis=1) | grip))
    terminal = {**fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"]),
                **target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)}
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays["applied_action"] = arrays["applied_action"].reshape(-1, 8)
    np.savez_compressed(output / "trace.npz", **arrays, initial_integration_state=initial_state,
                        terminal_integration_state=capture_state(model, data), terminal_state=data.qpos[:9].copy(),
                        terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy(), terminal_sim_time=float(data.time),
                        **{f"terminal_{k}": np.asarray(v) for k, v in terminal.items()})
    result = {**entry, "ordinal": ordinal, "rollout_seed": c.SEED + entry['episode_index'], "instruction": instruction,
              "outcome": outcome, "ticks": attempts, "action_attempts": attempts,
              "completed_control_ticks": ticks, "sim_seconds": ticks/30,
              "actual_chunk_predictions": calls, "chunk_size": 30, "n_action_steps": 10,
              "task_success": success, "unassisted_success": success and interventions == 0,
              "intervention_steps": interventions, "joint_clipped_ticks": int(np.any(masks, axis=1).sum()),
              "joint_clipped_ticks_per_dimension": masks.sum(axis=0).tolist(), "gripper_clipped_ticks": int(grip.sum()),
              "gripper_thresholded_ticks": int(np.sum(trace.get("gripper_thresholded", []))),
              "max_joint_clip_rad": max(trace.get("max_joint_clip_rad", [0.])),
              "gripper_adapter": "threshold_at_127_not_counted_as_clipping", "success_streak": stable,
              "final_metrics": {k: np.asarray(v).tolist() for k, v in terminal.items()},
              "wall_seconds": time.monotonic() - started}
    write_json(output / "episode.json", result)
    return result


def verify_final(output,binding):
    report = json.loads((output/'result.json').read_text())
    checkpoint = output/'checkpoints/005000'
    expected = {**c.recipe(False),'status':'complete','completed_updates':5000,
                'selected_checkpoint_update':5000,'final_data_binding_passed':True,
                'strict_reload_fp32_exact':True,'reload_max_absolute_error':0.,'binding':binding}
    if (any(report.get(k) != v for k,v in expected.items())
            or report['cpu_admission_sha256'] != c.sha256(c.CPU_ADMISSION)
            or report['final_checkpoint_files'] != c.inventory(checkpoint)
            or report['training_stats_sha256'] != c.sha256(output/'training_stats.json')
            or [m['update'] for m in report['checkpoints']] != [1,1000,2000,3000,4000,5000]):
        raise ValueError('Complete final-only5000 model/data/source/cache gate failed')
    config = json.loads((checkpoint/'config.json').read_text())
    if (config != json.loads((output/'initial_config/config.json').read_text())
            or (config['type'],config['chunk_size'],config['n_action_steps'],config['dtype']) != ('xvla',30,10,'float32')
            or {k:tuple(v['shape']) for k,v in config['input_features'].items()} != c.OBS
            or {k:v['shape'] for k,v in config['output_features'].items()} != {'action':[8]}):
        raise ValueError('Final saved XVLA real schema mismatch')
    return checkpoint


def run(output):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from .runner import failure, require_disk
    directory = output/'evaluation'
    c.claim_directory(directory)
    report = dict(status='preparing',stage='evaluation_admission',episodes=[],max_control_ticks=1200,
                  max_sim_seconds=40,strict_score_ticks=30,seed=c.SEED,final_only=True,
                  retries=False,resume=False,**summarize([]))
    before = training_bytes(output)
    try:
        binding = c.admission()
        checkpoint = verify_final(output,binding)
        manifest = require_complete(c.DATA)
        entries = c.old.select_entries(manifest)
        scene = (c.ROOT/manifest['scene_path']).resolve()
        if c.ROOT not in scene.parents:
            raise ValueError('Scene outside source root')
        model = mujoco.MjModel.from_xml_path(str(scene))
        if restore_timing(model,manifest)['substeps_per_control'] != 20:
            raise ValueError('Expected unchanged physics timing')
        initial = directory/'initial_states'
        initial.mkdir()
        planned = []
        for entry in entries:
            adapted,evidence = derive_initial(model,c.DATA/'replay',entry,initial)
            planned.append(dict(entry=adapted,provenance=evidence,ordinal=entry['episode_index']+1,
                                rollout_seed=c.SEED+entry['episode_index']))
        resets = c.inventory(initial)
        config = PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
        config.device = 'cpu'
        policy = c.p.load_weights(XVLAPolicy(config),checkpoint/'model.safetensors').to('cuda')
        policy.config.device = 'cuda'
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        pre,post = make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
        report.update(status='running',stage='rollouts',trials=planned,binding=binding,
                      training_files=before,privileged_policy_inputs=False)
        write_json(directory/'result.json',report)
        for trial in planned:
            require_disk(output,2)
            folder = directory/f"episode_{trial['entry']['episode_index']:06d}"
            folder.mkdir(exist_ok=False)
            report['active_episode'] = trial['entry']['episode_index']
            write_json(directory/'result.json',report)
            row = run_trial(policy,pre,post,model,manifest,trial['entry'],initial,folder,trial['ordinal'])
            if row['rollout_seed'] != trial['rollout_seed'] or row['outcome'] == 'nonfinite_action':
                raise ValueError('Seed mismatch or nonfinite action; no retries')
            report['episodes'].append(row)
            report.update(summarize(report['episodes']))
            write_json(directory/'result.json',report)
            print('EVAL',json.dumps({k:report[k] for k in ('finished_trials','seen','held_out')}),flush=True)
        after = c.admission()
        verify_final(output,after)
        if after != binding or c.inventory(initial) != resets or report['finished_trials'] != 13:
            raise ValueError('Final evaluation data/source/cache/reset gate failed')
        report.update(status='complete',stage='complete',active_episode=None,training_bytes_unchanged=True,
                      final_data_binding_passed=True,no_extension=True)
    except BaseException as exc:
        failure(directory,report,exc)
        raise
    finally:
        if training_bytes(output) != before:
            exc = ValueError('Training bytes changed during evaluation')
            failure(directory,report,exc)
            raise exc
    write_json(directory/'result.json',report)