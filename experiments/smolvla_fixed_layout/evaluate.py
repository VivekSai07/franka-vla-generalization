"""Final5000 only:72 observation probes,20 explicit seeds,one unchanged full reset."""
from contextlib import ExitStack
import json
import random
import time

import av
import mujoco
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from data_collection.benchmark import (apply_action, capture_state, restore_timing,
                                      step_control, synchronize_observation, write_json)
from validation.evaluate_pilot import restore_scene, advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF, SCENE_PATH
from experiments.fixed_layout.contract import SOURCE_SCENE_SHA256
from . import contract as c
from .runner import inference_batch


def evaluation_plan():
    return [dict(ordinal=i+1, seed=20260918+i, reset_file='reset.npz', source_episode_index=7) for i in range(20)]


def seed_trial(seed):
    if type(seed) is not int or seed not in range(20260918, 20260938): raise ValueError('Only20approved rollout seeds')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def summarize(rows):
    def finite(value):
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
        return bool(np.isfinite(value)) if isinstance(value, (float, np.floating)) else True
    complete = (finite(rows) and [r.get('seed') for r in rows] == list(range(20260918, 20260938))
                and all(type(r.get('task_success')) is bool and r.get('outcome') ==
                        ('success' if r['task_success'] else 'timeout') for r in rows))
    successes = sum(r.get('task_success') is True for r in rows)
    return dict(planned_trials=20, finished_trials=len(rows), successes=successes,
                success_rate=successes/len(rows) if rows else None, all_trials_valid=complete,
                capability_passed=complete and successes>=18, required_successes=18,
                scope='sampling_reliability_one_selected_geometry_not_spatial_generalization')


def require_complete_trials(rows):
    if not summarize(rows)['all_trials_valid']: raise ValueError('All20approved finite trials must finish')


def reset_entry(binding):
    if c.sha256(c.PARENT/'reset.npz') != binding['reset_sha256']: raise ValueError('Fixed reset changed')
    return dict(episode_index=7, source_episode_index=7, pick='red', place='green', colors=['red', 'green'],
                file='reset.npz', sha256=binding['reset_sha256'], scene_hash=SOURCE_SCENE_SHA256)


def stage_metrics(trace):
    """Persisted descriptive stages only; never supplied to policy inference."""
    actions = np.asarray(trace['applied_action']).reshape(-1, 8)
    contact = np.asarray(trace['bilateral_target_contact'], dtype=bool)
    lift = np.asarray(trace['pick_lift_m'], dtype=float)
    released = np.asarray(trace['released_stack'], dtype=bool)
    if any(len(x)!=len(actions) for x in (contact, lift, released)) or not np.isfinite(actions).all() or not np.isfinite(lift).all():
        raise ValueError('Invalid saved action/contact stages')
    closed = actions[:, 7]<=127
    def first(mask):
        ticks = np.flatnonzero(mask)
        return int(ticks[0]) if ticks.size else None
    def sustained(mask, count):
        streak = 0
        for i, value in enumerate(mask):
            streak = streak+1 if value else 0
            if streak>=count: return i
        return None
    closure = first(closed)
    after = np.arange(len(actions))>closure if closure is not None else np.zeros(len(actions), dtype=bool)
    return dict(first_closure_tick=closure, sustained_closure_30_tick=sustained(closed, 30),
                first_bilateral_contact_tick=first(contact), sustained_bilateral_5_tick=sustained(contact, 5),
                first_contact_lift_gt_5cm_tick=first(contact & (lift>.05)),
                first_release_after_closure_tick=first(after & ~closed), stable_stack_30_tick=sustained(released, 30),
                maximum_lift_m=float(lift.max()) if len(lift) else 0., bilateral_contact_ticks=int(contact.sum()), tick_origin=0)


def offline_probes(policy, pre, post, ds, device='cuda', output=None):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = policy.training
    devices = [torch.cuda.current_device()] if device=='cuda' else []
    rows, predictions, targets = [], [], []
    try:
        policy.eval()
        with torch.random.fork_rng(devices=devices):
            for episode in c.TRAIN+c.VAL:
                for frame in (60, 107, 347):
                    index = episode*528+frame
                    sample = c.policy_row(ds[index], training=False)
                    target = sample['action'].detach().cpu().clone()
                    batch = inference_batch(pre, c.observation(sample))
                    random.seed(c.SEED+index); np.random.seed(c.SEED+index); torch.manual_seed(c.SEED+index)
                    # Prior SmolVLA inference is not BF16 autocast (only training is).
                    with torch.inference_mode(): prediction = post(policy.predict_action_chunk(batch)).detach().cpu()
                    if prediction.shape != (1, 50, 8) or not torch.isfinite(prediction).all():
                        raise ValueError('Finite physical50x8 prediction required')
                    error = (prediction[0]-target).abs()
                    rows.append(dict(episode=episode, frame=frame, row=index,
                                     split='train' if episode in c.TRAIN else 'validation',
                                     arm_mae_rad=float(error[:, :7].mean()), first10_arm_mae_rad=float(error[:10, :7].mean()),
                                     gripper_binary_accuracy=float(((prediction[0, :, 7]>127)==(target[:, 7]>127)).float().mean())))
                    predictions.append(prediction[0].numpy()); targets.append(target.numpy())
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); policy.train(training)
    result = dict(checkpoint_update=5000, observation_only=True, validation_used_for_training=False,
                  validation_is_spatial_holdout=False, rows=rows)
    for split in ('train', 'validation'):
        selected = [r for r in rows if r['split']==split]
        result[split] = {'probes': len(selected), **{key: float(np.mean([r[key] for r in selected]))
                        for key in ('arm_mae_rad', 'first10_arm_mae_rad', 'gripper_binary_accuracy')}}
    if output is not None:
        with (output/'offline_probe_arrays.npz').open('xb') as stream:
            np.savez_compressed(stream, predictions=np.asarray(predictions), targets=np.asarray(targets),
                                rows=np.asarray([r['row'] for r in rows]))
        write_json(output/'offline_probes.json', result)
    return result


class CountedPolicy(SmolVLAPolicy):
    """Native10-action queue; subclass counting, never runtime method replacement."""
    chunk_calls = 0

    def _get_action_chunk(self, *args, **kwargs):
        self.chunk_calls += 1
        return super()._get_action_chunk(*args, **kwargs)


def load_counted(checkpoint):
    """Explicit precision loader port for the counting subclass; promote BEFORE copy."""
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_model
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.load_vlm_weights = False
    policy = CountedPolicy(config)
    c.h.promote_trainable_parameters(policy)
    policy.to(config.device)
    load_model(policy, str(checkpoint/'model.safetensors'), strict=True, device=config.device)
    policy.eval()
    return policy


def run_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed):
    if (policy.config.chunk_size, policy.config.n_action_steps) != (50, 10):
        raise ValueError('Expected chunk50/actionsteps10')
    if not 1 <= ordinal <= 20 or seed != 20260917+ordinal or entry['episode_index'] != 7:
        raise ValueError('Explicit approved seed/ordinal/reset identity required')
    data, scratch = restore_scene(model, bank, entry), mujoco.MjData(model)
    policy.reset()
    seed_trial(seed)
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
    trace = {k: [] for k in ('state', 'sim_time', 'raw_action', 'applied_action', 'post_state', 'post_sim_time')}
    stable = 0
    before_calls = policy.chunk_calls
    success, outcome = False, 'timeout'
    started = time.monotonic()
    with ExitStack() as resources:
        renderer = mujoco.Renderer(model, height=512, width=512)
        resources.callback(renderer.close)
        record = resources.enter_context(av.open(str(output/'cameras.mp4'), 'w'))
        stream = record.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'fast'}
        def images_and_video():
            images = []
            for camera in manifest['camera_names']:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format='rgb24')):
                record.mux(packet)
            return images
        try:
            for _ in range(1200):
                synchronize_observation(model, data)
                state = data.qpos[:9].copy()
                frame = {'observation.state': torch.tensor(state, dtype=torch.float32), 'task': instruction}
                for key, image in zip(('observation.images.image', 'observation.images.image2'), images_and_video()):
                    frame[key] = torch.from_numpy(image).permute(2, 0, 1).float()/255
                with torch.inference_mode():
                    action = post(policy.select_action(inference_batch(pre, c.observation(frame)))).squeeze(0).cpu().numpy()
                if action.shape != (8,):
                    raise ValueError('Policy action shape is not eight physical commands')
                for key, value in (('state', state), ('sim_time', float(data.time)), ('raw_action', action.copy())):
                    trace[key].append(value)
                if not np.isfinite(action).all():
                    outcome = 'nonfinite_action'
                    break
                bounded, clips = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
                apply_action(model, data, bounded, ACTIVE_JOINTS)
                if step_control(model, data, manifest['control_dt_requested'], ARM_DOF) != 20:
                    raise ValueError('Physics substep count changed')
                metrics = fresh_stack_metrics(model, data, scratch, entry['pick'], entry['place'])
                diagnostics = target_diagnostics(model, data, entry['pick'], entry['place'], grasp, initial_z)
                for key, value in {'applied_action': bounded.copy(), 'post_state': data.qpos[:9].copy(),
                        'post_sim_time': float(data.time), **clips, **metrics, **diagnostics}.items():
                    trace.setdefault(key, []).append(value)
                if not np.isfinite(capture_state(model, data)).all():
                    raise ValueError('Nonfinite simulator state')
                stable, success = advance_streak(stable, metrics)
                if success:
                    outcome = 'success'
                    break
            images_and_video()
        finally:
            for packet in stream.encode(): record.mux(packet)
    calls = policy.chunk_calls-before_calls
    attempts, ticks = len(trace['raw_action']), len(trace['applied_action'])
    if calls != (attempts+9)//10: raise ValueError('Chunk calls differ from ceil(attempts/10)')
    if not np.isclose(data.time-initial_time, ticks/30, rtol=0, atol=1e-8): raise ValueError('Control timing changed')
    masks = np.asarray(trace.get('joint_clip_mask', []), dtype=bool).reshape(-1, 7)
    grip = np.asarray(trace.get('gripper_clipped', []), dtype=bool)
    interventions = int(np.sum(np.any(masks, axis=1) | grip))
    terminal = {**fresh_stack_metrics(model, data, scratch, entry['pick'], entry['place']),
                **target_diagnostics(model, data, entry['pick'], entry['place'], grasp, initial_z)}
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays['applied_action'] = arrays['applied_action'].reshape(-1, 8)
    with (output/'trace.npz').open('xb') as file:
        np.savez_compressed(file, **arrays, initial_integration_state=initial_state,
                            terminal_integration_state=capture_state(model, data), terminal_state=data.qpos[:9].copy(),
                            terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy(), terminal_sim_time=float(data.time),
                            **{f'terminal_{k}': np.asarray(v) for k, v in terminal.items()})
    result = {**entry, 'ordinal': ordinal, 'seed': seed, 'rollout_seed': seed, 'instruction': instruction,
              'outcome': outcome, 'ticks': attempts, 'action_attempts': attempts, 'completed_control_ticks': ticks,
              'sim_seconds': ticks/30, 'actual_chunk_predictions': calls, 'chunk_size': 50, 'n_action_steps': 10,
              'task_success': success, 'unassisted_success': success and interventions==0,
              'intervention_steps': interventions, 'joint_clipped_ticks': int(np.any(masks, axis=1).sum()),
              'joint_clipped_ticks_per_dimension': masks.sum(axis=0).tolist(), 'gripper_clipped_ticks': int(grip.sum()),
              'gripper_thresholded_ticks': int(np.sum(trace.get('gripper_thresholded', []))),
              'max_joint_clip_rad': max(trace.get('max_joint_clip_rad', [0.])),
              'gripper_adapter': 'threshold_at_127_not_counted_as_clipping', 'success_streak': stable,
              'stages': stage_metrics(arrays) if ticks else {}, 'reset_sha256': entry['sha256'],
              'final_metrics': {k: np.asarray(v).tolist() for k, v in terminal.items()},
              'wall_seconds': time.monotonic()-started}
    write_json(output/'episode.json', result)
    return result


def verify_final(output, binding):
    report = json.loads((output/'result.json').read_text())
    c.require_fields(report, {**c.recipe(False), 'status': 'complete', 'completed_updates': 5000,
                     'selected_checkpoint_update': 5000, 'final_data_binding_passed': True, 'binding': binding,
                     'training_rows': 10560, 'complete_training_chunks': 9580, 'scratch_used_for_initialization': False},
                     'Complete final-only5000 report required')
    if (report.get('cpu_admission_sha256') != c.sha256(c.CPU_ADMISSION)
            or report.get('training_stats_sha256') != c.sha256(output/'training_stats.json')
            or report.get('scratch_result_sha256') != c.sha256(c.SCRATCH/'result.json')):
        raise ValueError('Final CPU/statistics/scratch binding changed')
    c.verify_checkpoints(report, output, False)
    stats = json.loads((output/'training_stats.json').read_text())
    if stats != json.loads(c.CPU_ADMISSION.read_text())['training_stats']: raise ValueError('Final stats differ from CPU')
    c.validate_scratch(json.loads((c.SCRATCH/'result.json').read_text()), binding, c.SCRATCH, stats, c.sha256(c.CPU_ADMISSION))
    checkpoint = output/'checkpoints/005000'
    config = json.loads((checkpoint/'config.json').read_text())
    if (config != json.loads((output/'initial_config/config.json').read_text())
            or (config['type'], config['chunk_size'], config['n_action_steps'], config['n_obs_steps']) != ('smolvla', 50, 10, 1)
            or {k: tuple(v['shape']) for k, v in config['input_features'].items()} != c.OBS
            or {k: v['shape'] for k, v in config['output_features'].items()} != {'action': [8]}):
        raise ValueError('Final SmolVLA config/schema changed')
    return checkpoint


def training_bytes(output):
    return {str(path.relative_to(output)): c.sha256(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.relative_to(output).parts[0]!='evaluation'}


def run(output):
    from lerobot.policies.factory import make_pre_post_processors
    from .runner import failure, require_disk
    directory = output/'evaluation'
    directory.mkdir(exist_ok=False)
    report = dict(status='preparing', stage='evaluation_admission', episodes=[], max_control_ticks=1200,
                  max_sim_seconds=40, strict_score_ticks=30, final_only=True, retries=False, resume=False, **summarize([]))
    before = training_bytes(output)
    try:
        ds, starts, stats, binding = c.prepare_data()
        checkpoint = verify_final(output, binding)
        manifest = json.loads((c.PARENT/'configuration.json').read_text())
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        if restore_timing(model, manifest)['substeps_per_control'] != 20: raise ValueError('Physics timing changed')
        entry = reset_entry(binding)
        restore_scene(model, c.PARENT, entry)  # Full state restore only, never resettle.
        torch.use_deterministic_algorithms(True)
        policy = load_counted(checkpoint)
        for parameter in policy.parameters(): parameter.requires_grad_(False)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        report.update(status='running', stage='offline_probes', binding=binding, training_files=before,
                      trials=evaluation_plan(), privileged_policy_inputs=False)
        write_json(directory/'result.json', report)
        report['offline_probes'] = offline_probes(policy, pre, post, ds, output=directory)
        if c.admission() != binding: raise ValueError('Probe phase binding changed')
        verify_final(output, binding)
        report['stage'] = 'rollouts'
        for trial in evaluation_plan():
            require_disk(output, 2)
            folder = directory/f"trial_{trial['ordinal']:02d}_seed_{trial['seed']}"
            folder.mkdir(exist_ok=False)
            report['active_seed'] = trial['seed']; write_json(directory/'result.json', report)
            row = run_trial(policy, pre, post, model, manifest, entry, c.PARENT, folder, trial['ordinal'], trial['seed'])
            if row['seed'] != trial['seed'] or row['outcome']=='nonfinite_action': raise ValueError('Seed/nonfinite action failure; no retry')
            report['episodes'].append(row); report.update(summarize(report['episodes']))
            write_json(directory/'result.json', report)
            print('EVAL', json.dumps({k: report[k] for k in ('finished_trials', 'successes', 'capability_passed')}), flush=True)
        require_complete_trials(report['episodes'])
        after = c.admission(); verify_final(output, after)
        if after != binding: raise ValueError('Final evaluation binding changed')
        report.update(status='complete', stage='complete', active_seed=None, training_bytes_unchanged=True,
                      final_data_binding_passed=True, no_extension=True)
    except BaseException as exc:
        failure(directory, report, exc); raise
    finally:
        if training_bytes(output) != before:
            exc = ValueError('Training bytes changed during evaluation')
            failure(directory, report, exc); raise exc
    write_json(directory/'result.json', report)