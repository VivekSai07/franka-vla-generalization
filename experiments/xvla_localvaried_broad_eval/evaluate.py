"""Cross-evaluation: the frozen X-VLA LOCAL-VARIED checkpoint on the BROAD holdout.

Zero-shot out-of-distribution probe. The policy was fine-tuned on 48 episodes confined to
+-1 cm around one base layout; here it is rolled out on the 36 reserved envelope-F7 layouts
that `experiments.xvla_broad_workspace_eval` uses, with an identical roster, identical seeds
and identical scene derivation. The only variable is which dataset the checkpoint saw.

This is what makes the broad-workspace number interpretable: 68.1% on its own holdout means
nothing without knowing what a narrowly-trained policy scores on the same 72 trials.

Binding is deliberately hybrid and both halves are checked:
  * `c` is `experiments.xvla_local_varied.contract` — it verifies the local-varied checkpoint,
    its training report and its own dataset for the offline probes.
  * `BROAD_PARENT` is the sealed broad-workspace run — it supplies plan, bank and scenes, and
    its seal is verified independently so holdout identity cannot drift.

Only `broad_holdout` runs; fixed_parity for this checkpoint is already recorded at 20/20 by
`experiments.xvla_local_varied_eval` and re-running it here would prove nothing new.
"""
from contextlib import ExitStack, nullcontext
import hashlib
import json
import random
import time

import av
import mujoco
import numpy as np
import torch
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

from data_collection.benchmark import (apply_action, capture_state, restore_state, restore_timing,
                                       step_control, synchronize_observation, write_json)
from data_collection.pilot_integrity import scene_digest
from validation.evaluate_pilot import advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF, SCENE_PATH
from experiments.fixed_layout.contract import SOURCE_SCENE_SHA256
from experiments.fixed_layout.source import Expert as BaseExpert, load_reset
from experiments.broad_workspace_data import plan as broad_plan
from experiments.broad_workspace_data.scenes import derive_scene, scene_hash, _validate_broad_geometry
from experiments.xvla_local_varied import contract as c


# Sealed broad-workspace run supplying the holdout; independent of the local-varied parent.
BROAD_PARENT = c.ROOT/'outputs/broad_workspace_data/broad_workspace_20260920'
BROAD_SEAL = '0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85'


TASK = 'pick up the red cube and place it on top of the green cube'
FIXED_PARITY_SEEDS = tuple(range(20260918, 20260938))
FIXED_PARITY_GATE = 18
BROAD_LAYOUT_COUNT = 36  # identical roster to experiments.xvla_broad_workspace_eval
BROAD_GATE = 65
MODES = ('broad_holdout',)


def _plan():
    """Deterministic frozen broad-workspace plan; identity is checked once per evaluation."""
    return broad_plan.make_plan()


def broad_policy_seeds():
    return tuple(_plan()['policy_seeds'])


def broad_layout_ids():
    """First 36 of the 80 layouts the collection reserved and never executed."""
    return tuple(_plan()['evaluation_layout_ids'][:BROAD_LAYOUT_COUNT])


def evaluation_plan_broad():
    """72 trials: layout-major, seed-minor cartesian product."""
    plan = _plan()
    trials, ordinal = [], 0
    for layout_id in plan['evaluation_layout_ids'][:BROAD_LAYOUT_COUNT]:
        for seed in plan['policy_seeds']:
            ordinal += 1
            trials.append(dict(ordinal=ordinal, layout_id=layout_id, seed=seed,
                               source_episode_index=7))
    if len(trials) != 72: raise ValueError('Broad roster must be 36 layouts * 2 seeds')
    return trials


def evaluation_plan_fixed_parity():
    """20 trials sharing the source reset with the fixed-layout baseline protocol."""
    return [dict(ordinal=i+1, seed=FIXED_PARITY_SEEDS[i], reset_file='source.npz',
                 source_episode_index=7) for i in range(20)]


def seed_trial_fixed_parity(seed):
    if type(seed) is not int or seed not in FIXED_PARITY_SEEDS:
        raise ValueError('Only20approved fixed-parity rollout seeds')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def seed_trial_broad(seed):
    approved = broad_policy_seeds()
    if type(seed) is not int or seed not in approved:
        raise ValueError('Only2approved broad policy seeds')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _finite(value):
    if isinstance(value, dict): return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value, (float, np.floating)) else True


def _summarize(rows, planned_seeds, required, scope):
    seeds = [r.get('seed') for r in rows]
    complete = (_finite(rows) and seeds == list(planned_seeds)
                and all(type(r.get('task_success')) is bool and r.get('outcome') ==
                        ('success' if r['task_success'] else 'timeout') for r in rows))
    successes = sum(r.get('task_success') is True for r in rows)
    planned = len(planned_seeds)
    return dict(planned_trials=planned, finished_trials=len(rows), successes=successes,
                success_rate=successes/len(rows) if rows else None, all_trials_valid=complete,
                capability_passed=complete and successes >= required, required_successes=required,
                scope=scope)


def summarize_broad(rows):
    trials = evaluation_plan_broad()
    return _summarize(rows, tuple(t['seed'] for t in trials), BROAD_GATE,
                      'broad_holdout_36_reserved_layouts_x_2_policy_seeds_spatial_generalization')


def summarize_fixed_parity(rows):
    return _summarize(rows, FIXED_PARITY_SEEDS, FIXED_PARITY_GATE,
                      'sampling_reliability_one_selected_geometry_not_spatial_generalization')


def require_complete_trials_broad(rows):
    if not summarize_broad(rows)['all_trials_valid']:
        raise ValueError('All72approved finite broad trials must finish')


def require_complete_trials_fixed_parity(rows):
    if not summarize_fixed_parity(rows)['all_trials_valid']:
        raise ValueError('All20approved finite fixed-parity trials must finish')


def _load_snapshots(binding):
    """Verify the broad-workspace holdout source independently of the local-varied binding."""
    if json.loads((BROAD_PARENT/'seal.json').read_text()) != {'fingerprint': BROAD_SEAL}:
        raise ValueError('Broad-workspace holdout seal changed')
    return json.loads((BROAD_PARENT/'snapshots.json').read_text())


def _load_bank_manifest():
    return json.loads((BROAD_PARENT/'bank.json').read_text())


def _materialize_broad_scene(mode_dir, layout_id, plan, bank):
    """Re-derive a reserved evaluation scene and prove it against the sealed bank.

    Broad-workspace stores no evaluation scene files, so the scene is rebuilt from the
    frozen plan and only written once every per-array SHA256 and the scene hash match
    bank.json. A drifted derivation fails here rather than silently rolling out.
    """
    if layout_id not in broad_layout_ids():
        raise ValueError('Layout not in the reserved evaluation set')
    if layout_id not in bank or layout_id not in plan['evaluation_holdout']:
        raise ValueError('Scene identity missing from frozen plan/bank')
    scene_path = mode_dir/f'{layout_id}.npz'
    if scene_path.exists(): raise FileExistsError('Evaluation scene already materialized')

    expert, base_source = _base()
    layout = {x['layout_id']: x for x in plan['layouts']}[layout_id]
    holdout = plan['evaluation_holdout'][layout_id]
    arrays = derive_scene(expert.model, base_source, layout,
                          dict(home_pose_jitter_deg=holdout['home_pose_jitter_deg']))

    frozen = bank[layout_id]
    derived = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
    if derived != frozen['arrays_sha256']:
        raise ValueError(f'Re-derived scene arrays differ from the sealed bank for {layout_id}')
    if scene_hash(arrays) != frozen['scene_hash']:
        raise ValueError(f'Re-derived scene hash differs from the sealed bank for {layout_id}')

    with scene_path.open('xb') as stream:
        np.savez(stream, initial_state=arrays['initial_state'], qpos=arrays['qpos'])
    return c.sha256(scene_path)


def broad_entry(layout_id, bank, scene_sha256):
    """Scene entry for the broad protocol; hash-bound to the sealed bank.json."""
    if layout_id not in broad_layout_ids():
        raise ValueError('Layout not in reserved evaluation set')
    if layout_id not in bank:
        raise ValueError('Scene artifact missing from frozen manifest')
    entry = bank[layout_id]
    return dict(layout_id=layout_id, source_episode_index=7, pick='red', place='green',
                colors=['red', 'green'], file=f'{layout_id}.npz', sha256=scene_sha256,
                scene_hash=entry['scene_hash'], cube_xy_m=entry['layout']['cube_xy_m'])


def fixed_parity_entry(binding, reset_sha256=None):
    """Source-reset entry: same initial state and scene hash as the fixed-layout baseline."""
    if c.sha256(c.PARENT/'source.npz') != binding['source_reset_sha256']:
        raise ValueError('Broad-workspace source reset changed')
    return dict(episode_index=7, source_episode_index=7, pick='red', place='green',
                colors=['red', 'green'], file='reset.npz',
                sha256=reset_sha256 if reset_sha256 is not None else binding['source_reset_sha256'],
                scene_hash=SOURCE_SCENE_SHA256)


def _materialize_fixed_parity_reset(mode_dir, binding):
    """Persist single-timestep reset extracted from source.npz for restore_scene consumption."""
    if c.sha256(c.PARENT/'source.npz') != binding['source_reset_sha256']:
        raise ValueError('Broad-workspace source reset changed')
    reset_path = mode_dir/'reset.npz'
    if reset_path.exists(): raise FileExistsError('Fixed-parity reset already materialized')
    with np.load(c.PARENT/'source.npz', allow_pickle=False) as source:
        initial_state = source['initial_state'].copy()
        qpos_full = source['qpos']
        if qpos_full.ndim != 2 or qpos_full.shape[1] != 37:
            raise ValueError('Unexpected source.npz qpos layout')
        qpos0 = qpos_full[0].copy()
    with reset_path.open('xb') as stream:
        np.savez(stream, initial_state=initial_state, qpos=qpos0)
    return c.sha256(reset_path)


_BASE = None


def _base():
    """Pilot base expert/reset, loaded once and shared by derivation and restore."""
    global _BASE
    if _BASE is None:
        expert = BaseExpert()
        _BASE = (expert, load_reset(expert))
    return _BASE


def restore_broad_scene(model, bank, entry, base_source):
    """`evaluate_pilot.restore_scene` with the envelope-F7 geometry gate.

    That helper validates against `pilot_integrity.validate_scene_geometry`, whose
    workspace predates F7 and caps |y| at 0.40 m; reserved evaluation layouts reach
    0.45 m. Every path, byte, state and scene-hash proof is preserved here — only the
    geometry gate is swapped for the one the collector itself used.
    """
    path = (bank/entry['file']).resolve()
    if path.parent != bank.resolve() or c.sha256(path) != entry['sha256']:
        raise ValueError('Scene artifact path/hash mismatch')
    data = mujoco.MjData(model)
    with np.load(path, allow_pickle=False) as state:
        restore_state(model, data, state['initial_state'])
        np.testing.assert_array_equal(capture_state(model, data), state['initial_state'])
        np.testing.assert_array_equal(data.qpos, state['qpos'])
    if scene_digest(data.qpos) != entry['scene_hash']:
        raise ValueError('Scene qpos hash mismatch')
    _validate_broad_geometry(model, data, base_source)
    return data


def stage_metrics(trace):
    """Persisted descriptive stages only; never supplied to policy inference."""
    actions = np.asarray(trace['applied_action']).reshape(-1, 8)
    contact = np.asarray(trace['bilateral_target_contact'], dtype=bool)
    lift = np.asarray(trace['pick_lift_m'], dtype=float)
    released = np.asarray(trace['released_stack'], dtype=bool)
    if any(len(x) != len(actions) for x in (contact, lift, released)) or not np.isfinite(actions).all() or not np.isfinite(lift).all():
        raise ValueError('Invalid saved action/contact stages')
    closed = actions[:, 7] <= 127

    def first(mask):
        ticks = np.flatnonzero(mask)
        return int(ticks[0]) if ticks.size else None

    def sustained(mask, count):
        streak = 0
        for i, value in enumerate(mask):
            streak = streak+1 if value else 0
            if streak >= count: return i
        return None

    closure = first(closed)
    after = np.arange(len(actions)) > closure if closure is not None else np.zeros(len(actions), dtype=bool)
    return dict(first_closure_tick=closure, sustained_closure_30_tick=sustained(closed, 30),
                first_bilateral_contact_tick=first(contact), sustained_bilateral_5_tick=sustained(contact, 5),
                first_contact_lift_gt_5cm_tick=first(contact & (lift > .05)),
                first_release_after_closure_tick=first(after & ~closed), stable_stack_30_tick=sustained(released, 30),
                maximum_lift_m=float(lift.max()) if len(lift) else 0., bilateral_contact_ticks=int(contact.sum()), tick_origin=0)


def offline_probes(policy, pre, post, ds, device='cuda', output=None):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = policy.training
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    rows, predictions, targets = [], [], []
    try:
        policy.eval()
        with torch.random.fork_rng(devices=devices):
            for episode in c.TRAIN+c.VAL:
                for frame in (60, 107, 347):
                    index = episode*528+frame
                    sample = c.policy_row(ds[index], training=False)
                    obs = c.observation({key: sample[key] for key in (*c.OBS, 'task')})
                    target = sample['action'].detach().cpu().clone()
                    batch = c.p.process_row(pre, obs)
                    if batch.get('action') is not None:
                        raise ValueError('Inference processor introduced a target action')
                    batch.pop('action', None)
                    random.seed(c.SEED+index); np.random.seed(c.SEED+index); torch.manual_seed(c.SEED+index)
                    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
                    with torch.inference_mode(), autocast:
                        prediction = post(policy.predict_action_chunk(batch)).detach().cpu()
                    if prediction.shape != (1, 30, 8) or not torch.isfinite(prediction).all():
                        raise ValueError('Finite physical30x8 prediction required')
                    error = (prediction[0]-target).abs()
                    rows.append(dict(episode=episode, frame=frame, row=index,
                                     split='train' if episode in c.TRAIN else 'validation',
                                     arm_mae_rad=float(error[:, :7].mean()), first10_arm_mae_rad=float(error[:10, :7].mean()),
                                     gripper_binary_accuracy=float(((prediction[0, :, 7] > 127) == (target[:, 7] > 127)).float().mean())))
                    predictions.append(prediction[0].numpy()); targets.append(target.numpy())
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); policy.train(training)
    result = dict(checkpoint_update=5000, observation_only=True, validation_used_for_training=False,
                  validation_is_spatial_holdout=False, rows=rows)
    for split in ('train', 'validation'):
        selected = [r for r in rows if r['split'] == split]
        result[split] = {'probes': len(selected), **{key: float(np.mean([r[key] for r in selected]))
                        for key in ('arm_mae_rad', 'first10_arm_mae_rad', 'gripper_binary_accuracy')}}
    if output is not None:
        with (output/'offline_probe_arrays.npz').open('xb') as stream:
            np.savez_compressed(stream, predictions=np.asarray(predictions), targets=np.asarray(targets),
                                rows=np.asarray([r['row'] for r in rows]))
        write_json(output/'offline_probes.json', result)
    return result


class CountedPolicy(XVLAPolicy):
    """Native10-action queue; subclass counting, never runtime method replacement."""
    chunk_calls = 0

    def _get_action_chunk(self, *args, **kwargs):
        self.chunk_calls += 1
        return super()._get_action_chunk(*args, **kwargs)


def load_counted(checkpoint):
    """Strict FP32 loader shared with training/preflight; promote to cuda after load."""
    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = 'cpu'
    policy = c.p.load_weights(CountedPolicy(config), checkpoint/'model.safetensors').to('cuda')
    policy.config.device = 'cuda'
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _rollout(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed, apply_seed):
    """Physics rollout body; caller enforces identity and seed policy."""
    if (policy.config.chunk_size, policy.config.n_action_steps) != (30, 10):
        raise ValueError('Expected chunk30/actionsteps10')
    data, scratch = restore_broad_scene(model, bank, entry, _base()[1]), mujoco.MjData(model)
    policy.reset()
    apply_seed(seed)
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
    if instruction != TASK: raise ValueError('Approved single instruction required')
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
                processed = pre(c.observation(frame))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    action = post(policy.select_action(processed)).squeeze(0).cpu().numpy()
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
              'sim_seconds': ticks/30, 'actual_chunk_predictions': calls, 'chunk_size': 30, 'n_action_steps': 10,
              'task_success': success, 'unassisted_success': success and interventions == 0,
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


def run_fixed_parity_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed):
    if not 1 <= ordinal <= 20 or seed != 20260917+ordinal or entry['episode_index'] != 7:
        raise ValueError('Explicit approved fixed-parity seed/ordinal/reset identity required')
    return _rollout(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed,
                    seed_trial_fixed_parity)


def run_broad_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed):
    approved_layouts = broad_layout_ids()
    approved_seeds = broad_policy_seeds()
    if (not 1 <= ordinal <= 72 or seed not in approved_seeds
            or entry.get('layout_id') not in approved_layouts):
        raise ValueError('Explicit approved broad layout/seed/ordinal identity required')
    return _rollout(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed,
                    seed_trial_broad)


def verify_final(output, binding):
    report = json.loads((output/'result.json').read_text())
    c.require_fields(report, {**c.recipe(False), 'status': 'complete', 'completed_updates': 5000,
                     'selected_checkpoint_update': 5000, 'final_data_binding_passed': True, 'binding': binding,
                     'training_rows': c.TRAINING_ROWS, 'complete_training_chunks': c.COMPLETE_STARTS,
                     'scratch_used_for_initialization': False},
                     'Complete final-only5000 report required')
    if (report.get('cpu_admission_sha256') != c.sha256(c.CPU_ADMISSION)
            or report.get('training_stats_sha256') != c.sha256(output/'training_stats.json')):
        raise ValueError('Final CPU/statistics binding changed')
    c.verify_checkpoints(report, output, False)
    stats = json.loads((output/'training_stats.json').read_text())
    if stats != json.loads(c.CPU_ADMISSION.read_text())['training_stats']: raise ValueError('Final stats differ from CPU')
    scratch_result = c.SCRATCH/'result.json'
    if scratch_result.is_file():
        if report.get('scratch_result_sha256') != c.sha256(scratch_result):
            raise ValueError('Final scratch binding changed')
        c.validate_scratch(json.loads(scratch_result.read_text()), binding, c.SCRATCH, stats, c.sha256(c.CPU_ADMISSION))
    checkpoint = output/'checkpoints'/c.checkpoint_directory(5000)
    config = json.loads((checkpoint/'config.json').read_text())
    if (config != json.loads((output/'initial_config/config.json').read_text())
            or (config['type'], config['chunk_size'], config['n_action_steps'], config['n_obs_steps'], config['dtype']) != ('xvla', 30, 10, 1, 'float32')
            or {k: tuple(v['shape']) for k, v in config['input_features'].items()} != c.OBS
            or {k: v['shape'] for k, v in config['output_features'].items()} != {'action': [8]}):
        raise ValueError('Final X-VLA config/schema changed')
    return checkpoint


EVAL_DIRNAME = 'evaluation_broad_holdout'  # never the original 'evaluation/' from xvla_local_varied_eval


def training_bytes(output):
    return {str(path.relative_to(output)): c.sha256(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.relative_to(output).parts[0] not in ('evaluation', EVAL_DIRNAME)}


def _run_broad_holdout(evaluation_dir, output, policy, pre, post, model, manifest,
                       snapshots, bank_manifest, binding):
    mode_dir = evaluation_dir/'broad_holdout'
    mode_dir.mkdir(exist_ok=False)
    from experiments.xvla_broad_workspace.runner import require_disk
    plan = _plan()
    trials = evaluation_plan_broad()
    rows = []
    progress = dict(mode='broad_holdout', trials=trials, **summarize_broad([]))
    write_json(mode_dir/'result.json', progress)
    materialized = {}
    for trial in trials:
        require_disk(output, 2)
        layout_id = trial['layout_id']
        # Each layout is derived and proven once, then shared by its two seeds.
        if layout_id not in materialized:
            materialized[layout_id] = _materialize_broad_scene(mode_dir, layout_id, plan, bank_manifest)
        entry = broad_entry(layout_id, bank_manifest, materialized[layout_id])
        folder = mode_dir/f"trial_{trial['ordinal']:03d}_layout_{layout_id}_seed_{trial['seed']}"
        folder.mkdir(exist_ok=False)
        row = run_broad_trial(policy, pre, post, model, manifest, entry, mode_dir, folder,
                              trial['ordinal'], trial['seed'])
        if row['seed'] != trial['seed'] or row['layout_id'] != layout_id or row['outcome'] == 'nonfinite_action':
            raise ValueError('Seed/layout/nonfinite failure; no retry')
        rows.append(row); progress.update(episodes=rows, **summarize_broad(rows))
        write_json(mode_dir/'result.json', progress)
        print('EVAL_BROAD', json.dumps({k: progress[k] for k in ('finished_trials', 'successes', 'capability_passed')}), flush=True)
    require_complete_trials_broad(rows)
    return dict(trials=trials, episodes=rows, **summarize_broad(rows))


def _run_fixed_parity(evaluation_dir, output, policy, pre, post, model, manifest, binding):
    mode_dir = evaluation_dir/'fixed_parity'
    mode_dir.mkdir(exist_ok=False)
    from experiments.xvla_broad_workspace.runner import require_disk
    reset_sha256 = _materialize_fixed_parity_reset(mode_dir, binding)
    entry = fixed_parity_entry(binding, reset_sha256=reset_sha256)
    restore_broad_scene(model, mode_dir, entry, _base()[1])  # Full state restore only, never resettle.
    trials = evaluation_plan_fixed_parity()
    rows = []
    progress = dict(mode='fixed_parity', trials=trials, **summarize_fixed_parity([]))
    write_json(mode_dir/'result.json', progress)
    for trial in trials:
        require_disk(output, 2)
        folder = mode_dir/f"trial_{trial['ordinal']:02d}_seed_{trial['seed']}"
        folder.mkdir(exist_ok=False)
        row = run_fixed_parity_trial(policy, pre, post, model, manifest, entry, mode_dir, folder,
                                     trial['ordinal'], trial['seed'])
        if row['seed'] != trial['seed'] or row['outcome'] == 'nonfinite_action':
            raise ValueError('Seed/nonfinite action failure; no retry')
        rows.append(row); progress.update(episodes=rows, **summarize_fixed_parity(rows))
        write_json(mode_dir/'result.json', progress)
        print('EVAL_FIXED_PARITY', json.dumps({k: progress[k] for k in ('finished_trials', 'successes', 'capability_passed')}), flush=True)
    require_complete_trials_fixed_parity(rows)
    return dict(trials=trials, episodes=rows, **summarize_fixed_parity(rows))


def run(output, mode='broad_holdout'):
    from lerobot.policies.factory import make_pre_post_processors
    from experiments.xvla_broad_workspace.runner import failure
    if mode not in MODES: raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    directory = output/EVAL_DIRNAME
    resume_fixed_parity_only = False
    if not resume_fixed_parity_only:
        directory.mkdir(exist_ok=False)  # No overwrite / resume / retry.
    report = dict(status='preparing', stage='evaluation_admission', mode=mode, max_control_ticks=1200,
                  max_sim_seconds=40, strict_score_ticks=30, final_only=True, retries=False,
                  resume=resume_fixed_parity_only)
    before = training_bytes(output)
    try:
        ds, starts, stats, binding = c.prepare_data()
        checkpoint = verify_final(output, binding)
        manifest = json.loads((c.PARENT/'configuration.json').read_text())
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        if restore_timing(model, manifest)['substeps_per_control'] != 20: raise ValueError('Physics timing changed')
        snapshots = _load_snapshots(binding) if mode in ('broad_holdout', 'both') else None
        bank_manifest = _load_bank_manifest() if mode in ('broad_holdout', 'both') else None
        torch.use_deterministic_algorithms(True)
        policy = load_counted(checkpoint)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        report.update(status='running', stage='offline_probes', binding=binding, training_files=before,
                      privileged_policy_inputs=False)
        write_json(directory/'result.json', report)
        if resume_fixed_parity_only:
            report['offline_probes'] = json.loads((directory/'offline_probes.json').read_text())
            report['broad_holdout'] = json.loads((directory/'broad_holdout'/'result.json').read_text())
        else:
            report['offline_probes'] = offline_probes(policy, pre, post, ds, output=directory)
        if c.admission() != binding: raise ValueError('Probe phase binding changed')
        verify_final(output, binding)
        if mode in ('broad_holdout', 'both'):
            report['stage'] = 'rollouts_broad_holdout'; write_json(directory/'result.json', report)
            report['broad_holdout'] = _run_broad_holdout(directory, output, policy, pre, post, model,
                                                         manifest, snapshots, bank_manifest, binding)
        if mode in ('fixed_parity', 'both'):
            report['stage'] = 'rollouts_fixed_parity'; write_json(directory/'result.json', report)
            report['fixed_parity'] = _run_fixed_parity(directory, output, policy, pre, post, model,
                                                       manifest, binding)
        after = c.admission(); verify_final(output, after)
        if after != binding: raise ValueError('Final evaluation binding changed')
        report.update(status='complete', stage='complete', training_bytes_unchanged=True,
                      final_data_binding_passed=True, no_extension=True)
    except BaseException as exc:
        failure(directory, report, exc); raise
    finally:
        if training_bytes(output) != before:
            exc = ValueError('Training bytes changed during evaluation')
            failure(directory, report, exc); raise exc
    write_json(directory/'result.json', report)


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=list(MODES), default='broad_holdout')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(c.OUTPUT, args.mode)


if __name__ == '__main__':
    main()
