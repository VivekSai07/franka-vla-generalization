"""B1 language-grounding evaluation: does the trained policy follow the instruction?

Forked from `experiments.xvla_broad_workspace_long_eval`, protocol shape held identical (same
rollout body, gates, scene-restore/geometry proof, checkpoint pinning), narrowed to what B1
needs and dropping what it doesn't:

* Single protocol `language_holdout` on the 80 reserved evaluation layouts, replacing
  `broad_holdout` + `fixed_parity`. `fixed_parity` existed to compare against an old single-task
  fixed-layout baseline that has no B1 analogue.
* Each reserved layout supports all four cells (pick x arrangement): `pick`/`place`/`instruction`
  are a PER-TRIAL choice, not baked into the layout, because language grounding is exactly the
  question of whether the same physical scene is handled differently under a different
  instruction. `broad_entry`'s hardcoded `pick='red', place='green'` becomes `language_entry`,
  parameterized by cell.
* The evaluation layouts are split into two disjoint halves: `SCREEN_LAYOUTS` (first 40) select
  the checkpoint by capability alone (cell C1 only, cheap); `REPORT_LAYOUTS` (last 40) measure
  the grounding contrast on the winner across all four cells. Reusing the screening layouts for
  the reported contrast would be cherry-picking (checkpoint selection and the resulting metric
  must come from disjoint data) -- decided before any trial ran, per the B1 design spec.
* Every `c.observation(...)` call is wrapped in `c.approved_instruction(...)`, reusing the same
  context manager training uses to widen `xvla_single_task.contract`'s single hardcoded `TASK`
  (FAILURES #29): the shared module remains bound by four contracts and is never edited.
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
from validation.evaluate_pilot import advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF, SCENE_PATH
from experiments.fixed_layout.source import Expert as BaseExpert, load_reset
from experiments.b1_language_data import plan as bp
from experiments.b1_language_data import scenes as bs
from experiments.b1_language_data.scenes import _validate_broad_geometry
from experiments.xvla_b1_language import contract as c

CANDIDATES = (16700, 25000, 33400)  # the three viable checkpoints; 1 and 8000 are not evaluated here
SCREEN_CELL = 'C1'  # red-pick, familiar -- the canonical cell, used only to rank raw capability
SCREEN_TRIALS_PER_CANDIDATE = 40
REPORT_TRIALS_PER_CELL = 40
MODES = ('smoke', 'screen', 'report')


def _plan():
    return bp.make_plan()


def screen_layout_ids():
    return tuple(_plan()['evaluation_layout_ids'][:40])


def report_layout_ids():
    return tuple(_plan()['evaluation_layout_ids'][40:])


def evaluation_plan(mode):
    """`smoke`: 4 trials, one per cell on one layout -- proves the rollout body handles every
    pick/place/instruction combination before hours are spent on `screen`/`report`. `screen`: 40
    trials, one cell, one candidate at a time (candidate is a CLI arg, not roster). `report`: 160
    trials, all four cells x all 40 report layouts."""
    if mode == 'smoke':
        lid = screen_layout_ids()[0]
        return [dict(ordinal=i+1, layout_id=lid, cell=cell) for i, cell in enumerate(bp.CELL_IDS)]
    if mode == 'screen':
        return [dict(ordinal=i+1, layout_id=lid, cell=SCREEN_CELL)
                for i, lid in enumerate(screen_layout_ids())]
    if mode == 'report':
        trials, ordinal = [], 0
        for lid in report_layout_ids():
            for cell in bp.CELL_IDS:
                ordinal += 1
                trials.append(dict(ordinal=ordinal, layout_id=lid, cell=cell))
        return trials
    raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')


def seed_trial(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _finite(value):
    if isinstance(value, dict): return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value, (float, np.floating)) else True


def summarize(rows, planned, required=None):
    ordinals = [r.get('ordinal') for r in rows]
    complete = (_finite(rows) and ordinals == [t['ordinal'] for t in planned]
                and all(type(r.get('task_success')) is bool and r.get('outcome') ==
                        ('success' if r['task_success'] else 'timeout') for r in rows))
    successes = sum(r.get('task_success') is True for r in rows)
    result = dict(planned_trials=len(planned), finished_trials=len(rows), successes=successes,
                 success_rate=successes/len(rows) if rows else None, all_trials_valid=complete)
    if required is not None:
        result.update(capability_passed=complete and successes >= required, required_successes=required)
    return result


def summarize_by_cell(rows):
    return {cell: summarize([r for r in rows if r['cell'] == cell],
                            [t for t in evaluation_plan('report') if t['cell'] == cell])
            for cell in bp.CELL_IDS}


def require_complete_trials(rows, planned):
    if not summarize(rows, planned)['all_trials_valid']:
        raise ValueError(f'All {len(planned)} approved finite trials must finish')


def _load_bank_manifest():
    return json.loads((c.PARENT/'bank.json').read_text())


_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        expert = BaseExpert(); _BASE = (expert, load_reset(expert))
    return _BASE


def _materialize_scene(mode_dir, layout_id, cell, plan, bank_manifest_dict):
    """Re-derive one (layout, arrangement) scene and prove it against the sealed bank.

    Bank.json records hashes only; B1 persists no evaluation scene files (FAILURES #29's sealed
    parent), so every scene is rebuilt here from the frozen plan and only written once its arrays
    and scene hash match the manifest exactly.
    """
    arrangement = bp.CELLS[cell]['arrangement']
    key = bs.eval_bank_key(layout_id, arrangement)
    if key not in bank_manifest_dict:
        raise ValueError('Scene identity missing from frozen bank manifest')
    scene_path = mode_dir/f"{layout_id}_{arrangement}.npz"
    if scene_path.exists():
        return c.sha256(scene_path), bank_manifest_dict[key]

    expert, base_source = _base()
    layouts_by_id = {x['layout_id']: x for x in plan['layouts']}
    holdout = plan['evaluation_holdout'][layout_id]
    placed = dict(layouts_by_id[layout_id], cube_xy_m=list(holdout['arrangements'][arrangement]))
    arrays = bs.derive_scene(expert.model, base_source, placed,
                             dict(home_pose_jitter_deg=holdout['home_pose_jitter_deg']))

    frozen = bank_manifest_dict[key]
    derived = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
    if derived != frozen['arrays_sha256']:
        raise ValueError(f'Re-derived scene arrays differ from the sealed bank for {key}')
    if bs.scene_hash(arrays) != frozen['scene_hash']:
        raise ValueError(f'Re-derived scene hash differs from the sealed bank for {key}')

    with scene_path.open('xb') as stream:
        np.savez(stream, initial_state=arrays['initial_state'], qpos=arrays['qpos'])
    return c.sha256(scene_path), frozen


def language_entry(layout_id, cell, scene_sha256, manifest_entry):
    cells = bp.CELLS[cell]
    return dict(layout_id=layout_id, cell=cell, arrangement=cells['arrangement'],
                pick=cells['pick'], place=cells['place'],
                instruction=bp.instruction_for(cells['pick'], cells['place']),
                colors=['red', 'green'], file=f"{layout_id}_{cells['arrangement']}.npz",
                sha256=scene_sha256, scene_hash=manifest_entry['scene_hash'])


def restore_language_scene(model, mode_dir, entry, base_source):
    path = (mode_dir/entry['file']).resolve()
    if path.parent != mode_dir.resolve() or c.sha256(path) != entry['sha256']:
        raise ValueError('Scene artifact path/hash mismatch')
    data = mujoco.MjData(model)
    with np.load(path, allow_pickle=False) as state:
        restore_state(model, data, state['initial_state'])
        np.testing.assert_array_equal(capture_state(model, data), state['initial_state'])
        np.testing.assert_array_equal(data.qpos, state['qpos'])
    from data_collection.pilot_integrity import scene_digest
    if scene_digest(data.qpos) != entry['scene_hash']:
        raise ValueError('Scene qpos hash mismatch')
    _validate_broad_geometry(model, data, base_source)
    return data


def stage_metrics(trace):
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


def offline_probes(policy, pre, post, ds, device='cuda', output=None, update=None):
    """Same shape as the training-side probe, wrapped per-episode in `approved_instruction`."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = policy.training
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    rows, predictions, targets = [], [], []
    planned = c.planned_instructions()
    try:
        policy.eval()
        with torch.random.fork_rng(devices=devices):
            for episode in c.TRAIN+c.VAL:
                instruction = planned[episode]
                for frame in (60, 107, 347):
                    index = episode*528+frame
                    sample = c.policy_row(ds[index], training=False)
                    with c.approved_instruction(instruction):
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
                        raise ValueError('Finite physical 30x8 prediction required')
                    error = (prediction[0]-target).abs()
                    rows.append(dict(episode=episode, frame=frame, row=index,
                                     split='train' if episode in c.TRAIN else 'validation',
                                     instruction=instruction,
                                     arm_mae_rad=float(error[:, :7].mean()), first10_arm_mae_rad=float(error[:10, :7].mean()),
                                     gripper_binary_accuracy=float(((prediction[0, :, 7] > 127) == (target[:, 7] > 127)).float().mean())))
                    predictions.append(prediction[0].numpy()); targets.append(target.numpy())
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); policy.train(training)
    result = dict(checkpoint_update=update, observation_only=True, validation_used_for_training=False,
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
    chunk_calls = 0

    def _get_action_chunk(self, *args, **kwargs):
        self.chunk_calls += 1
        return super()._get_action_chunk(*args, **kwargs)


def load_counted(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = 'cpu'
    policy = c.p.load_weights(CountedPolicy(config), checkpoint/'model.safetensors').to('cuda')
    policy.config.device = 'cuda'
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _rollout(policy, pre, post, model, manifest, entry, mode_dir, output, ordinal, seed):
    if (policy.config.chunk_size, policy.config.n_action_steps) != (30, 10):
        raise ValueError('Expected chunk30/actionsteps10')
    data, scratch = restore_language_scene(model, mode_dir, entry, _base()[1]), mujoco.MjData(model)
    policy.reset()
    seed_trial(seed)
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = entry['instruction']
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
                with c.approved_instruction(instruction):
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
    result = {**entry, 'ordinal': ordinal, 'seed': seed, 'rollout_seed': seed,
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


def run_trial(policy, pre, post, model, manifest, entry, mode_dir, output, ordinal, seed, layout_ids):
    if entry['layout_id'] not in layout_ids or entry['cell'] not in bp.CELL_IDS:
        raise ValueError('Explicit approved layout/cell identity required')
    return _rollout(policy, pre, post, model, manifest, entry, mode_dir, output, ordinal, seed)


def evaluation_dirname(mode, candidate):
    return f'evaluation_{mode}_step_{candidate:06d}'


def verify_checkpoint(output, candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f'Unapproved evaluation candidate; expected one of {CANDIDATES}')
    report = json.loads((output/'result.json').read_text())
    c.verify_checkpoints(report, output, False)
    checkpoint = output/'checkpoints'/c.checkpoint_directory(candidate)
    config = json.loads((checkpoint/'config.json').read_text())
    if (config['type'], config['chunk_size'], config['n_action_steps'], config['n_obs_steps'], config['dtype']) != ('xvla', 30, 10, 1, 'float32'):
        raise ValueError('Checkpoint config/schema changed')
    return checkpoint


def training_bytes(output):
    return {str(path.relative_to(output)): c.sha256(path) for path in sorted(output.rglob('*'))
            if path.is_file() and not path.relative_to(output).parts[0].startswith('evaluation')}


def _run_mode(mode, mode_dir, output, policy, pre, post, model, manifest, bank_manifest_dict, layout_ids):
    from experiments.xvla_b1_language.runner import require_disk
    plan = _plan()
    trials = evaluation_plan(mode)
    rows = []
    for trial in trials:
        require_disk(output, 2)
        layout_id, cell = trial['layout_id'], trial['cell']
        scene_sha256, manifest_entry = _materialize_scene(mode_dir, layout_id, cell, plan, bank_manifest_dict)
        entry = language_entry(layout_id, cell, scene_sha256, manifest_entry)
        folder = mode_dir/f"trial_{trial['ordinal']:03d}_layout_{layout_id}_cell_{cell}"
        folder.mkdir(exist_ok=False)
        seed = 20260925*10 + trial['ordinal']
        row = run_trial(policy, pre, post, model, manifest, entry, mode_dir, folder,
                        trial['ordinal'], seed, layout_ids)
        if row['outcome'] == 'nonfinite_action':
            raise ValueError('Nonfinite-action failure; no retry')
        rows.append(row)
        write_json(mode_dir/'result.json', dict(mode=mode_dir.name, trials=trials, episodes=rows,
                                                **summarize(rows, trials)))
        print('EVAL', mode_dir.name, json.dumps({k: rows[-1][k] for k in
              ('layout_id', 'cell', 'task_success')}), flush=True)
    require_complete_trials(rows, trials)
    return rows


def run(output, mode, candidate):
    from lerobot.policies.factory import make_pre_post_processors
    from experiments.xvla_b1_language.runner import failure
    if mode not in MODES:
        raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    layout_ids = {'smoke': screen_layout_ids(), 'screen': screen_layout_ids(),
                  'report': report_layout_ids()}[mode]
    directory = output/evaluation_dirname(mode, candidate)
    directory.mkdir(exist_ok=False)
    report = dict(status='preparing', stage='evaluation_admission', mode=mode, candidate=candidate,
                 max_control_ticks=1200, max_sim_seconds=40, strict_score_ticks=30, retries=False)
    before = training_bytes(output)
    try:
        ds, starts, stats, binding = c.prepare_data()
        checkpoint = verify_checkpoint(output, candidate)
        manifest = json.loads((c.PARENT/'configuration.json').read_text())
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        if restore_timing(model, manifest)['substeps_per_control'] != 20:
            raise ValueError('Physics timing changed')
        bank_manifest_dict = _load_bank_manifest()
        torch.use_deterministic_algorithms(True)
        policy = load_counted(checkpoint)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        report.update(status='running', stage='offline_probes', binding=binding, training_files=before,
                      privileged_policy_inputs=False)
        write_json(directory/'result.json', report)
        report['offline_probes'] = offline_probes(policy, pre, post, ds, output=directory, update=candidate)
        if c.admission() != binding:
            raise ValueError('Source/statistics binding changed during offline probes')
        report['stage'] = 'rollouts'; write_json(directory/'result.json', report)
        rows = _run_mode(mode, directory, output, policy, pre, post, model, manifest, bank_manifest_dict, layout_ids)
        trials = evaluation_plan(mode)
        report[mode] = dict(trials=trials, episodes=rows, **summarize(rows, trials),
                            **({'by_cell': summarize_by_cell(rows)} if mode == 'report' else {}))
        report.update(status='complete', stage='complete', training_bytes_unchanged=True, no_extension=True)
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
    parser.add_argument('--mode', choices=list(MODES), required=True)
    parser.add_argument('--checkpoint', type=int, choices=CANDIDATES, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(c.OUTPUT, args.mode, args.checkpoint)


if __name__ == '__main__':
    main()
