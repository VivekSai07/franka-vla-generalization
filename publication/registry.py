"""Read-only, selected-field registry of exactly the two completed baselines."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from publication.package_baselines import MODELS, RELEASE_RELATIVE, audit_json, sha256

TIMES = {
    'smolvla': ('2026-09-18T17:41:15Z', '2026-09-18T19:32:07Z'),
    'xvla': ('2026-09-18T15:14:07Z', '2026-09-18T16:56:38Z'),
}
CONFIG_FIELDS = (
    'type', 'n_obs_steps', 'chunk_size', 'n_action_steps', 'max_state_dim',
    'max_action_dim', 'num_steps', 'num_denoising_steps', 'action_mode',
    'freeze_vision_encoder', 'freeze_language_encoder', 'train_expert_only',
    'train_state_proj', 'train_policy_transformer', 'train_soft_prompts',
    'scheduler_warmup_steps', 'scheduler_decay_steps', 'scheduler_decay_lr',
)
RECIPE_FIELDS = ('seed', 'microbatch', 'accumulation', 'effective_batch', 'chunk_size',
                 'n_action_steps', 'completed_updates', 'training_rows', 'complete_training_chunks')
OPTIMIZER_FIELDS = ('peak_lr', 'vlm_scale', 'soft_prompt_scale', 'weight_decay',
                    'vlm_weight_decay', 'betas', 'eps', 'clip')
SCHEDULE_FIELDS = ('warmup', 'total', 'peak', 'floor', 'floor_fraction')
MEMORY_FIELDS = ('allocated', 'reserved', 'peak_allocated', 'peak_reserved', 'device_free', 'device_total')
STAGE_FIELDS = ('first_closure_tick', 'sustained_closure_30_tick',
                'first_bilateral_contact_tick', 'sustained_bilateral_5_tick',
                'first_contact_lift_gt_5cm_tick', 'first_release_after_closure_tick',
                'stable_stack_30_tick', 'maximum_lift_m', 'bilateral_contact_ticks')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite or nonnumeric historical value')
    return value


def numeric_tree(value):
    if isinstance(value, list):
        return [numeric_tree(v) for v in value]
    return number(value)


def metric_record(raw):
    record = {'update': number(raw['update'])}
    for key in ('loss', 'grad_norm', 'seconds', 'sampled_chunks', 'epoch',
                'completed_microbatches', 'peak_cuda_allocated_bytes', 'lr'):
        if key in raw:
            record['train/' + key] = number(raw[key])
    for key in ('vlm', 'soft_prompts', 'other'):
        if key in raw.get('learning_rates', {}):
            record['train/lr/' + key] = number(raw['learning_rates'][key])
    for key in MEMORY_FIELDS:
        if key in raw.get('memory', {}):
            record['train/cuda_memory_bytes/' + key] = number(raw['memory'][key])
    return record


def trial_record(raw):
    result = {key: number(raw[key]) for key in (
        'ordinal', 'seed', 'ticks', 'sim_seconds', 'wall_seconds',
        'joint_clipped_ticks', 'gripper_clipped_ticks', 'success_streak')}
    if type(raw['task_success']) is not bool or raw['outcome'] not in ('success', 'timeout', 'failure'):
        raise ValueError('Unexpected historical outcome')
    result.update(task_success=raw['task_success'], outcome=raw['outcome'])
    for key in ('xy_m', 'z_m'):
        result['final_' + key] = number(raw['final_metrics'][key])
    for key in STAGE_FIELDS:
        value = raw['stages'][key]
        result[key] = None if value is None else number(value)
    return result


def build_row(repo, model):
    repo = Path(repo)
    root = repo / 'outputs/train' / f'{model}_fixed_layout_20260918'
    sources = {}

    def read(path):
        path = Path(path)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Historical input cannot be a symlink')
        data = path.read_bytes()
        sources[path.relative_to(repo).as_posix()] = {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return data

    train = json.loads(read(root / 'result.json'))
    evaluation = json.loads(read(root / 'evaluation/result.json'))
    probes = json.loads(read(root / 'evaluation/offline_probes.json'))
    stats = json.loads(read(root / 'training_stats.json'))
    cfg = json.loads(read(root / 'checkpoints/005000/config.json'))
    metrics = [metric_record(json.loads(line)) for line in read(root / 'metrics.jsonl').splitlines()]
    manifest = json.loads(read(repo / RELEASE_RELATIVE / 'packages' / model / 'release_manifest.json'))
    doc = repo / 'docs' / f'{model.upper()}_FIXED_LAYOUT_20260918.md'
    read(doc)
    if (train['status'] != 'complete' or train['completed_updates'] != 5000
            or evaluation['status'] != 'complete' or evaluation['all_trials_valid'] is not True
            or [r['update'] for r in metrics] != list(range(1, 5001))):
        raise ValueError('Incomplete historical baseline')
    if manifest['source_training_report_sha256'] != sources[(root / 'result.json').relative_to(repo).as_posix()]['sha256']:
        raise ValueError('Release/report mismatch')
    if sha256(root / 'training_stats.json') != train['training_stats_sha256']:
        raise ValueError('Training statistics mismatch')
    trials = [trial_record(raw) for raw in evaluation['episodes']]
    if ([t['ordinal'] for t in trials] != list(range(1, 21))
            or [t['seed'] for t in trials] != list(range(20260918, 20260938))):
        raise ValueError('Trial coverage mismatch')
    successes = sum(t['task_success'] for t in trials)
    if successes != evaluation['successes'] or evaluation['finished_trials'] != 20:
        raise ValueError('Counted/reported outcome mismatch')
    config = {k: number(train[k]) for k in RECIPE_FIELDS}
    config['optimizer'] = {k: numeric_tree(train['optimizer'][k]) for k in OPTIMIZER_FIELDS if k in train['optimizer']}
    config['actual_schedule'] = {k: number(train['schedule'][k]) for k in SCHEDULE_FIELDS if k in train['schedule']}
    selected = {k: cfg[k] for k in CONFIG_FIELDS if k in cfg}
    # Only two known string fields; all other selected config fields are scalars.
    for k, value in selected.items():
        if isinstance(value, str):
            if (k == 'type' and value == model) or (k == 'action_mode' and value in ('joints', 'joint', 'ee6d', 'auto')):
                continue
            raise ValueError('Unreviewed selected config string')
        if not isinstance(value, bool):
            number(value)
    start, finish = TIMES[model]
    total_seconds = (datetime.fromisoformat(finish.replace('Z', '+00:00')) -
                     datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
    config.update(
        model=model, logging_mode='historical_import', original_start=start, original_finish=finish,
        original_time_scope='container training plus evaluation; documented UTC times',
        selected_serialized_config=selected,
        serialized_scheduler_is_not_actual_recipe=True,
        trainable_scope='frozen_vlm_expert_and_state_projection' if model == 'smolvla' else 'all_parameters',
        total_parameters=number(train['total_parameters'] if model == 'smolvla' else train['parameter_count']),
        trainable_parameters=number(train['trainable_parameters'] if model == 'smolvla' else train['parameter_count']),
        train_episodes=20, validation_episodes=4, validation_is_spatial_holdout=False,
        objective='stock_flow_matching' if model == 'smolvla' else 'native_MSE',
        precision='FP32 trainables and Adam; BF16 training autocast',
        missing_telemetry='past GPU utilization and power not recorded',
    )
    if config['actual_schedule']['warmup'] != 500 or config['actual_schedule']['total'] != 5000:
        raise ValueError('Unexpected historical schedule')
    probe_summary = {split: {k: number(probes[split][k]) for k in
                            ('probes', 'arm_mae_rad', 'first10_arm_mae_rad', 'gripper_binary_accuracy')}
                     for split in ('train', 'validation')}
    selected_stats = {feature: {k: numeric_tree(stats[feature][k]) for k in
                                ('mean', 'std', 'min', 'max', 'q01', 'q99') if k in stats[feature]}
                      for feature in ('observation.state', 'action')}
    summary = dict(
        successes=successes, failures=20 - successes, trials=20, success_rate=successes / 20,
        reported_successes=evaluation['successes'], failed_ordinals=[t['ordinal'] for t in trials if not t['task_success']],
        training_elapsed_seconds=number(train['elapsed_seconds']),
        logged_update_seconds=sum(m['train/seconds'] for m in metrics),
        evaluation_trial_wall_seconds=sum(t['wall_seconds'] for t in trials),
        original_job_elapsed_seconds=total_seconds,
        sampled_chunks=metrics[-1]['train/sampled_chunks'],
        first_loss=metrics[0]['train/loss'], final_loss=metrics[-1]['train/loss'],
        mean_loss=sum(m['train/loss'] for m in metrics) / 5000,
        peak_cuda_allocated_bytes=max(m.get('train/peak_cuda_allocated_bytes',
                                           m.get('train/cuda_memory_bytes/peak_allocated', 0)) for m in metrics),
        gpu_utilization=None, gpu_power_watts=None, gpu_energy_joules=None,
        missing_telemetry='not_recorded', offline_probes=probe_summary, training_statistics=selected_stats,
        scope='20 sampling seeds, one selected geometry; not spatial generalization or architecture-only comparison',
    )
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    row = dict(model=model, source_hash=source_hash, source_files=sources, config=config,
               summary=summary, history=metrics, trials=trials,
               earlier_experiment_index='docs/BENCHMARK_STATUS.md',
               registry_scope='only the two validated fixed-layout baselines, not every past attempt')
    audit_json(row)
    json.dumps(row, allow_nan=False)
    return row


def build_registry(repo):
    return [build_row(repo, model) for model in MODELS]