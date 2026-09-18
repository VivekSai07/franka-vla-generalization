"""Admission and bounded orchestration. Actual policy trials are MAIN-only operations."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

import numpy as np
import torch

from data_collection.benchmark import write_json
from experiments.smolvla_fixed_layout import evaluate as smol
from experiments.xvla_fixed_layout import evaluate as xvla
from . import contract as c, scenes as s, telemetry

EVALUATORS = {'smolvla': smol, 'xvla': xvla}
SEMANTIC_FIELDS = (
    'episode_index', 'source_episode_index', 'pick', 'place', 'colors', 'ordinal', 'seed', 'rollout_seed',
    'instruction', 'outcome', 'ticks', 'action_attempts', 'completed_control_ticks', 'sim_seconds',
    'actual_chunk_predictions', 'chunk_size', 'n_action_steps', 'task_success', 'unassisted_success',
    'intervention_steps', 'joint_clipped_ticks', 'joint_clipped_ticks_per_dimension', 'gripper_clipped_ticks',
    'gripper_thresholded_ticks', 'max_joint_clip_rad', 'gripper_adapter', 'success_streak', 'stages', 'final_metrics')
ARTIFACT_FIELDS = ('file', 'sha256', 'scene_hash', 'reset_sha256', 'wall_seconds')


def baseline_directory(model, ordinal):
    return EVALUATORS[model].c.OUTPUT/f'evaluation/trial_{ordinal:02d}_seed_{20260917+ordinal}'


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not arrays or any(a.dtype.kind not in 'bifu' or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite/non-numeric or empty saved trace')
    return arrays


def compare_control(row, trace_path, baseline):
    reference = json.loads((baseline/'episode.json').read_text())
    required = set(SEMANTIC_FIELDS) | set(ARTIFACT_FIELDS)
    if set(row) != required or set(reference) != required:
        raise ValueError('Unlisted/missing episode semantic field')
    for key in SEMANTIC_FIELDS:
        if not c._same(row[key], reference[key]): raise ValueError(f'Control semantic reproduction mismatch: {key}')
    actual, expected = read_arrays(trace_path), read_arrays(baseline/'trace.npz')
    if actual.keys() != expected.keys(): raise ValueError('Control trace array inventory mismatch')
    for key in expected:
        if not s.exact(actual[key], expected[key]): raise ValueError(f'Control trace byte reproduction mismatch: {key}')
    return dict(matched=True, arrays=len(actual), semantic_fields=len(SEMANTIC_FIELDS),
                reference_trace_sha256=c.sha256(baseline/'trace.npz'),
                reference_episode_sha256=c.sha256(baseline/'episode.json'),
                expected_task_success=reference['task_success'], video_byte_equality_required=False)


def source_snapshot():
    """Bounded consumed inventories; no old experiment-wide scan or final-scene bank."""
    snapshot = dict(package_sources=c.package_hashes(), baselines={}, required_image=c.IMAGE,
                    original_evaluation_files=c.verify_baselines())
    for name, evaluator in EVALUATORS.items():
        binding = evaluator.c.admission()
        checkpoint = evaluator.verify_final(evaluator.c.OUTPUT, binding)
        evaluation = evaluator.c.OUTPUT/'evaluation/result.json'
        report = json.loads(evaluation.read_text())
        if (report.get('status') != 'complete' or report.get('binding') != binding
                or report.get('final_data_binding_passed') is not True
                or report.get('training_bytes_unchanged') is not True):
            raise ValueError('Original evaluated baseline is not complete and currently bound')
        evaluator.require_complete_trials(report['episodes'])
        files = {str(evaluation.relative_to(c.ROOT)): c.sha256(evaluation),
                 str((evaluator.c.OUTPUT/'result.json').relative_to(c.ROOT)): c.sha256(evaluator.c.OUTPUT/'result.json')}
        for ordinal in (1, 6):
            directory = baseline_directory(name, ordinal)
            row = json.loads((directory/'episode.json').read_text())
            if not c._same(row, report['episodes'][ordinal-1]): raise ValueError('Baseline episode/result disagreement')
            if (row['seed'] != 20260917+ordinal or row['ordinal'] != ordinal or row['episode_index'] != 7
                    or row['source_episode_index'] != 7 or row['reset_sha256'] != c.RESET_SHA256
                    or row['scene_hash'] != c.SOURCE_SCENE_SHA256
                    or row['task_success'] is not (not (name == 'smolvla' and ordinal == 6))):
                raise ValueError('Original control identity/outcome changed, including known Smol failure')
            compare_control(row, directory/'trace.npz', directory)
            for file in ('episode.json', 'trace.npz'):
                path = directory/file; files[str(path.relative_to(c.ROOT))] = c.sha256(path)
        snapshot['baselines'][name] = dict(binding=binding, checkpoint=str(checkpoint.relative_to(c.ROOT)),
                                          checkpoint_files=evaluator.c.inventory(checkpoint), evaluated_files=files)
    snapshot['reset_sha256'] = c.sha256(c.PARENT/'reset.npz')
    snapshot['configuration_sha256'] = c.sha256(c.PARENT/'configuration.json')
    if snapshot['reset_sha256'] != c.RESET_SHA256: raise ValueError('Source reset changed')
    return snapshot


def preflight() -> dict:
    """Read-only CPU admission: no policy construction, writes, steps or rendering."""
    plan = c.make_plan(); c.validate_plan(plan)
    snapshot = source_snapshot()  # BOTH old admission + verify_final gates finish before any inference.
    model, manifest, source = s.load_source()
    bank = [s.bank_summary(s.derive_scene(model, source, condition)) for condition in plan['conditions']]
    if len({item['scene_hash'] for item in bank}) != 17: raise ValueError('Duplicate scene in bank')
    if c.package_hashes() != snapshot['package_sources']: raise ValueError('Package changed during CPU preflight')
    return dict(status='passed', scope='CPU_nonstepping_no_policy_no_renderer_no_output_writes',
                policy_loads=0, physics_steps=0, bank_conditions=17, planned_trials=68,
                plan_sha256=c.digest(plan), snapshot=snapshot, bank=bank,
                main_independent_review_and_fresh_cpu_preflight_required=True)


def claim_output(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)): raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not smol.c.is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Output already exists; no resume/retry/overwrite')
    else: path.mkdir(parents=True, exist_ok=False)
    with (path/'run_claim.json').open('x') as stream:
        json.dump(dict(purpose='local_position_probe', required_image=c.IMAGE, trials=68, resume=False), stream)


def require_disk(path):
    path = Path(path)
    while not path.exists(): path = path.parent
    free = shutil.disk_usage(path).free
    if free < c.MIN_DISK_BYTES: raise ValueError('20GiB output-filesystem reserve required; no override')
    return free


def require_gpu():
    if not torch.cuda.is_available(): raise ValueError('CUDA inference device required for --run only')
    free, total = torch.cuda.mem_get_info()
    if free < c.MIN_GPU_BYTES: raise ValueError('12GiB free GPU memory required; no override')
    return dict(device=torch.cuda.current_device(), free_bytes=free, total_bytes=total)


@contextmanager
def wall_guard():
    """Hard main-thread infrastructure cap, including admission/load and each frozen trial."""
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.): raise ValueError('Existing process alarm; refuse to override')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame): raise TimeoutError('Four-hour infrastructure wall guard expired; no extension')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, c.MAX_SECONDS)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def save_bank(output, proof):
    plan = c.make_plan(); c.validate_plan(plan)
    model, manifest, source = s.load_source()
    derived = [s.derive_scene(model, source, condition) for condition in plan['conditions']]
    summary = [s.bank_summary(item) for item in derived]
    if 'bank' in proof and summary != proof['bank']: raise ValueError('Bank differs from nonstepping preflight')
    bank = output/'bank'; bank.mkdir(exist_ok=False)
    entries = {}
    for item in derived:
        condition_id = item['condition']['condition_id']
        filename = condition_id+'.npz'
        with (bank/filename).open('xb') as stream: np.savez_compressed(stream, **item['arrays'])
        entry = dict(episode_index=7, source_episode_index=7, pick='red', place='green', colors=['red', 'green'],
                     file=filename, sha256=c.sha256(bank/filename), scene_hash=item['scene_hash'])
        # Exercise the SAME restore path used by both frozen runners before any model loading.
        smol.restore_scene(model, bank, entry)
        entries[condition_id] = entry
    write_json(bank/'manifest.json', dict(source_reset_sha256=c.RESET_SHA256, entries=entries, checks=summary))
    write_json(output/'plan.json', plan)
    write_json(output/'preflight.json', proof)
    return model, manifest, bank, entries


def bank_files(output):
    names = {condition['condition_id']+'.npz' for condition in c.make_plan()['conditions']} | {'manifest.json'}
    if {p.name for p in (output/'bank').iterdir()} != names: raise ValueError('Bank file inventory changed')
    paths = [output/'plan.json', output/'preflight.json']+[output/'bank'/name for name in sorted(names)]
    if any(p.is_symlink() for p in paths): raise ValueError('Bank artifact symlink forbidden')
    return {str(p.relative_to(output)): c.sha256(p) for p in paths}


def load_policy(name):
    """Original FP32-aware loaders and saved pre/postprocessors; no fresh probes/optimizer."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    evaluator = EVALUATORS[name]
    checkpoint = evaluator.c.OUTPUT/'checkpoints/005000'
    if name == 'smolvla':
        policy = smol.load_counted(checkpoint).to('cuda')
    else:
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = 'cpu'
        policy = xvla.c.p.load_weights(xvla.CountedPolicy(config), checkpoint/'model.safetensors').to('cuda')
    policy.config.device = 'cuda'
    policy.eval()
    for parameter in policy.parameters(): parameter.requires_grad_(False)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    return policy, pre, post


def release_policy(bundle):
    # Caller also drops its reference before loading the next model; no simultaneous models.
    del bundle
    gc.collect()


def invoke_trial(model_name, bundle, model, manifest, entry, bank, output, ordinal, seed):
    return EVALUATORS[model_name].run_trial(*bundle, model, manifest, entry, bank, output, ordinal, seed)


def validate_result(slot, row, directory, entry, bank):
    if set(row) != set(SEMANTIC_FIELDS) | set(ARTIFACT_FIELDS): raise ValueError('Unexpected result fields')
    expected = dict(**entry, ordinal=slot['baseline_ordinal'], seed=slot['seed'], rollout_seed=slot['seed'],
                    reset_sha256=entry['sha256'], chunk_size=50 if slot['model'] == 'smolvla' else 30, n_action_steps=10)
    if any(not c._same(row.get(k), v) for k, v in expected.items()): raise ValueError('Trial seed/ordinal/scene identity changed')
    if (type(row['task_success']) is not bool or row['outcome'] != ('success' if row['task_success'] else 'timeout')
            or not finite(row)):
        raise ValueError('Invalid/nonfinite result; no retry or continuation')
    if not c._same(json.loads((directory/'episode.json').read_text()), row): raise ValueError('Saved episode differs from return')
    trace = read_arrays(directory/'trace.npz')
    initial = read_arrays(bank/entry['file'])['initial_state']
    if not s.exact(trace['initial_integration_state'], initial): raise ValueError('Trial did not start from saved full bank state')
    stages = EVALUATORS[slot['model']].stage_metrics(trace)
    if not c._same(row['stages'], stages): raise ValueError('Saved trace/stages disagree')


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value, (float, np.floating)) else True


def stage_failure(row):
    if row['task_success']: return 'success'
    for field, label in (('sustained_closure_30_tick', 'closure'), ('sustained_bilateral_5_tick', 'grasp'),
                         ('first_contact_lift_gt_5cm_tick', 'lift'), ('stable_stack_30_tick', 'placement')):
        if row['stages'].get(field) is None: return label
    return 'timeout_after_stages'


def summarize(rows) -> dict:
    plan = c.make_plan()
    valid_prefix = (len(rows) <= 68 and finite(rows) and all(
        all(c._same(row.get(k), v) for k, v in slot.items()) and isinstance(row.get('result'), dict)
        and all(c._same(row['result'].get(k), v) for k, v in dict(seed=slot['seed'], rollout_seed=slot['seed'],
            ordinal=slot['baseline_ordinal'], episode_index=7, source_episode_index=7).items())
        and type(row['result'].get('task_success')) is bool
        and row['result'].get('outcome') == ('success' if row['result']['task_success'] else 'timeout')
        for row, slot in zip(rows, plan['trials'])))
    complete = valid_prefix and len(rows) == 68
    def counts(selected, planned):
        successes = sum(row['result']['task_success'] is True for row in selected)
        failures = {}
        for row in selected:
            key = stage_failure(row['result']); failures[key] = failures.get(key, 0)+1
        return dict(planned=planned, finished=len(selected), successes=successes,
                    success_rate=successes/planned if complete else None, stage_counts=failures)
    models = {}
    for model in c.MODELS:
        selected = [row for row in rows if row['model'] == model]
        perturbations = [row for row in selected if row['condition_id'] != 'control']
        models[model] = dict(controls=counts([row for row in selected if row['condition_id'] == 'control'], 2),
                            perturbations=counts(perturbations, 32),
                            per_condition={item['condition_id']: counts([row for row in selected if row['condition_id'] == item['condition_id']], 2)
                                           for item in plan['conditions']},
                            per_object={color: counts([row for row in perturbations if row['condition_id'].startswith(color+'_')], 16) for color in ('red', 'green')},
                            per_magnitude_mm={str(mm): counts([row for row in perturbations if f'_{mm}mm_' in row['condition_id']], 16) for mm in (5, 10)})
    indexed = {(row['model'], row['condition_id'], row['seed']): row['result']['task_success'] for row in rows}
    pairs = []
    for item in plan['conditions']:
        for seed in c.SEEDS:
            keys = [(model, item['condition_id'], seed) for model in c.MODELS]
            if all(key in indexed for key in keys):
                pairs.append(dict(condition_id=item['condition_id'], seed=seed,
                                  smolvla=indexed[keys[0]], xvla=indexed[keys[1]]))
    return dict(planned_trials=68, finished_trials=len(rows), all_trials_valid=complete, valid_canonical_prefix=valid_prefix,
                per_model=models, paired_outcomes=pairs, descriptive_only=True,
                partial_success_rates_suppressed=not complete, scope=plan['scope'])


def cpu_resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(logical_cpus=os.cpu_count(), user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                max_rss_kib=usage.ru_maxrss, python_version='.'.join(map(str, __import__('sys').version_info[:3])))


def journal(path, event, slot):
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event=event, utc=telemetry.utc(), monotonic_seconds=time.monotonic(), **slot), allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def run() -> dict:
    output = c.OUTPUT
    require_disk(output)
    claim_output(output)  # Single whole-job claim. Failed claims never touch previous artifacts.
    started = time.monotonic()
    report = dict(status='preparing', phase='admission', started_utc=telemetry.utc(), episodes=[],
                  cpu_start=cpu_resources(), telemetry_scope='devicewide_shared_gpu', cloud_logging='deferred_not_required',
                  **summarize([]))
    write_json(output/'result.json', report)
    bundle, snapshot, sampler = None, None, None
    try:
        with wall_guard():
            proof = preflight(); snapshot = proof['snapshot']
            model, manifest, bank, entries = save_bank(output, proof)
            report['bank_files'] = bank_files(output)
            if source_snapshot() != snapshot: raise ValueError('Admission/source changed before model loading')
            # Match frozen runner setup before CUDA initialization; never load an environment file.
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                              HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                              CUBLAS_WORKSPACE_CONFIG=':4096:8')
            report['gpu_before_load'] = require_gpu()
            report['disk_free_before_load'] = require_disk(output)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            with telemetry.Sampler(output/'telemetry.jsonl', lambda: report['phase']) as sampler:
                try:
                    for model_name in c.MODELS:
                        report.update(status='running', phase=f'load:{model_name}')
                        write_json(output/'result.json', report)
                        require_gpu(); require_disk(output)
                        torch.set_num_threads(8 if model_name == 'smolvla' else 4)
                        bundle = load_policy(model_name)
                        for slot in [item for item in c.make_plan()['trials'] if item['model'] == model_name]:
                            if len(report['episodes']) != slot['study_index']-1 or len(report['episodes']) >= 68:
                                raise ValueError('Noncanonical trial count/order; no extra invocation')
                            require_disk(output)
                            if time.monotonic()-started >= c.MAX_SECONDS: raise TimeoutError('Four-hour wall guard')
                            report.update(phase=slot['trial_id'], active_trial=slot)
                            write_json(output/'result.json', report)
                            directory = output/slot['trial_id']; directory.mkdir(exist_ok=False)
                            entry = entries[slot['condition_id']]
                            if c.sha256(bank/entry['file']) != entry['sha256']: raise ValueError('Saved bank changed')
                            journal(output/'journal.jsonl', 'trial_start', slot)
                            row = invoke_trial(model_name, bundle, model, manifest, entry.copy(), bank, directory,
                                               slot['baseline_ordinal'], slot['seed'])
                            if time.monotonic()-started >= c.MAX_SECONDS: raise TimeoutError('Four-hour wall guard after trial')
                            validate_result(slot, row, directory, entry, bank)
                            reproduction = None
                            if slot['condition_id'] == 'control':
                                reproduction = compare_control(row, directory/'trace.npz', baseline_directory(model_name, slot['baseline_ordinal']))
                            report['episodes'].append(dict(**slot, result=row, control_reproduction=reproduction))
                            journal(output/'journal.jsonl', 'trial_end', slot)
                            report.update(summarize(report['episodes']))
                            write_json(output/'result.json', report)
                        release_policy(bundle); bundle = None
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if not report['all_trials_valid']: raise ValueError('Exactly 68 canonical finite slots required')
                finally:
                    if bundle is not None:
                        release_policy(bundle); bundle = None
                    report['phase'] = 'final_source_recheck'
                    if bank_files(output) != report['bank_files']: raise ValueError('Bank/plan/proof bytes changed during run')
                    if source_snapshot() != snapshot: raise ValueError('Source/baseline/checkpoint changed during run')
            if time.monotonic()-started >= c.MAX_SECONDS: raise TimeoutError('Four-hour wall guard before completion')
            report.update(status='complete', phase='complete', active_trial=None, final_source_binding_passed=True)
    except BaseException as exc:
        report.update(status='failed', phase='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(finished_utc=telemetry.utc(), wall_seconds=time.monotonic()-started, cpu_end=cpu_resources(),
                      telemetry_errors=sampler.errors if sampler is not None else [])
        write_json(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='read-only nonstepping CPU preflight')
    mode.add_argument('--run', action='store_true', help='MAIN only: exclusive fixed 68-trial GPU job')
    args = parser.parse_args()
    print(json.dumps(preflight() if args.check else run(), indent=2, allow_nan=False))


if __name__ == '__main__': main()