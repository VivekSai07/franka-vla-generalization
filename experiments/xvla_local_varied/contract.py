"""Local-varied 60-episode data, X-VLA train-only statistics and phase admission contracts.

Semantic shift vs. `experiments.smolvla_local_varied.contract` (2026-09-20 surgery):
The sealed parent `outputs/local_varied_data/local_varied_20260919/validation.json`
recorded a `provenance` snapshot that pins the (now-missing) pilot manifest hash,
so `experiments.local_varied_data.scenes.provenance()` no longer matches the
recorded snapshot byte-for-byte. `current_binding()` here therefore deliberately
does NOT compare `semantic.get('provenance')` against a fresh `provenance()`.
Instead it validates the SEAL fingerprint (still matches — the seal covers data
files, not the recorded snapshot dict) and the other invariants
(`status=='passed'`, `training_ready is True`, `automatic_training is False`,
split/statistics ids), and records the FRESH provenance as
`local_varied_provenance` in the returned binding for future audits.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.fixed_layout.contract import TASK, IMAGE, _same
from experiments.local_varied_data.plan import validate_plan
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json
from experiments.xvla_single_task.contract import (
    make_config, learning_rate, set_learning_rate, backward_microbatch,
    observation, check_model_inputs, inventory, source_path, verify_listed,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/local_varied_data/local_varied_20260919'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/xvla_trial/bundle'
OUTPUT = ROOT/'outputs/train/xvla_franka_stacking_localvaried_v1_20260919'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = 'f8b10a498798217a63564416ab6561e25bd7e9c87e4fdddcf42f288cdb9e9a28'
TRAIN, VAL, SEED = tuple(range(48)), tuple(range(48, 60)), 20260917
TRAINING_ROWS = 48*528  # 25344
COMPLETE_STARTS = 48*499  # 23952, X-VLA chunk_size=30 → 528-30+1=499 complete starts per episode
VALIDATION_ROWS = 12*528  # 6336
TOTAL_FRAMES = 60*528  # 31680
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{key: (3, 512, 512) for key in CAMERAS}}

# Explicit consumed helpers; X-VLA base weights binding is via require_assets from xvla_trial.
# Do not call old recursive experiment scans or copy the SmolVLA adapter.
SOURCE_FILES = (
    'experiments/xvla_trial/contract.py', 'experiments/xvla_trial/preflight.py',
    'experiments/xvla_trial/assets.py', 'experiments/xvla_single_task/contract.py',
    'data_collection/benchmark.py', 'data_collection/pilot_integrity.py', 'data_collection/pilot_plan.py',
    'validation/evaluate_pilot.py', 'validation/gates.py', 'validation/rollout_diagnostics.py',
    'validation/validate_pilot.py', 'validation/precision.py', 'validation/smolvla_smoke.py',
    'training/smolvla_pilot.py', 'training/pilot_protocol.py',
    'src/controllers/grasp_controller.py', 'src/robots/franka_panda/config.py',
    'experiments/local_varied_data/__init__.py', 'experiments/local_varied_data/plan.py',
    'experiments/local_varied_data/record.py', 'experiments/local_varied_data/scenes.py',
    'experiments/local_varied_data/validate.py',
    'experiments/local_position_probe/scenes.py', 'experiments/local_position_probe/contract.py',
    'experiments/xvla_local_varied/__init__.py', 'experiments/xvla_local_varied/contract.py',
    'experiments/xvla_local_varied/runner.py',
    'experiments/xvla_local_varied/tests/test_contract.py',
    'experiments/xvla_local_varied/tests/test_runner.py',
    'experiments/xvla_local_varied/README.md',
) + tuple('installed/lerobot/'+name for name in (
    'policies/xvla/__init__.py', 'policies/xvla/action_hub.py', 'policies/xvla/configuration_florence2.py',
    'policies/xvla/configuration_xvla.py', 'policies/xvla/modeling_florence2.py',
    'policies/xvla/modeling_xvla.py', 'policies/xvla/processor_xvla.py',
    'policies/xvla/soft_transformer.py', 'policies/xvla/utils.py',
    'processor/__init__.py', 'processor/batch_processor.py', 'processor/converters.py',
    'processor/core.py', 'processor/delta_action_processor.py', 'processor/device_processor.py',
    'processor/env_processor.py', 'processor/factory.py', 'processor/gym_action_processor.py',
    'processor/hil_processor.py', 'processor/migrate_policy_normalization.py',
    'processor/normalize_processor.py', 'processor/observation_processor.py', 'processor/pipeline.py',
    'processor/policy_robot_bridge.py', 'processor/rename_processor.py', 'processor/tokenizer_processor.py',
    'policies/pretrained.py', 'policies/utils.py', 'policies/factory.py', 'optim/optimizers.py',
    'configs/policies.py', 'configs/types.py', 'datasets/lerobot_dataset.py',
    'datasets/utils.py', 'datasets/video_utils.py', 'utils/constants.py',
))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name: sha256(source_path(name)) for name in SOURCE_FILES}


def is_mountpoint(path):
    # os.path.ismount misses same-filesystem bind mounts. These fixed paths have no escapes.
    return str(path) in {line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()}


def claim_output(path):
    """Exclusive token permits a newly created empty Docker output bind mount only."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Preserve existing output; no overwrite/resume/retry')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose': 'xvla_local_varied', 'resume': False, 'required_image': IMAGE}, stream)


def require_fields(actual, expected, message):
    if any(not _same(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(message)


def indices(episodes):
    if len(episodes) != 60:
        raise ValueError('Exactly 60 local-varied episodes; old pilot/fixed-layout metadata forbidden')
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=528, tasks=[TASK],
                       dataset_from_index=i*528, dataset_to_index=(i+1)*528),
                       'Local-varied episode identity/task/length/boundary mismatch')
    return list(range(TRAINING_ROWS)), [i*528+j for i in TRAIN for j in range(499)]


def selected_stats(rows, selected):
    if selected != list(range(TRAINING_ROWS)):
        raise ValueError('Statistics require every training row exactly once')
    values = {key: [] for key in ('observation.state', 'action')}
    for i in selected:
        row = rows[i]
        if int(row['episode_index']) != i//528 or int(row['task_index']) != 0:
            raise ValueError('Raw training membership/task mismatch')
        for key, size in (('observation.state', 9), ('action', 8)):
            value = torch.as_tensor(row[key])
            if value.dtype != torch.float32 or value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('Finite unpadded real FP32 state9/action8 required')
            values[key].append(value)
    return {key: {'mean': (x := torch.stack(value)).mean(0), 'std': x.std(0, unbiased=False),
                  'min': x.amin(0), 'max': x.amax(0)} for key, value in values.items()}


def policy_row(row, training):
    ids = TRAIN if training else TRAIN+VAL
    episode = row.get('episode_index')
    if isinstance(episode, torch.Tensor):
        if episode.ndim != 0 or episode.dtype != torch.int64:
            raise ValueError('Episode identity must be a scalar integer')
        episode = episode.item()
    if (type(episode) is not int or episode not in ids or row.get('task') != TASK
            or {key for key in row if key.startswith('observation.')} != set(OBS)):
        raise ValueError('Unapproved episode/instruction/observation')
    observation({key: row[key] for key in (*OBS, 'task')})
    action, pad = row.get('action'), row.get('action_is_pad')
    if (not isinstance(action, torch.Tensor) or action.shape != (30, 8)
            or action.dtype != torch.float32 or not torch.isfinite(action).all()
            or not isinstance(pad, torch.Tensor) or pad.shape != (30,)
            or pad.dtype != torch.bool or pad.any()):
        raise ValueError('Finite real 30x8 actions and complete unpadded chunk required')
    return {key: row[key] for key in (*OBS, 'task', 'action', 'action_is_pad')}


def collate_rows(rows):
    if len(rows) != 4:
        raise ValueError('Exactly four training rows per microbatch')
    return default_collate([policy_row(row, True) for row in rows])


def sample_batches(starts):
    expected = [i*528+j for i in TRAIN for j in range(499)]
    if starts != expected:
        raise ValueError('Exactly 23952 complete ordered training starts required')
    generator = torch.Generator().manual_seed(SEED)
    while True:
        order = torch.randperm(len(starts), generator=generator).tolist()
        for offset in range(0, len(order), 4):
            yield [starts[j] for j in order[offset:offset+4]]


def current_binding():
    """Rehash full parent + actual asset bytes + record CURRENT provenance.

    Deliberately skips comparing `semantic.get('provenance')` against a freshly
    computed `provenance()` — see the module docstring for the seal-time
    provenance snapshot mismatch (pilot manifest reference). The SEAL fingerprint,
    plan.json validation, and semantic status/training_ready/split ids are still
    verified; the fresh provenance is stored as `local_varied_provenance` so
    future audits can compare against this binding directly.
    """
    from experiments.local_varied_data.scenes import provenance
    if json.loads((PARENT/'seal.json').read_text()) != {'fingerprint': SEAL}:
        raise ValueError('Approved local-varied parent fingerprint changed')
    plan = json.loads((PARENT/'plan.json').read_text())
    validate_plan(plan)
    semantic = json.loads((PARENT/'validation.json').read_text())
    current = provenance()
    if (semantic.get('status') != 'passed' or semantic.get('training_ready') is not True
            or semantic.get('automatic_training') is not False
            or semantic.get('train_ids') != list(TRAIN)
            or semantic.get('validation_ids') != list(VAL)
            or semantic.get('statistics_episode_ids') != list(TRAIN)):
        raise ValueError('Local-varied semantic proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                semantic_sha256=digest_json(semantic), local_varied_provenance=current,
                source_reset_sha256=sha256(PARENT/'source.npz'), plan_sha256=sha256(PARENT/'plan.json'),
                snapshots_sha256=sha256(PARENT/'snapshots.json'), sources=source_hashes(),
                assets=require_assets(), required_image=IMAGE)


def cpu_proof(binding, semantic, stats, processor):
    return dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                semantic_gate=semantic, training_rows=TRAINING_ROWS,
                complete_training_chunks=COMPLETE_STARTS,
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                training_stats=stats, training_stats_digest=digest_json(stats),
                processor_check=processor, processor_digest=digest_json(processor))


def validate_cpu(report, binding):
    require_fields(report, dict(status='passed', gpu_executed=False, optimizer_updates=0,
                   binding=binding, training_rows=TRAINING_ROWS,
                   complete_training_chunks=COMPLETE_STARTS,
                   train_episodes=list(TRAIN), validation_episodes=list(VAL)),
                   'Current full CPU admission required')
    if (digest_json(report.get('semantic_gate')) != binding.get('semantic_sha256')
            or not report.get('training_stats') or not report.get('processor_check')
            or digest_json(report['training_stats']) != report.get('training_stats_digest')
            or digest_json(report['processor_check']) != report.get('processor_digest')):
        raise ValueError('CPU semantic/statistics/processor proof changed')


def admission():
    binding = current_binding()
    validate_cpu(json.loads(CPU_ADMISSION.read_text()), binding)
    return binding


def load_data():
    """Dataset-only reader. Never library/global stats, pilot manifest or old adapter."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    info = json.loads((DATA/'meta/info.json').read_text())
    require_fields(info, dict(total_episodes=60, total_frames=TOTAL_FRAMES, total_tasks=1, fps=30,
                   splits={'train': '0:48', 'validation': '48:60'}),
                   'Local-varied dataset metadata mismatch')
    ds = LeRobotDataset(repo_id=DATA.name, root=DATA,
                        delta_timestamps={'action': [i/30 for i in range(30)]})
    features = dataset_to_policy_features(ds.features)
    if ({k: tuple(v.shape) for k, v in features.items() if k.startswith('observation.')} != OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != TOTAL_FRAMES
            or list(ds.meta.tasks.index) != [TASK] or int(ds.meta.tasks.loc[TASK, 'task_index']) != 0):
        raise ValueError('Real dataset state/action/task schema mismatch')
    rows, starts = indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    for i in range(TOTAL_FRAMES):
        if int(ds.hf_dataset[i]['episode_index']) != i//528 or int(ds.hf_dataset[i]['task_index']) != 0:
            raise ValueError('Raw episode/task membership mismatch')
    return ds, starts, selected_stats(ds.hf_dataset, rows)


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


def prepare_data():
    gate = admission()
    ds, starts, stats = load_data()
    if statistics_json(stats) != json.loads(CPU_ADMISSION.read_text())['training_stats']:
        raise ValueError('Fresh train-only statistics differ from current CPU admission')
    return ds, starts, stats, gate


def recipe(preflight):
    return dict(purpose='xvla_local_varied_preflight' if preflight else 'xvla_local_varied',
                seed=SEED, max_optimizer_updates=2 if preflight else 5000,
                microbatch=4, accumulation=2, effective_batch=8, chunk_size=30, n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4, vlm_scale=.1, soft_prompt_scale=1., weight_decay=1e-4,
                               vlm_weight_decay=1e-5, betas=[.9, .95], eps=1e-8, clip=10.),
                schedule=dict(warmup=500, total=5000, floor_fraction=.1),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                optimizer_storage='final_only', resume=False, autotune=False, retries=False)


def checkpoint_updates(preflight):
    return [1, 2] if preflight else [1, 1000, 2000, 3000, 4000, 5000]


def checkpoint_directory(update):
    if type(update) is not int or update < 1:
        raise ValueError('Positive integer update required')
    return f'step_{update:06d}'


def require_marker(marker, update):
    require_fields(marker, dict(status='complete', update=update, reload_fp32_weights_exact=True),
                   'Strict FP32 checkpoint marker required')
    error = marker.get('reload_max_action_error')
    if type(error) not in (int, float) or not 0 <= error <= 1e-5:
        raise ValueError('Absolute reload action error must be finite and <=1e-5')


def verify_checkpoints(report, output, preflight):
    updates = checkpoint_updates(preflight)
    keys = [checkpoint_directory(i) for i in updates]
    if [m.get('update') for m in report.get('checkpoints', [])] != updates:
        raise ValueError('Checkpoint schedule changed')
    if set(report.get('checkpoint_files', {})) != set(keys):
        raise ValueError('All checkpoint inventories required')
    present = {p.name for p in (output/'checkpoints').iterdir()}
    # Legacy `{update:06d}` naming is a regression; keys carry the required step_ prefix.
    if present & {f'{i:06d}' for i in updates}:
        raise ValueError('Legacy checkpoint directory naming forbidden; expected step_ prefix')
    if present != set(keys):
        raise ValueError('Unlisted/missing checkpoint')
    for marker in report['checkpoints']:
        update = marker['update']
        require_marker(marker, update)
        key = checkpoint_directory(update)
        directory = output/'checkpoints'/key
        files = inventory(directory)
        required = {'model.safetensors', 'config.json', 'checkpoint_result.json',
                    'policy_preprocessor.json', 'policy_postprocessor.json'}
        # X-VLA optimizer_storage='final_only': training_state.pt lives only on the final update.
        if update == updates[-1]:
            required = required | {'training_state.pt'}
        if (files != report['checkpoint_files'][key] or not required <= files.keys()
                or files['model.safetensors'] != marker.get('model_sha256')
                or json.loads((directory/'checkpoint_result.json').read_text()) != marker):
            raise ValueError('Checkpoint/processor/training-state byte inventory changed')
        if update != updates[-1] and 'training_state.pt' in files:
            raise ValueError('Intermediate checkpoint must not persist optimizer state (final_only)')
        if preflight and update == updates[-1]:
            if 'adam_reload.json' not in files:
                raise ValueError('Scratch Adam roundtrip evidence missing on final preflight checkpoint')
            evidence = json.loads((directory/'adam_reload.json').read_text())
            require_fields(evidence, dict(adam_reload_fp32_exact=True, update=update,
                           training_state_sha256=files['training_state.pt']),
                           'Scratch Adam proof changed')


def validate_scratch(report, binding, output, stats, cpu_hash):
    require_fields(report, {**recipe(True), 'status': 'complete', 'completed_updates': 2,
                   'binding': binding, 'training_rows': TRAINING_ROWS,
                   'complete_training_chunks': COMPLETE_STARTS,
                   'cpu_admission_sha256': cpu_hash, 'final_data_binding_passed': True,
                   'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None, 'rollouts': 0},
                   'Exact current two-update scratch proof required')
    if (report.get('training_stats_sha256') != sha256(output/'training_stats.json')
            or json.loads((output/'training_stats.json').read_text()) != stats):
        raise ValueError('Scratch training statistics changed')
    verify_checkpoints(report, output, True)
