"""B3 robosuite 560-episode data, X-VLA train-only statistics and phase admission contracts.

Forked from the B2 X-VLA training contract against the sealed parent
`outputs/b3_robosuite_data/b3_robosuite_20261003` (robosuite scene, MuJoCo 3.4.0). Training needs
no simulator and runs in the admission image.

What differs from the fork source:

* The parent is bound by its seven-field `seal.json` record (`SEAL_RECORD`), by `plan.json`
  (`validate_plan` and the sealed plan digest), by the empty `substitutions.json`, by
  `provenance.json` against a fresh `b3_robosuite_data.contract.provenance()`, and by
  `validation.json`. The full fingerprint recomputation is `parent_fingerprint()`, called by
  `--check` only.
* **The split comes from the plan.** LeRobot's `meta/info.json` calls all 560 episodes `train`
  (`{'train': '0:560'}`); episodes 0-499 train and 500-559 validation per `plan.json`.
* **Tasks are matched by their text.** LeRobot numbers the 10 instructions by first appearance,
  not in the plan's sorted order, so a task number is never trusted.
* The approved instruction set is exactly the 10 trained instructions. The two held-out ones
  (`yellow->red`, `green->blue`) are added only in the evaluation package.
* Horizon: 156,000 updates = 5.0 epochs of 249,500 complete starts, the exposure B2 had.

`experiments/b3_robosuite_data/**` and everything in its `provenance()` are frozen for the
lifetime of this training run.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.fixed_layout.contract import IMAGE, _same
from experiments.b3_robosuite_data import contract as b3_contract
from experiments.b3_robosuite_data import plan as b3_plan
from experiments.b3_robosuite_data import validate as b3_validate
from experiments.b3_robosuite_data.plan import INSTRUCTIONS, validate_plan
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json
from experiments.xvla_single_task import contract as single_task
from experiments.xvla_single_task.contract import (
    make_config as _make_config, backward_microbatch,
    observation, check_model_inputs, inventory, source_path,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/b3_robosuite_data/b3_robosuite_20261003'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/xvla_trial/bundle'
# v2: v1 had no resumable state (Adam saved only on the final update) and was superseded.
OUTPUT = ROOT/'outputs/train/xvla_franka_stacking_b3_robosuite_v2'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
PROOF = OUTPUT.with_name(OUTPUT.name+'_resumeproof')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = 'e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0'
SEAL_RECORD = {
    'fingerprint': SEAL,
    'provenance_sha256': 'd99d2c629099bfbbc9eac723cda48623aa0931dafa94527cef4316f5dd2de7ba',
    'plan_sha256': 'ea6b572f69d15e1b28a1bc70293817f8f33098aa2984ace1ba8acaad0150140f',
    'substitutions_sha256': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    'image': 'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e',
    'min_clearance_m': 0.08,
    'episodes': 560,
}
VALIDATION_RECORD = {'episodes': 560, 'train': 500, 'validation': 60, 'per_cell_train': 25,
                     'per_cell_validation': 3, 'distractors_train': {'0': 100, '1': 200, '2': 200},
                     'edge_train': {'inner': 50, 'interior': 400, 'outer': 50},
                     'substituted_slots': 0, 'held_out_pairs_collected': 0}
TRAIN, VAL, SEED = tuple(range(500)), tuple(range(500, 560)), 20260917
TRAINING_ROWS = 500*528  # 264000
COMPLETE_STARTS = 500*499  # 249500, X-VLA chunk_size=30 -> 528-30+1=499 complete starts per episode
VALIDATION_ROWS = 60*528  # 31680
TOTAL_FRAMES = 560*528  # 295680
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{key: (3, 512, 512) for key in CAMERAS}}

# Explicit consumed helpers; X-VLA base weights binding is via require_assets from xvla_trial.
# Do not call old recursive experiment scans or copy the SmolVLA adapter. The list is exact:
# `tests/test_fork_hygiene.py` asserts it equals the repo modules that importing the contract and
# the runner load (plus this package's tests and README).
SOURCE_FILES = (
    'experiments/xvla_trial/contract.py', 'experiments/xvla_trial/preflight.py',
    'experiments/xvla_trial/assets.py', 'experiments/xvla_single_task/contract.py',
    'experiments/b3_robosuite_data/__init__.py', 'experiments/b3_robosuite_data/plan.py',
    'experiments/b3_robosuite_data/contract.py', 'experiments/b3_robosuite_data/validate.py',
    'experiments/fixed_layout/__init__.py', 'experiments/fixed_layout/contract.py',
    'experiments/paired_demonstrations/plan.py',
    'experiments/xvla_b3_robosuite/__init__.py', 'experiments/xvla_b3_robosuite/contract.py',
    'experiments/xvla_b3_robosuite/runner.py',
    'experiments/xvla_b3_robosuite/tests/test_contract.py',
    'experiments/xvla_b3_robosuite/tests/test_runner.py',
    'experiments/xvla_b3_robosuite/tests/test_fork_hygiene.py',
    'experiments/xvla_b3_robosuite/README.md',
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


@lru_cache(maxsize=1)
def planned_instructions():
    """Plan slot -> instruction, from the sealed plan.json; the dataset must agree episode for episode."""
    plan = json.loads((PARENT/'plan.json').read_text())
    return tuple(episode['instruction'] for episode in plan['episodes'])


def task_index_map(ds):
    """LeRobot numbers tasks by first appearance, not by the plan's sorted order: match by text."""
    table = {str(k): int(v) for k, v in zip(ds.meta.tasks.index, ds.meta.tasks['task_index'])}
    if set(table) != set(INSTRUCTIONS) or sorted(table.values()) != list(range(len(INSTRUCTIONS))):
        raise ValueError(f'Dataset tasks {sorted(table)} are not a bijection over the planned instructions')
    return table


@contextmanager
def approved_instruction(instruction):
    """Swap the approved instruction that `xvla_single_task.contract.observation` compares against.

    That module hardcodes a single `TASK` and is bound by several contracts, so it cannot be
    edited. The membership check keeps the guarantee the original provides: only a trained
    instruction is ever accepted (the held-out ones are not in `INSTRUCTIONS`).
    """
    if instruction not in INSTRUCTIONS:
        raise ValueError(f'Unapproved instruction: {instruction!r}')
    original = single_task.TASK
    single_task.TASK = instruction
    try:
        yield
    finally:
        single_task.TASK = original


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
            raise FileExistsError('Preserve existing output; no overwrite; resume only via --resume')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose': 'xvla_b3_robosuite', 'resume': False, 'required_image': IMAGE}, stream)


def claim_resume(path):
    """Resume only the bind-mounted output that a fresh claim created; every resume leaves a token."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    claim = path/'run_claim.json'
    if not path.is_dir() or not is_mountpoint(path) or not claim.is_file():
        raise FileExistsError('Resume needs the original bind-mounted output holding run_claim.json')
    if json.loads(claim.read_text()) != {'purpose': 'xvla_b3_robosuite', 'resume': False,
                                         'required_image': IMAGE}:
        raise ValueError('run_claim.json is not this package\'s claim')
    number = len(list(path.glob('resume_claim_*.json')))+1
    with (path/f'resume_claim_{number:02d}.json').open('x') as stream:
        json.dump({'purpose': 'xvla_b3_robosuite', 'resume': number, 'required_image': IMAGE}, stream)
    return number


def require_fields(actual, expected, message):
    if any(not _same(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(message)


def indices(episodes):
    if len(episodes) != 560:
        raise ValueError('Exactly 560 B3 episodes; other packages\' metadata forbidden')
    planned = planned_instructions()
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=528, tasks=[planned[i]],
                       dataset_from_index=i*528, dataset_to_index=(i+1)*528),
                       'B3 episode identity/instruction/length/boundary mismatch')
    return list(range(TRAINING_ROWS)), [i*528+j for i in TRAIN for j in range(499)]


def selected_stats(rows, selected, task_of_episode):
    if selected != list(range(TRAINING_ROWS)):
        raise ValueError('Statistics require every training row exactly once')
    values = {key: [] for key in ('observation.state', 'action')}
    for i in selected:
        row = rows[i]
        if int(row['episode_index']) != i//528 or int(row['task_index']) != task_of_episode[i//528]:
            raise ValueError('Raw training membership/instruction mismatch')
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
    if (type(episode) is not int or episode not in ids
            or row.get('task') != planned_instructions()[episode]
            or {key for key in row if key.startswith('observation.')} != set(OBS)):
        raise ValueError('Unapproved episode/instruction/observation')
    with approved_instruction(planned_instructions()[episode]):
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
        raise ValueError(f'Exactly {len(expected)} complete ordered training starts required')
    generator = torch.Generator().manual_seed(SEED)
    while True:
        order = torch.randperm(len(starts), generator=generator).tolist()
        for offset in range(0, len(order), 4):
            yield [starts[j] for j in order[offset:offset+4]]


def parent_fingerprint():
    """Full recomputation over every file in the parent; `--check` only (it reads ~3 GB)."""
    return b3_validate.fingerprint(PARENT)


def current_binding():
    """Cheap, repeatable rehash of the parent's small files, the B3 sources and the assets.

    `provenance.json` must equal a fresh `b3_robosuite_data.contract.provenance()` and hash to the
    digest the seal recorded, so editing any file the collection depended on is caught here.
    """
    if json.loads((PARENT/'seal.json').read_text()) != SEAL_RECORD:
        raise ValueError('Approved B3 parent seal record changed')
    plan = json.loads((PARENT/'plan.json').read_text())
    validate_plan(plan)
    if b3_plan.digest(plan) != SEAL_RECORD['plan_sha256']:
        raise ValueError('B3 plan of record differs from the sealed plan digest')
    substitutions = json.loads((PARENT/'substitutions.json').read_text())
    if substitutions != {} or b3_plan.digest(substitutions) != SEAL_RECORD['substitutions_sha256']:
        raise ValueError('B3 substitutions are not the sealed (empty) record')
    recorded = json.loads((PARENT/'provenance.json').read_text())
    current = b3_contract.provenance()
    if recorded != current or b3_plan.digest(recorded) != SEAL_RECORD['provenance_sha256']:
        raise ValueError('B3 source provenance drifted from the sealed snapshot')
    validation = json.loads((PARENT/'validation.json').read_text())
    if set(validation) != set(VALIDATION_RECORD):
        raise ValueError('B3 validation record has other fields than the sealed one')
    require_fields(validation, VALIDATION_RECORD, 'B3 validation proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                plan_sha256=SEAL_RECORD['plan_sha256'], provenance_sha256=SEAL_RECORD['provenance_sha256'],
                validation_sha256=digest_json(validation), b3_provenance=current,
                sources=source_hashes(), assets=require_assets(), required_image=IMAGE)


def semantic_gate():
    """The collection's own semantic proof (counts, balance, no held-out pair), bound by digest."""
    return json.loads((PARENT/'validation.json').read_text())


def cpu_proof(binding, semantic, stats, processor, parent_fingerprint):
    return dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                semantic_gate=semantic, training_rows=TRAINING_ROWS,
                complete_training_chunks=COMPLETE_STARTS,
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                parent_fingerprint_recomputed=parent_fingerprint == SEAL,
                training_stats=stats, training_stats_digest=digest_json(stats),
                processor_check=processor, processor_digest=digest_json(processor))


def validate_cpu(report, binding):
    require_fields(report, dict(status='passed', gpu_executed=False, optimizer_updates=0,
                   binding=binding, training_rows=TRAINING_ROWS,
                   complete_training_chunks=COMPLETE_STARTS,
                   train_episodes=list(TRAIN), validation_episodes=list(VAL),
                   parent_fingerprint_recomputed=True),
                   'Current full CPU admission required')
    if (digest_json(report.get('semantic_gate')) != binding.get('validation_sha256')
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
    # LeRobot's own split metadata calls every episode "train"; the real split is the plan's.
    require_fields(info, dict(total_episodes=560, total_frames=TOTAL_FRAMES,
                   total_tasks=len(INSTRUCTIONS), fps=30, splits={'train': '0:560'}),
                   'B3 dataset metadata mismatch')
    ds = LeRobotDataset(repo_id=DATA.name, root=DATA,
                        delta_timestamps={'action': [i/30 for i in range(30)]})
    features = dataset_to_policy_features(ds.features)
    if ({k: tuple(v.shape) for k, v in features.items() if k.startswith('observation.')} != OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != TOTAL_FRAMES):
        raise ValueError('Real dataset state/action/task schema mismatch')
    task_of_episode = tuple(task_index_map(ds)[x] for x in planned_instructions())
    rows, starts = indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    for i in range(TOTAL_FRAMES):
        if (int(ds.hf_dataset[i]['episode_index']) != i//528
                or int(ds.hf_dataset[i]['task_index']) != task_of_episode[i//528]):
            raise ValueError('Raw episode/instruction membership mismatch')
    return ds, starts, selected_stats(ds.hf_dataset, rows, task_of_episode)


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


def prepare_data():
    gate = admission()
    ds, starts, stats = load_data()
    if statistics_json(stats) != json.loads(CPU_ADMISSION.read_text())['training_stats']:
        raise ValueError('Fresh train-only statistics differ from current CPU admission')
    return ds, starts, stats, gate


# 249,500 complete starts x 5.0 epochs / effective batch 8 = 155,937.5 updates, rounded to
# 156,000 = 5 x 31,200 so the checkpoints fall on whole-epoch boundaries (31,200 x 8 = 249,600
# chunks, 1.0004 epochs). B2's exposure was 5.01 epochs.
TOTAL_UPDATES = 156000
# Resume snapshot (model + Adam + RNG, ~10.5 GB) every 2,600 updates (~36 min): it divides 31,200, so
# every epoch checkpoint is also a snapshot update, and a crash loses at most 2,600 updates.
SNAPSHOT_EVERY = 2600
KEEP_SNAPSHOTS = 2


def snapshot_updates(preflight):
    """Updates after which a resume snapshot is written; the final update has the final checkpoint."""
    every, maximum = (1, 2) if preflight else (SNAPSHOT_EVERY, TOTAL_UPDATES)
    return list(range(every, maximum, every))


def learning_rate(update):
    """Same shape as `xvla_single_task.contract.learning_rate`, stretched to the longer horizon.

    The shared helper hard-caps `update` at 5000 and `make_config` fixes `scheduler_decay_steps`
    at 5000, so a longer run would raise at update 5001. That file is bound by several contracts,
    so it cannot be edited. The schedule is therefore redefined locally: identical 500-update
    linear warmup to 1e-4, then cosine decay to the same 1e-5 floor over the full horizon.
    """
    if type(update) is not int or not 1 <= update <= TOTAL_UPDATES:
        raise ValueError(f'Update must be integer in [1,{TOTAL_UPDATES}]')
    if update <= 500:
        return 1e-4*update/500
    span = TOTAL_UPDATES-500
    return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(update-500)/span))


def set_learning_rate(optimizer, update):
    """Port of the shared setter bound to the local schedule; same group checks and vlm scale."""
    lr = learning_rate(update)
    if [g['name'] for g in optimizer.param_groups] != ['vlm', 'soft_prompts', 'other']:
        raise ValueError('Expected all three native named optimizer groups')
    for group in optimizer.param_groups:
        scale = .1 if group['name'] == 'vlm' else 1.
        if (not math.isclose(group['weight_decay'], 1e-4*scale, rel_tol=0, abs_tol=1e-15)
                or tuple(group['betas']) != (.9, .95) or group['eps'] != 1e-8):
            raise ValueError('Native optimizer recipe changed')
        group['lr'] = lr*scale


def make_config(device):
    """Shared config with the decay horizon stretched to match the local schedule."""
    config = _make_config(device)
    config.scheduler_decay_steps = TOTAL_UPDATES
    return config


def recipe(preflight):
    return dict(purpose='xvla_b3_robosuite_preflight' if preflight else 'xvla_b3_robosuite',
                seed=SEED, max_optimizer_updates=2 if preflight else TOTAL_UPDATES,
                microbatch=4, accumulation=2, effective_batch=8, chunk_size=30, n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4, vlm_scale=.1, soft_prompt_scale=1., weight_decay=1e-4,
                               vlm_weight_decay=1e-5, betas=[.9, .95], eps=1e-8, clip=10.),
                schedule=dict(warmup=500, total=TOTAL_UPDATES, floor_fraction=.1),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                optimizer_storage='final_only', resume='manual_from_verified_snapshot',
                snapshot_updates=snapshot_updates(preflight), keep_snapshots=KEEP_SNAPSHOTS,
                autotune=False, retries=False)


def checkpoint_updates(preflight):
    # Update 1 and each whole epoch. Disk-bound: ~3.3 GB each plus ~9.9 GB on the final one
    # (optimizer state): 5 x 3.3 + 9.9 = 26.4 GB. The evaluation screens the last four.
    return [1, 2] if preflight else [1] + [TOTAL_UPDATES//5*k for k in range(1, 6)]


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


def validate_resume_proof(report, binding, scratch_report, output):
    """A resumed run reproduced the uninterrupted run's second update bit for bit."""
    require_fields(report, {**recipe(True), 'status': 'complete', 'completed_updates': 2,
                   'binding': binding, 'final_data_binding_passed': True,
                   'adam_reload_fp32_exact': True, 'rollouts': 0},
                   'Exact current resume proof required')
    scratch_model = scratch_report['checkpoints'][-1]['model_sha256']
    require_fields(report.get('resume_proof') or {},
                   dict(from_update=1, compared_update=2, metrics_equal=True,
                        model_sha256_equal=True, training_state_sha256_equal=True,
                        scratch_model_sha256=scratch_model),
                   'Resumed run must reproduce the uninterrupted second update exactly')
    if report['checkpoints'][-1]['model_sha256'] != scratch_model:
        raise ValueError('Resumed final weights differ from the uninterrupted run')
    if report.get('training_stats_sha256') != sha256(output/'training_stats.json'):
        raise ValueError('Resume-proof training statistics changed')
    verify_checkpoints(report, output, True)
