"""Props P3 X-VLA training contract: sealed 544-episode Food-vs-Drinks dataset, admission.

Forked from `experiments.xvla_props_p2` (itself from P1 and B3/B2). Training needs no simulator and runs
in the admission image. The recipe is P1's and P2's, unchanged, so the three differ only in their data.

What differs from the fork source:

* The parent is the sealed `outputs/props_p3_data/props_p3_20261020` (16 merged shards, robosuite
  scene, MuJoCo 3.4.0): 544 episodes of 452 frames, one object each, four trained objects (can, milk,
  bread, lemon), bottle and cereal never collected. Bound by its `seal.json` (`SEAL_RECORD`),
  `plan.json`, the empty `substitutions.json`, `provenance.json` against a fresh
  `props_p3_data.contract.provenance()`, and `validation.json` (its own fields: per-cell balance, objects).
* **One instruction.** Every frame carries the fixed prompt `sort the item`, so the P2 machinery that
  matched ten instructions by text is re-derived for a single string: the dataset's task table must be
  exactly `{prompt: 0}`, every plan slot must carry that prompt, and `approved_instruction` accepts only it.
* Same split rule (plan, not LeRobot metadata), same 452-frame episodes and 480 training episodes, so
  the same 203,040 starts and 126,900 updates (5.0 epochs). Validation is 64 episodes (8 per cell).
* The runner's loop is P2's (fast, identical math, proven by `--equivalence` before `--train`).

`experiments/props_p3_data/**` and everything in its `provenance()` are frozen for this run's lifetime.
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
from experiments.props_p3_data import contract as p3_contract
from experiments.props_p3_data import plan as p3_plan
from experiments.props_p3_data import validate as p3_validate
from experiments.props_p3_data.plan import PROMPT, validate_plan
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json
from experiments.xvla_single_task import contract as single_task
from experiments.xvla_single_task.contract import (
    make_config as _make_config, backward_microbatch,
    observation, check_model_inputs, inventory, source_path,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/props_p3_data/props_p3_20261020'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/xvla_trial/bundle'
OUTPUT = ROOT/'outputs/train/xvla_props_p3_v1'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
PROOF = OUTPUT.with_name(OUTPUT.name+'_resumeproof')
EQUIVALENCE = OUTPUT.with_name(OUTPUT.name+'_equivalence')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = '6430d34ea2a13c9c49a6621b0a2509350cc5e84c489209154367dc27cd151acf'
SEAL_RECORD = {
    'fingerprint': SEAL,
    'provenance_sha256': 'e1861e02b1f887256d5b64f465828afc22765f0b10d1fba82fb8fbcebfa328e0',
    'plan_sha256': '279dec81aa9b46f0ddd5b3e7e9cc7394f954c93613b9c58e54effb1cad3c7b46',
    'substitutions_sha256': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    'image': 'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e',
    'episodes': 544,
}
VALIDATION_RECORD = {'episodes': 544, 'train': 480, 'validation': 64, 'per_cell_train': 60,
                     'per_cell_validation': 8, 'objects': {'can': 136, 'milk': 136, 'bread': 136, 'lemon': 136},
                     'substituted_slots': 0, 'held_out_objects_collected': 0, 'prompt': PROMPT}
FRAMES_PER_EPISODE = 452
CHUNK_SIZE = 30
TRAIN, VAL, SEED = tuple(range(480)), tuple(range(480, 544)), 20261007
TRAINING_ROWS = 480*FRAMES_PER_EPISODE  # 216960
COMPLETE_STARTS = 480*(FRAMES_PER_EPISODE-CHUNK_SIZE+1)  # 203040, 452-30+1=423 starts/episode
VALIDATION_ROWS = 64*FRAMES_PER_EPISODE  # 28928
TOTAL_FRAMES = 544*FRAMES_PER_EPISODE  # 245888
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{key: (3, 512, 512) for key in CAMERAS}}

# Explicit consumed helpers; X-VLA base weights binding is via require_assets from xvla_trial.
# Do not call old recursive experiment scans or copy the SmolVLA adapter. The list is exact:
# `tests/test_fork_hygiene.py` asserts it equals the repo modules that importing the contract and
# the runner load (plus this package's tests and README).
SOURCE_FILES = (
    'experiments/xvla_trial/contract.py', 'experiments/xvla_trial/preflight.py',
    'experiments/xvla_trial/assets.py', 'experiments/xvla_single_task/contract.py',
    'experiments/props_p3_data/__init__.py', 'experiments/props_p3_data/plan.py',
    'experiments/props_p3_data/contract.py', 'experiments/props_p3_data/validate.py',
    'experiments/props_scene/__init__.py', 'experiments/props_scene/constants.py',
    'experiments/fixed_layout/__init__.py', 'experiments/fixed_layout/contract.py',
    'experiments/xvla_props_p3/__init__.py', 'experiments/xvla_props_p3/contract.py',
    'experiments/xvla_props_p3/runner.py',
    'experiments/xvla_props_p3/tests/test_contract.py',
    'experiments/xvla_props_p3/tests/test_runner.py',
    'experiments/xvla_props_p3/tests/test_fork_hygiene.py',
    'experiments/xvla_props_p3/README.md',
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

# The one instruction every P3 frame carries (the plan's fixed prompt). The expert's own
# "put the X in the Y bin" never reaches the dataset and is not approved here.
INSTRUCTIONS = (PROMPT,)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def planned_instructions():
    """Plan slot -> instruction, from the sealed plan.json; every slot must carry the fixed prompt."""
    plan = json.loads((PARENT/'plan.json').read_text())
    planned = tuple(episode['instruction'] for episode in plan['episodes'])
    if set(planned) != set(INSTRUCTIONS):
        raise ValueError(f'Plan instructions {sorted(set(planned))} are not exactly the fixed prompt')
    return planned


def task_index_map(ds):
    """The dataset's task table must be exactly {fixed prompt: 0}; with one task, index and text agree."""
    table = {str(k): int(v) for k, v in zip(ds.meta.tasks.index, ds.meta.tasks['task_index'])}
    if table != {PROMPT: 0}:
        raise ValueError(f'Dataset tasks {sorted(table)} are not exactly the fixed prompt')
    return table


@contextmanager
def approved_instruction(instruction):
    """Swap the approved instruction that `xvla_single_task.contract.observation` compares against.

    That module hardcodes a single `TASK` and is bound by several contracts, so it cannot be
    edited. The membership check keeps the guarantee the original provides: only a trained
    instruction is ever accepted (the expert's own "put the X in the Y bin" is not in `INSTRUCTIONS`).
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
        json.dump({'purpose': 'xvla_props_p3', 'resume': False, 'required_image': IMAGE}, stream)


def claim_resume(path):
    """Resume only the bind-mounted output that a fresh claim created; every resume leaves a token."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    claim = path/'run_claim.json'
    if not path.is_dir() or not is_mountpoint(path) or not claim.is_file():
        raise FileExistsError('Resume needs the original bind-mounted output holding run_claim.json')
    if json.loads(claim.read_text()) != {'purpose': 'xvla_props_p3', 'resume': False,
                                         'required_image': IMAGE}:
        raise ValueError('run_claim.json is not this package\'s claim')
    number = len(list(path.glob('resume_claim_*.json')))+1
    with (path/f'resume_claim_{number:02d}.json').open('x') as stream:
        json.dump({'purpose': 'xvla_props_p3', 'resume': number, 'required_image': IMAGE}, stream)
    return number


def require_fields(actual, expected, message):
    if any(not _same(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(message)


def indices(episodes):
    if len(episodes) != 544:
        raise ValueError('Exactly 544 props P3 episodes; other packages\' metadata forbidden')
    planned = planned_instructions()
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=FRAMES_PER_EPISODE, tasks=[planned[i]],
                       dataset_from_index=i*FRAMES_PER_EPISODE, dataset_to_index=(i+1)*FRAMES_PER_EPISODE),
                       'Props P3 episode identity/instruction/length/boundary mismatch')
    return (list(range(TRAINING_ROWS)),
            [i*FRAMES_PER_EPISODE+j for i in TRAIN for j in range(FRAMES_PER_EPISODE-CHUNK_SIZE+1)])


def selected_stats(rows, selected, task_of_episode):
    if selected != list(range(TRAINING_ROWS)):
        raise ValueError('Statistics require every training row exactly once')
    values = {key: [] for key in ('observation.state', 'action')}
    for i in selected:
        row = rows[i]
        if (int(row['episode_index']) != i//FRAMES_PER_EPISODE
                or int(row['task_index']) != task_of_episode[i//FRAMES_PER_EPISODE]):
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
    if (not isinstance(action, torch.Tensor) or action.shape != (CHUNK_SIZE, 8)
            or action.dtype != torch.float32 or not torch.isfinite(action).all()
            or not isinstance(pad, torch.Tensor) or pad.shape != (CHUNK_SIZE,)
            or pad.dtype != torch.bool or pad.any()):
        raise ValueError('Finite real chunk actions and complete unpadded chunk required')
    return {key: row[key] for key in (*OBS, 'task', 'action', 'action_is_pad')}


def collate_rows(rows):
    if len(rows) != 4:
        raise ValueError('Exactly four training rows per microbatch')
    return default_collate([policy_row(row, True) for row in rows])


def sample_batches(starts):
    expected = [i*FRAMES_PER_EPISODE+j for i in TRAIN for j in range(FRAMES_PER_EPISODE-CHUNK_SIZE+1)]
    if starts != expected:
        raise ValueError(f'Exactly {len(expected)} complete ordered training starts required')
    generator = torch.Generator().manual_seed(SEED)
    while True:
        order = torch.randperm(len(starts), generator=generator).tolist()
        for offset in range(0, len(order), 4):
            yield [starts[j] for j in order[offset:offset+4]]


def parent_fingerprint():
    """Full recomputation over every file in the parent; `--check` only (it reads ~1.1 GB)."""
    return p3_validate.fingerprint(PARENT)


def current_binding():
    """Cheap, repeatable rehash of the parent's small files, the props P3 sources and the assets.

    `provenance.json` must equal a fresh `props_p3_data.contract.provenance()` and hash to the
    digest the seal recorded, so editing any file the collection depended on is caught here.
    """
    if json.loads((PARENT/'seal.json').read_text()) != SEAL_RECORD:
        raise ValueError('Approved props P3 parent seal record changed')
    plan = json.loads((PARENT/'plan.json').read_text())
    validate_plan(plan)
    if p3_plan.digest(plan) != SEAL_RECORD['plan_sha256']:
        raise ValueError('Props P3 plan of record differs from the sealed plan digest')
    substitutions = json.loads((PARENT/'substitutions.json').read_text())
    if p3_plan.digest(substitutions) != SEAL_RECORD['substitutions_sha256']:
        raise ValueError('Props P3 substitutions are not the sealed record')
    recorded = json.loads((PARENT/'provenance.json').read_text())
    current = p3_contract.provenance()
    if recorded != current or p3_plan.digest(recorded) != SEAL_RECORD['provenance_sha256']:
        raise ValueError('Props P3 source provenance drifted from the sealed snapshot')
    validation = json.loads((PARENT/'validation.json').read_text())
    if set(validation) != set(VALIDATION_RECORD):
        raise ValueError('Props P3 validation record has other fields than the sealed one')
    require_fields(validation, VALIDATION_RECORD, 'Props P3 validation proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                plan_sha256=SEAL_RECORD['plan_sha256'], provenance_sha256=SEAL_RECORD['provenance_sha256'],
                validation_sha256=digest_json(validation), props_p3_provenance=current,
                sources=source_hashes(), assets=require_assets(), required_image=IMAGE)


def semantic_gate():
    """The collection's own semantic proof (counts, balance, no held-out object), bound by digest."""
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
    require_fields(info, dict(total_episodes=544, total_frames=TOTAL_FRAMES,
                   total_tasks=len(INSTRUCTIONS), fps=30, splits={'train': '0:544'}),
                   'Props P3 dataset metadata mismatch')
    ds = LeRobotDataset(repo_id=DATA.name, root=DATA,
                        delta_timestamps={'action': [i/30 for i in range(CHUNK_SIZE)]})
    features = dataset_to_policy_features(ds.features)
    if ({k: tuple(v.shape) for k, v in features.items() if k.startswith('observation.')} != OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != TOTAL_FRAMES):
        raise ValueError('Real dataset state/action/task schema mismatch')
    task_of_episode = tuple(task_index_map(ds)[x] for x in planned_instructions())
    rows, starts = indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    for i in range(TOTAL_FRAMES):
        if (int(ds.hf_dataset[i]['episode_index']) != i//FRAMES_PER_EPISODE
                or int(ds.hf_dataset[i]['task_index']) != task_of_episode[i//FRAMES_PER_EPISODE]):
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


# 203,040 complete starts x 5.0 epochs / effective batch 8 = 126,900 updates exactly (no rounding).
TOTAL_UPDATES = 126900
# Disk: 7 checkpoints (6x3.3 + 9.9 = 29.7 GB) plus the resume snapshot working set (2 kept + 1
# being written = 3x~10.5 GB = 31.5 GB) peaks at ~61.2 GB; the runner's disk guard is 65 GiB. (A
# first-pass plan estimate of 55 GB missed the transient third snapshot, the same arithmetic slip
# class as FAILURES #51/#52 — corrected here from direct computation, not copied forward.)
# Dense schedule from epoch 2: whole epochs plus half-epochs from 3.5 on, as P1 (same recipe).
# Not all land on a snapshot update.
CHECKPOINT_UPDATES = (1, 50760, 76140, 88830, 101520, 114210, 126900)
# Resume snapshot (model + Adam + RNG, ~10.5 GB, ~16 s to write) every 2,600 updates, as P1: about 20 min
# at the faster loop. Night batches pause at a snapshot, so this is also the pause granularity.
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
    return dict(purpose='xvla_props_p3_preflight' if preflight else 'xvla_props_p3',
                seed=SEED, max_optimizer_updates=2 if preflight else TOTAL_UPDATES,
                microbatch=4, accumulation=2, effective_batch=8, chunk_size=CHUNK_SIZE, n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4, vlm_scale=.1, soft_prompt_scale=1., weight_decay=1e-4,
                               vlm_weight_decay=1e-5, betas=[.9, .95], eps=1e-8, clip=10.),
                schedule=dict(warmup=500, total=TOTAL_UPDATES, floor_fraction=.1),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                optimizer_storage='final_only', resume='manual_from_verified_snapshot',
                snapshot_updates=snapshot_updates(preflight), keep_snapshots=KEEP_SNAPSHOTS,
                autotune=False, retries=False)


def checkpoint_updates(preflight):
    # Update 1 and the dense schedule above. Disk-bound: 6x3.3 + 9.9 = 29.7 GB plus snapshots.
    return [1, 2] if preflight else list(CHECKPOINT_UPDATES)


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


def validate_equivalence(report, binding):
    """The fast loop reproduced P1's loop bit for bit: per-update metrics, weights and Adam state."""
    require_fields(report, dict(status='complete', binding=binding, updates=30, metrics_equal=True,
                                model_state_equal=True, optimizer_state_equal=True),
                   'Byte-exact loop equivalence proof required before --train')


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
