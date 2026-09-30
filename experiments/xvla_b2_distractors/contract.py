"""B2 distractor 200-episode data, X-VLA train-only statistics and phase admission contracts.

Forked from `experiments.xvla_b1_language.contract` against the sealed parent
`outputs/b2_distractor_data/b2_distractor_20260928`. The two-instruction handling (task index
map, `approved_instruction`, per-episode instruction checks) is inherited **unchanged**: B2's
plan also carries two instructions as background randomisation across the same four cells, so
every check B1 needed for that axis already holds here with no code change.

What actually differs:

* The horizon is uncertain rather than doubled by a specific ratio. B2's finest per-condition
  split (128 distractor-present training episodes / 4 cells = 32/cell) is only ~20% thinner than
  B1's 40/cell, which is not itself evidence that B1's 33,400 updates suffices — the design spec
  treats it as a floor to test, not a target. `TOTAL_UPDATES` = 50,000, extending 25%/50% past
  B1's own winning checkpoint (`B1_WINNER_UPDATES` = 33,400) so the curve can be read past it if
  it is still climbing.
* Checkpoints at B1's own schedule (`B1_CALIBRATION_UPDATES` = 16,700, and 33,400) are kept so
  B2 can be compared against B1's screen numbers at those identical updates — do B2's scores land
  visibly below, confirming distractors make the task harder at matched update count?

`experiments/b2_distractor_data/**/*.py` is frozen for the lifetime of this training run.
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
from experiments.b2_distractor_data.plan import INSTRUCTIONS, make_plan, validate_plan
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json
from experiments.xvla_single_task import contract as single_task
from experiments.xvla_single_task.contract import (
    make_config as _make_config, backward_microbatch,
    observation, check_model_inputs, inventory, source_path, verify_listed,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/b2_distractor_data/b2_distractor_20260928'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/xvla_trial/bundle'
OUTPUT = ROOT/'outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = 'b6cfff3f27be92ba6fe12fef406cb481c384a521c9d2aeade41aa25da0c12340'  # injected from the sealed parent
TRAIN, VAL, SEED = tuple(range(160)), tuple(range(160, 200)), 20260917
TRAINING_ROWS = 160*528  # 84480
COMPLETE_STARTS = 160*499  # 79840, X-VLA chunk_size=30 → 528-30+1=499 complete starts per episode
VALIDATION_ROWS = 40*528  # 21120
TOTAL_FRAMES = 200*528  # 105600
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
    'experiments/b2_distractor_data/__init__.py', 'experiments/b2_distractor_data/plan.py',
    'experiments/b2_distractor_data/record.py', 'experiments/b2_distractor_data/scenes.py',
    'experiments/b2_distractor_data/validate.py', 'experiments/b2_distractor_data/expert.py',
    'experiments/local_position_probe/scenes.py', 'experiments/local_position_probe/contract.py',
    'experiments/xvla_b2_distractors/__init__.py', 'experiments/xvla_b2_distractors/contract.py',
    'experiments/xvla_b2_distractors/runner.py',
    'experiments/xvla_b2_distractors/tests/test_contract.py',
    'experiments/xvla_b2_distractors/tests/test_runner.py',
    'experiments/xvla_b2_distractors/README.md',
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
    """Plan slot -> instruction; the dataset must agree episode for episode."""
    return tuple(episode['instruction'] for episode in make_plan()['episodes'])


def task_index_map(ds):
    """LeRobot numbers tasks by first appearance, not by the plan's sorted order."""
    table = {str(k): int(v) for k, v in zip(ds.meta.tasks.index, ds.meta.tasks['task_index'])}
    if set(table) != set(INSTRUCTIONS) or sorted(table.values()) != list(range(len(INSTRUCTIONS))):
        raise ValueError(f'Dataset tasks {sorted(table)} are not a bijection over the planned instructions')
    return table


@contextmanager
def approved_instruction(instruction):
    """Swap the approved instruction that `xvla_single_task.contract.observation` compares against.

    That module hardcodes a single `TASK` and is bound by four contracts (fixed_layout,
    local_varied, broad_workspace and this one), so it cannot be edited. The membership check
    keeps the guarantee the original provides: only a planned instruction is ever accepted.
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
            raise FileExistsError('Preserve existing output; no overwrite/resume/retry')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose': 'xvla_b2_distractors', 'resume': False, 'required_image': IMAGE}, stream)


def require_fields(actual, expected, message):
    if any(not _same(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(message)


def indices(episodes):
    if len(episodes) != 200:
        raise ValueError('Exactly 200 B2 episodes; other packages\' metadata forbidden')
    planned = planned_instructions()
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=528, tasks=[planned[i]],
                       dataset_from_index=i*528, dataset_to_index=(i+1)*528),
                       'B2 episode identity/instruction/length/boundary mismatch')
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


def current_binding():
    """Rehash full parent + actual asset bytes + verify provenance against the seal.

    The recorded snapshot matches a fresh `provenance()` exactly, so the comparison the
    local-varied contract had to waive is enforced here. This freezes
    `experiments/b2_distractor_data/**/*.py` for the run's lifetime.
    """
    from experiments.b2_distractor_data.scenes import provenance
    if SEAL == 'PENDING_B2_ADMISSION':
        raise ValueError('Parent seal not yet injected; run admission before training')
    if json.loads((PARENT/'seal.json').read_text()) != {'fingerprint': SEAL}:
        raise ValueError('Approved B2 parent fingerprint changed')
    plan = json.loads((PARENT/'plan.json').read_text())
    validate_plan(plan)
    semantic = json.loads((PARENT/'validation.json').read_text())
    current = provenance()
    if semantic.get('provenance') != current:
        raise ValueError('B2 distractor source provenance drifted from the sealed snapshot')
    if (semantic.get('status') != 'passed' or semantic.get('training_ready') is not True
            or semantic.get('automatic_training') is not False
            or semantic.get('train_ids') != list(TRAIN)
            or semantic.get('validation_ids') != list(VAL)
            or semantic.get('statistics_episode_ids') != list(TRAIN)):
        raise ValueError('B2 distractor semantic proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                semantic_sha256=digest_json(semantic), b2_distractor_provenance=current,
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
    require_fields(info, dict(total_episodes=200, total_frames=TOTAL_FRAMES,
                   total_tasks=len(INSTRUCTIONS), fps=30,
                   splits={'train': '0:160', 'validation': '160:200'}),
                   'B2 dataset metadata mismatch')
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


# Budget is uncertain, not doubled by a specific ratio (see module docstring): B2's finest
# per-condition split (128 distractor-present episodes / 4 cells = 32/cell) is only ~20% thinner
# than B1's 40/cell, which does not by itself prove B1's horizon suffices for a harder task.
# B1_CALIBRATION_UPDATES/B1_WINNER_UPDATES are B1's own schedule points, kept so B2 can be
# compared against B1's screen numbers at identical update counts.
B1_CALIBRATION_UPDATES = 16700
B1_WINNER_UPDATES = 33400
TOTAL_UPDATES = 50000


def learning_rate(update):
    """Same shape as `xvla_single_task.contract.learning_rate`, stretched to the longer horizon.

    The shared helper hard-caps `update` at 5000 and `make_config` fixes `scheduler_decay_steps`
    at 5000, so a longer run would raise at update 5001. That file is bound by four contracts
    (fixed_layout, local_varied, broad_workspace and this one), so it cannot be edited without
    invalidating three sealed runs. The schedule is therefore redefined locally: identical 500-update
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
    return dict(purpose='xvla_b2_distractors_preflight' if preflight else 'xvla_b2_distractors',
                seed=SEED, max_optimizer_updates=2 if preflight else TOTAL_UPDATES,
                microbatch=4, accumulation=2, effective_batch=8, chunk_size=30, n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4, vlm_scale=.1, soft_prompt_scale=1., weight_decay=1e-4,
                               vlm_weight_decay=1e-5, betas=[.9, .95], eps=1e-8, clip=10.),
                schedule=dict(warmup=500, total=TOTAL_UPDATES, floor_fraction=.1),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                optimizer_storage='final_only', resume=False, autotune=False, retries=False)


def checkpoint_updates(preflight):
    # Disk-bound: ~3.3 GB each plus ~9.9 GB on the final one (optimizer state), against ~194 GB
    # free once collection lands. 16700 and 33400 are B1's own schedule points, kept for direct
    # comparison; 25000 fills the same gap B1 screened; 41700/50000 extend 25%/50% past B1's
    # winner because B2's budget is a floor to test, not an assumed target (see module docstring).
    return [1, 2] if preflight else [1, B1_CALIBRATION_UPDATES, 25000, B1_WINNER_UPDATES, 41700, TOTAL_UPDATES]


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
