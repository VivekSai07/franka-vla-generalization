"""Props P2 pi0.5 training contract: the sealed 540-episode distractor dataset, admission-pi image.

Forked in design from `experiments.xvla_props_p2` (its dataset binding is copied, not imported, so P2's file stays
untouched; `tests/test_p2_parity.py` checks the copy against the original on the real dataset). Training needs no
simulator and runs in `admission-pi` (existing admission image + the transformers fork pi0.5 needs, `docker/pi05/`).

What differs from the X-VLA contract:

* pi0.5's own recipe: chunk 50, quantile normalisation with train-only q01/q99, 224 x 224 images, its learning rate,
  warmup, betas, weight decay and clip. Named deviations (`recipe()['deviations']`): effective batch 8, the cosine decay
  spanning the whole run, 10 of 50 actions executed per inference.
* Exposure is 5.0 epochs over this model's complete 50-step windows, so the horizon is derived, never typed. A chunk of
  50 removes 20 more observation starts per episode than a chunk of 30; `exposure()` reports starts, samples and
  supervised action positions separately and the arms are not called identical in exposure.
* Processors are built locally from the config and the train-only statistics; the hub's preprocessor file is
  pinned (`assets.py`) but never loaded.

`experiments/props_p2_data/**` and everything in its `provenance()` are frozen for this run's lifetime.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.pi05_props_p2 import assets
from experiments.props_p2_data import contract as p2_contract
from experiments.props_p2_data import plan as p2_plan
from experiments.props_p2_data import validate as p2_validate
from experiments.props_p2_data.plan import TRAINED_PAIRS, instruction, validate_plan

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:a229f2fbe98f589caaca37068c1ba4a8761d111c6e4855b2360f783c1f290eae'
PARENT = ROOT/'outputs/props_p2_data/props_p2_20261013'
DATA = PARENT/'dataset'
OUTPUT = ROOT/'outputs/train/pi05_props_p2_v1'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
PROOF = OUTPUT.with_name(OUTPUT.name+'_resumeproof')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
PROBE = ROOT/'outputs/train/pi05_props_p2_probe'  # scratch of `--probe-memory`; never a gate
SEAL = 'b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898'
SEAL_RECORD = {
    'fingerprint': SEAL,
    'provenance_sha256': '955e80a65b28d3078186af9697ed9b4065bb299734c4239bd3dfec34eed7cc47',
    'plan_sha256': '372cbe743f60c5a06443296f4140882bcba057ba25d4faa5cde9534952d6cc64',
    'substitutions_sha256': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    'image': 'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e',
    'episodes': 540,
}
VALIDATION_RECORD = {'episodes': 540, 'train': 480, 'validation': 60, 'per_cell_per_count_train': 12,
                     'per_cell_validation': 3, 'distractor_counts': {'1': 270, '2': 270},
                     'substituted_slots': 0, 'held_out_pairs_collected': 0}
FRAMES_PER_EPISODE = 452
CHUNK_SIZE = 50
N_ACTION_STEPS = 10
EFFECTIVE_BATCH = 8
EPOCHS = 5.0
TRAIN, VAL, SEED = tuple(range(480)), tuple(range(480, 540)), 20261007
TRAINING_ROWS = 480*FRAMES_PER_EPISODE
VALIDATION_ROWS = 60*FRAMES_PER_EPISODE
TOTAL_FRAMES = 540*FRAMES_PER_EPISODE
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{key: (3, 512, 512) for key in CAMERAS}}
STATE_KEYS = (('observation.state', 9), ('action', 8))
# Fractions of the run (X-VLA's grid), plus update 1.
CHECKPOINT_FRACTIONS = (.4, .6, .7, .8, .9, 1.0)

# Explicit list of what is hashed; `tests/test_fork_hygiene.py` asserts the repo modules importing the contract loads are
# all listed and nothing else is. The evaluation adapter and its tests are deliberately absent.
SOURCE_FILES = (
    'experiments/props_p2_data/__init__.py', 'experiments/props_p2_data/plan.py',
    'experiments/props_p2_data/contract.py', 'experiments/props_p2_data/validate.py',
    'experiments/props_scene/__init__.py', 'experiments/props_scene/constants.py',
    'experiments/pi05_props_p2/__init__.py', 'experiments/pi05_props_p2/assets.py',
    'experiments/pi05_props_p2/contract.py', 'experiments/pi05_props_p2/weights.py',
    'experiments/pi05_props_p2/strategy.py', 'experiments/pi05_props_p2/runner.py',
    'experiments/pi05_props_p2/tests/test_assets.py', 'experiments/pi05_props_p2/tests/test_contract.py',
    'experiments/pi05_props_p2/tests/test_fork_hygiene.py', 'experiments/pi05_props_p2/tests/test_image.py',
    'experiments/pi05_props_p2/tests/test_p2_parity.py', 'experiments/pi05_props_p2/tests/test_processor.py',
    'experiments/pi05_props_p2/tests/test_weights_load.py', 'experiments/pi05_props_p2/tests/test_runner.py',
    'experiments/pi05_props_p2/tests/test_strategy.py', 'experiments/pi05_props_p2/tests/test_weights.py',
    'experiments/pi05_props_p2/tests/test_reference_path.py',
    'experiments/pi05_props_p2/README.md',
) + tuple('installed/lerobot/'+name for name in (
    'policies/pi05/__init__.py', 'policies/pi05/configuration_pi05.py', 'policies/pi05/modeling_pi05.py',
    'policies/pi05/processor_pi05.py', 'policies/rtc/configuration_rtc.py', 'policies/rtc/debug_tracker.py',
    'policies/rtc/modeling_rtc.py', 'policies/pretrained.py', 'policies/utils.py',
    'processor/__init__.py', 'processor/batch_processor.py', 'processor/converters.py', 'processor/core.py',
    'processor/delta_action_processor.py', 'processor/device_processor.py', 'processor/factory.py',
    'processor/gym_action_processor.py', 'processor/hil_processor.py', 'processor/normalize_processor.py',
    'processor/observation_processor.py', 'processor/pipeline.py', 'processor/policy_robot_bridge.py',
    'processor/rename_processor.py', 'processor/tokenizer_processor.py',
    'configs/policies.py', 'configs/types.py', 'optim/optimizers.py', 'optim/schedulers.py',
    'datasets/lerobot_dataset.py', 'datasets/utils.py', 'datasets/video_utils.py', 'datasets/compute_stats.py',
    'datasets/transforms.py', 'utils/constants.py',
))

INSTRUCTIONS = tuple(sorted(instruction(obj, bin_colour) for obj, bin_colour in TRAINED_PAIRS))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    return assets.sha256(path)


def source_path(name, root=ROOT):
    if name.startswith('installed/lerobot/'):
        import lerobot
        root, name = Path(lerobot.__file__).parent, name.removeprefix('installed/lerobot/')
    root = Path(root).resolve()
    relative = Path(name)
    path = root/relative
    if (relative.is_absolute() or '..' in relative.parts or not path.is_file()
            or any(x.is_symlink() for x in (path, *path.parents)) or root not in path.resolve().parents):
        raise ValueError('Bound file missing, symlinked, or outside allowed root: '+name)
    return path


def source_hashes():
    return {name: sha256(source_path(name)) for name in SOURCE_FILES}


def same(actual, expected):
    """Exact JSON types as well as values: False/0/0.0 and tuple/list differ."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys() and all(type(k) is str for k in actual)
                and all(same(actual[k], value) for k, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def require_fields(actual, expected, message):
    if any(not same(actual.get(key), value) for key, value in expected.items()):
        raise ValueError(message)


# --- instructions ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def planned_instructions():
    """Plan slot -> instruction, from the sealed plan.json; the dataset must agree episode for episode."""
    plan = json.loads((PARENT/'plan.json').read_text())
    return tuple(episode['instruction'] for episode in plan['episodes'])


def require_instruction(text):
    if text not in INSTRUCTIONS:
        raise ValueError(f'Unapproved instruction: {text!r}')


def task_index_map(ds):
    """LeRobot numbers tasks by first appearance, not by the plan's sorted order: match by text."""
    table = {str(k): int(v) for k, v in zip(ds.meta.tasks.index, ds.meta.tasks['task_index'])}
    if set(table) != set(INSTRUCTIONS) or sorted(table.values()) != list(range(len(INSTRUCTIONS))):
        raise ValueError(f'Dataset tasks {sorted(table)} are not a bijection over the planned instructions')
    return table


# --- windows, exposure, statistics ----------------------------------------------------------------------------

def windows():
    """Start frames of every complete CHUNK_SIZE-step window inside a training episode."""
    return [i*FRAMES_PER_EPISODE+j for i in TRAIN for j in range(FRAMES_PER_EPISODE-CHUNK_SIZE+1)]


TOTAL_UPDATES = round(EPOCHS*len(windows())/EFFECTIVE_BATCH)


def exposure():
    """Observation starts, samples and supervised action positions, next to X-VLA's, never conflated."""
    samples = TOTAL_UPDATES*EFFECTIVE_BATCH
    x_vla_starts = len(TRAIN)*(FRAMES_PER_EPISODE-30+1)
    x_vla_samples = round(EPOCHS*x_vla_starts)
    return dict(observation_starts_per_epoch=len(windows()), samples=samples,
                supervised_action_positions=samples*CHUNK_SIZE, chunk_size=CHUNK_SIZE,
                x_vla_observation_starts_per_epoch=x_vla_starts, x_vla_samples=x_vla_samples,
                x_vla_supervised_action_positions=x_vla_samples*30)


def indices(episodes):
    if len(episodes) != 540:
        raise ValueError('Exactly 540 props P2 episodes; other packages\' metadata forbidden')
    planned = planned_instructions()
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=FRAMES_PER_EPISODE, tasks=[planned[i]],
                       dataset_from_index=i*FRAMES_PER_EPISODE, dataset_to_index=(i+1)*FRAMES_PER_EPISODE),
                       'Props P2 episode identity/instruction/length/boundary mismatch')
    return list(range(TRAINING_ROWS)), windows()


def statistics(rows, selected, task_of_episode):
    """Train-only mean/std/min/max/q01/q99 of the state (9) and the action (8), one row at a time."""
    if selected != list(range(TRAINING_ROWS)):
        raise ValueError('Statistics require every training row exactly once')
    values = {key: [] for key, _ in STATE_KEYS}
    for i in selected:
        row = rows[i]
        if (int(row['episode_index']) != i//FRAMES_PER_EPISODE
                or int(row['task_index']) != task_of_episode[i//FRAMES_PER_EPISODE]):
            raise ValueError('Raw training membership/instruction mismatch')
        for key, size in STATE_KEYS:
            value = torch.as_tensor(row[key])
            if value.dtype != torch.float32 or value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('Finite unpadded real FP32 state9/action8 required')
            values[key].append(value)
    stats = {}
    for key, value in values.items():
        x = torch.stack(value)
        q = torch.quantile(x.double(), torch.tensor([.01, .99], dtype=torch.float64), dim=0).float()
        stats[key] = {'mean': x.mean(0), 'std': x.std(0, unbiased=False), 'min': x.amin(0), 'max': x.amax(0),
                      'q01': q[0], 'q99': q[1]}
    return stats


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


# --- parent binding -------------------------------------------------------------------------------------------

def semantic_gate():
    """The collection's own semantic proof (counts, balance, no held-out pair), bound by digest."""
    return json.loads((PARENT/'validation.json').read_text())


def parent_fingerprint():
    """Full recomputation over every file in the parent; `--check` only (it reads ~1.1 GB)."""
    return p2_validate.fingerprint(PARENT)


def current_binding():
    """Cheap, repeatable rehash of the parent's small files, the props P2 sources, this package and the assets."""
    if json.loads((PARENT/'seal.json').read_text()) != SEAL_RECORD:
        raise ValueError('Approved props P2 parent seal record changed')
    plan = json.loads((PARENT/'plan.json').read_text())
    validate_plan(plan)
    if p2_plan.digest(plan) != SEAL_RECORD['plan_sha256']:
        raise ValueError('Props P2 plan of record differs from the sealed plan digest')
    substitutions = json.loads((PARENT/'substitutions.json').read_text())
    if p2_plan.digest(substitutions) != SEAL_RECORD['substitutions_sha256']:
        raise ValueError('Props P2 substitutions are not the sealed record')
    recorded = json.loads((PARENT/'provenance.json').read_text())
    current = p2_contract.provenance()
    if recorded != current or p2_plan.digest(recorded) != SEAL_RECORD['provenance_sha256']:
        raise ValueError('Props P2 source provenance drifted from the sealed snapshot')
    validation = json.loads((PARENT/'validation.json').read_text())
    if set(validation) != set(VALIDATION_RECORD):
        raise ValueError('Props P2 validation record has other fields than the sealed one')
    require_fields(validation, VALIDATION_RECORD, 'Props P2 validation proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                plan_sha256=SEAL_RECORD['plan_sha256'], provenance_sha256=SEAL_RECORD['provenance_sha256'],
                validation_sha256=digest_json(validation), props_p2_provenance=current,
                sources=source_hashes(), assets=assets.require_assets(), required_image=IMAGE)


def load_data():
    """Dataset-only reader: the sealed rows, the plan's split, the train-only statistics."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    info = json.loads((DATA/'meta/info.json').read_text())
    # LeRobot's own split metadata calls every episode "train"; the real split is the plan's.
    require_fields(info, dict(total_episodes=540, total_frames=TOTAL_FRAMES,
                   total_tasks=len(INSTRUCTIONS), fps=30, splits={'train': '0:540'}),
                   'Props P2 dataset metadata mismatch')
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
    return ds, starts, statistics(ds.hf_dataset, rows, task_of_episode)


# --- recipe, config, processors -------------------------------------------------------------------------------

# D5's three candidates. FP32 on-GPU full tuning (57.9 GB before activations) cannot fit and is not listed.
STRATEGIES = {
    'bf16_all': dict(dtype='bfloat16', train_expert_only=False, gradient_checkpointing=True),
    'bf16_cpu_master': dict(dtype='bfloat16', train_expert_only=False, gradient_checkpointing=True),
    'expert_only': dict(dtype='bfloat16', train_expert_only=True, gradient_checkpointing=False),
}
KEEP_SNAPSHOTS = 1
SNAPSHOT_SLOTS = KEEP_SNAPSHOTS+1  # the kept ones plus the one being written
DISK_MARGIN_BYTES = 2*1024**3
DECISION_KEYS = ('strategy', 'microbatch', 'snapshot_every', 'checkpoint_bytes', 'snapshot_bytes',
                 'final_state_bytes', 'gpu_peak_bytes')
GPU_MARGIN_BYTES = 2*1024**3
# The user's decision after `--probe-memory`; None until it is made, and `--check` refuses while it is None.
DECISION = None


def validate_decision(decision):
    if not isinstance(decision, dict) or set(decision) != set(DECISION_KEYS):
        raise ValueError(f'Decision needs exactly {DECISION_KEYS}')
    if decision['strategy'] not in STRATEGIES:
        raise ValueError(f'Unknown memory strategy {decision["strategy"]!r}')
    if decision['microbatch'] not in (1, 2, 4, 8):
        raise ValueError('Microbatch must divide the effective batch of 8')
    for key in DECISION_KEYS[2:]:
        if type(decision[key]) is not int or decision[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    return decision


def gpu_requirement_bytes(decision=None):
    return (decision or require_decision())['gpu_peak_bytes']+GPU_MARGIN_BYTES


def require_decision():
    if DECISION is None:
        raise RuntimeError('The memory strategy is not decided: run --probe-memory and record the decision')
    return validate_decision(DECISION)


def accumulation(decision=None):
    return EFFECTIVE_BATCH//(decision or require_decision())['microbatch']


def snapshot_updates(preflight, decision=None):
    """Updates after which a resume snapshot is written; the final update has the final checkpoint."""
    every, maximum = (1, 2) if preflight else ((decision or require_decision())['snapshot_every'], TOTAL_UPDATES)
    return list(range(every, maximum, every))


def disk_requirement_bytes(preflight, decision=None):
    """Checkpoints, the final optimizer state, the kept snapshots plus the one being written, and a margin."""
    d = decision or require_decision()
    return (len(checkpoint_updates(preflight))*d['checkpoint_bytes'] + d['final_state_bytes']
            + SNAPSHOT_SLOTS*d['snapshot_bytes'] + DISK_MARGIN_BYTES)


def learning_rate(update):
    """LeRobot's cosine-with-warmup lambda at scheduler step `update - 1` (its scheduler steps after each update)."""
    if type(update) is not int or not 1 <= update <= TOTAL_UPDATES:
        raise ValueError(f'Update must be integer in [1,{TOTAL_UPDATES}]')
    peak, decay, warmup = 2.5e-5, 2.5e-6, 1000
    step = update-1
    if step < warmup:
        return peak*((1/(warmup+1)-1)*(1-step/warmup)+1)
    alpha = decay/peak
    return peak*((1-alpha)*.5*(1+math.cos(math.pi*min(step, TOTAL_UPDATES)/TOTAL_UPDATES))+alpha)


def set_learning_rate(optimizer, update, lr=None):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(update) if lr is None else lr


def checkpoint_updates(preflight):
    return [1, 2] if preflight else [1] + [round(f*TOTAL_UPDATES) for f in CHECKPOINT_FRACTIONS]


def recipe(preflight):
    return dict(purpose='pi05_props_p2_preflight' if preflight else 'pi05_props_p2',
                seed=SEED, max_optimizer_updates=2 if preflight else TOTAL_UPDATES,
                effective_batch=EFFECTIVE_BATCH, chunk_size=CHUNK_SIZE, n_action_steps=N_ACTION_STEPS,
                optimizer=dict(peak_lr=2.5e-5, betas=[.9, .95], eps=1e-8, weight_decay=.01, clip=1.0),
                schedule=dict(warmup=1000, total=TOTAL_UPDATES, decay_lr=2.5e-6),
                normalization='quantiles_train_only_q01_q99_no_clipping',
                deviations=['effective_batch_8', 'decay_spans_run', 'n_action_steps_10'],
                train_episodes=list(TRAIN), validation_episodes=list(VAL), exposure=exposure(), decision=DECISION)


def make_config(device, strategy=None):
    """The spec's config; `strategy` (a key of STRATEGIES) adds only the dtype/freezing/checkpointing it names."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    extra = {} if strategy is None else dict(STRATEGIES[strategy])
    return PI05Config(
        device=device, chunk_size=CHUNK_SIZE, n_action_steps=N_ACTION_STEPS, scheduler_decay_steps=TOTAL_UPDATES,
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (9,)),
                        **{key: PolicyFeature(FeatureType.VISUAL, (3, 512, 512)) for key in CAMERAS}},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))}, **extra)


@contextmanager
def local_tokenizer():
    """The installed factory names the tokenizer by hub id; point it at the pinned local files, restore after."""
    import lerobot.policies.pi05.processor_pi05 as module
    original = module.TokenizerProcessorStep

    def build(**kwargs):
        if kwargs.get('tokenizer_name') != assets.TOKENIZER_REPO:
            raise ValueError('Installed factory asks for another tokenizer than the pinned one')
        return original(**{**kwargs, 'tokenizer_name': str(assets.TOKENIZER_DIR)})
    module.TokenizerProcessorStep = build
    try:
        yield
    finally:
        module.TokenizerProcessorStep = original


def make_processors(config, stats):
    """Pre- and post-processor from the local config and train-only statistics, never from the hub's json."""
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    with local_tokenizer():
        return make_pi05_pre_post_processors(config, stats)


# --- rows and sampling ----------------------------------------------------------------------------------------

def policy_row(row, training=True):
    """One dataset row as the policy may see it: approved episode and instruction, whitelisted keys, real chunk."""
    ids = TRAIN if training else TRAIN+VAL
    episode = row.get('episode_index')
    if isinstance(episode, torch.Tensor):
        if episode.ndim != 0 or episode.dtype != torch.int64:
            raise ValueError('Episode identity must be a scalar integer')
        episode = episode.item()
    if (type(episode) is not int or episode not in ids or row.get('task') != planned_instructions()[episode]
            or {key for key in row if key.startswith('observation.')} != set(OBS)):
        raise ValueError('Unapproved episode/instruction/observation')
    require_instruction(row['task'])
    for key, shape in OBS.items():
        value = row[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError(f'{key}: finite FP32 tensor of shape {shape} required')
    for key in CAMERAS:
        if row[key].min() < 0 or row[key].max() > 1:
            raise ValueError(f'{key} must lie in [0, 1]')
    action, pad = row.get('action'), row.get('action_is_pad')
    if (not isinstance(action, torch.Tensor) or tuple(action.shape) != (CHUNK_SIZE, 8)
            or action.dtype != torch.float32 or not torch.isfinite(action).all()
            or not isinstance(pad, torch.Tensor) or tuple(pad.shape) != (CHUNK_SIZE,)
            or pad.dtype != torch.bool or pad.any()):
        raise ValueError('Finite real chunk actions and a complete unpadded chunk required')
    return {key: row[key] for key in (*OBS, 'task', 'action', 'action_is_pad')}


def collate_rows(rows):
    if not 1 <= len(rows) <= EFFECTIVE_BATCH:
        raise ValueError('A microbatch holds between 1 and the effective batch of rows')
    return default_collate([policy_row(row, True) for row in rows])


def sample_updates(starts):
    """Endless stream of effective-batch groups of window starts, a fresh permutation per epoch, fixed by SEED."""
    if starts != windows():
        raise ValueError(f'Exactly {len(windows())} complete ordered training windows required')
    if len(starts) % EFFECTIVE_BATCH:
        raise ValueError('An epoch must be a whole number of updates')
    generator = torch.Generator().manual_seed(SEED)
    while True:
        order = torch.randperm(len(starts), generator=generator).tolist()
        for offset in range(0, len(order), EFFECTIVE_BATCH):
            yield [starts[j] for j in order[offset:offset+EFFECTIVE_BATCH]]


# --- claims, checkpoints and gate proofs ----------------------------------------------------------------------

def inventory(directory):
    directory = Path(directory)
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError('Artifact directory symlinks forbidden')
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Artifact symlinks forbidden')
        if path.is_file():
            result[str(path.relative_to(directory))] = sha256(path)
    if not result:
        raise ValueError('Artifact directory is empty')
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def claim_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    path.mkdir(parents=True, exist_ok=False)


def is_mountpoint(path):
    return str(path) in {line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()}


def claim_output(path):
    """Exclusive token; an existing path is allowed only as a newly created, empty Docker output bind mount."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Preserve existing output; no overwrite; resume only via --resume')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose': 'pi05_props_p2', 'resume': False, 'required_image': IMAGE}, stream)


def claim_resume(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    claim = path/'run_claim.json'
    if not path.is_dir() or not is_mountpoint(path) or not claim.is_file():
        raise FileExistsError('Resume needs the original bind-mounted output holding run_claim.json')
    if json.loads(claim.read_text()) != {'purpose': 'pi05_props_p2', 'resume': False, 'required_image': IMAGE}:
        raise ValueError('run_claim.json is not this package\'s claim')
    number = len(list(path.glob('resume_claim_*.json')))+1
    with (path/f'resume_claim_{number:02d}.json').open('x') as stream:
        json.dump({'purpose': 'pi05_props_p2', 'resume': number, 'required_image': IMAGE}, stream)
    return number


def checkpoint_directory(update):
    if type(update) is not int or update < 1:
        raise ValueError('Positive integer update required')
    return f'step_{update:06d}'


def require_marker(marker, update):
    require_fields(marker, dict(status='complete', update=update, reload_weights_exact=True),
                   'Strict checkpoint marker required')
    error = marker.get('reload_max_action_error')
    if type(error) not in (int, float) or not 0 <= error <= 1e-5:
        raise ValueError('Absolute reload action error must be finite and <=1e-5')


def cpu_proof(binding, semantic, stats, processor, parent_fingerprint):
    return dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                semantic_gate=semantic, training_rows=TRAINING_ROWS, complete_training_windows=len(windows()),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                parent_fingerprint_recomputed=parent_fingerprint == SEAL,
                training_stats=stats, training_stats_digest=digest_json(stats),
                processor_check=processor, processor_digest=digest_json(processor))


def validate_cpu(report, binding):
    require_fields(report, dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                   training_rows=TRAINING_ROWS, complete_training_windows=len(windows()),
                   train_episodes=list(TRAIN), validation_episodes=list(VAL), parent_fingerprint_recomputed=True),
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


def verify_checkpoints(report, output, preflight):
    updates = checkpoint_updates(preflight)
    keys = [checkpoint_directory(i) for i in updates]
    if [m.get('update') for m in report.get('checkpoints', [])] != updates:
        raise ValueError('Checkpoint schedule changed')
    if set(report.get('checkpoint_files', {})) != set(keys):
        raise ValueError('All checkpoint inventories required')
    if {p.name for p in (output/'checkpoints').iterdir()} != set(keys):
        raise ValueError('Unlisted/missing checkpoint')
    for marker in report['checkpoints']:
        update = marker['update']
        require_marker(marker, update)
        directory = output/'checkpoints'/checkpoint_directory(update)
        files = inventory(directory)
        required = {'model.safetensors', 'config.json', 'checkpoint_result.json'}
        if update == updates[-1]:
            required |= {'training_state.pt'}
        if (files != report['checkpoint_files'][checkpoint_directory(update)] or not required <= files.keys()
                or files['model.safetensors'] != marker.get('model_sha256')
                or json.loads((directory/'checkpoint_result.json').read_text()) != marker):
            raise ValueError('Checkpoint byte inventory changed')
        if update != updates[-1] and 'training_state.pt' in files:
            raise ValueError('Intermediate checkpoint must not persist optimizer state')


def validate_scratch(report, binding, output, stats, cpu_hash):
    require_fields(report, {**recipe(True), 'status': 'complete', 'completed_updates': 2, 'binding': binding,
                   'training_rows': TRAINING_ROWS, 'complete_training_windows': len(windows()),
                   'cpu_admission_sha256': cpu_hash, 'final_data_binding_passed': True,
                   'selected_checkpoint_update': None, 'rollouts': 0},
                   'Exact current two-update scratch proof required')
    if (report.get('training_stats_sha256') != sha256(output/'training_stats.json')
            or json.loads((output/'training_stats.json').read_text()) != stats):
        raise ValueError('Scratch training statistics changed')
    verify_checkpoints(report, output, True)


def validate_resume_proof(report, binding, scratch_report, output):
    """A resumed run reproduced the uninterrupted run's second update bit for bit."""
    require_fields(report, {**recipe(True), 'status': 'complete', 'completed_updates': 2, 'binding': binding,
                   'final_data_binding_passed': True, 'rollouts': 0}, 'Exact current resume proof required')
    scratch_model = scratch_report['checkpoints'][-1]['model_sha256']
    require_fields(report.get('resume_proof') or {},
                   dict(from_update=1, compared_update=2, metrics_equal=True, model_sha256_equal=True,
                        training_state_sha256_equal=True, scratch_model_sha256=scratch_model),
                   'Resumed run must reproduce the uninterrupted second update exactly')
    if report['checkpoints'][-1]['model_sha256'] != scratch_model:
        raise ValueError('Resumed final weights differ from the uninterrupted run')
    if report.get('training_stats_sha256') != sha256(output/'training_stats.json'):
        raise ValueError('Resume-proof training statistics changed')
    verify_checkpoints(report, output, True)
