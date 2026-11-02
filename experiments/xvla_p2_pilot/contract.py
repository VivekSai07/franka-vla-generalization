"""X-VLA P2 continuation pilots: 10,000 updates from the bundle's checkpoint 101,520, one extra-state variant per run.

A pilot is a screen (docs/superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md), never a reported result. The
data, the sampler, the optimizer settings and the loop are those of `experiments.xvla_p2_force` (M2, frozen, imported
read-only: its constants, helpers and the force transform); what differs is the start and the schedule:

* **Start:** the bundle's P2 checkpoint 101,520 (`START`), hash-pinned file by file, loaded with the strict FP32 loader, with a
  *fresh* AdamW (the bundle keeps weights only). The action statistics the processors normalise with must equal the ones
  saved in that checkpoint, so the continuation lives in the same action space.
* **Schedule:** linear warm-up over 200 updates to 1e-5 (the floor of P2's own schedule; the VLM group runs at a tenth), then
  constant, 10,000 updates at effective batch 8 (80,000 samples, 0.39 epoch of P2's 203,040 windows). The sample order is
  M2's `sample_batches` (fixed seed), the same for every variant.
* **Variants are data:** `variants/<name>.json` names a state builder from `BUILDERS` and a loss term (none exist yet: a loss
  term is code and needs a package version). A variant file is hashed into the run record but is not in `SOURCE_FILES`, so
  adding one never edits what a run is bound to. The robot state is always 19 values, 9 joint positions then 10 extras
  padded with zeros: `control` has no extras (all zero), `force` M2's 8 standardised wrench/finger values, `box` S0's
  `target_box` (2 cameras x u_min, v_min, u_max, v_max, in_frame, raw). The model pads proprioception to 20 slots with
  zeros whichever variant, so `control` equals P2's input byte for byte and `force` equals M2's.
* **Output:** the final weights only (`checkpoints/step_010000`), no snapshots and no optimizer state: a crashed pilot is
  re-run, and it is deterministic.

The rollout side (`eval_adapter.py`, `eval_tests/`, `variants/`) is deliberately not in `SOURCE_FILES`: fixing an evaluation
adapter after training must not invalidate the training binding.
"""
import hashlib
import json
import math
import re
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from experiments.fixed_layout.contract import IMAGE
from experiments.props_p2_data import contract as p2_contract
from experiments.props_p2_data import plan as p2_plan
from experiments.props_p2_data.plan import validate_plan
from experiments.vla_inputs_signals import sidecar as s0_sidecar
from experiments.xvla_p2_force import contract as m2
from experiments.xvla_p2_force import forces
from experiments.xvla_single_task import contract as single_task
from experiments.xvla_single_task.contract import backward_microbatch, inventory, source_path
from experiments.xvla_single_task.contract import make_config as _make_config
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
VARIANTS = PACKAGE/'variants'
RECIPE_VERSION = 'pilot_v1'

# The dataset, its split, the sampler and the instructions are M2's (P2's): imported, never retyped.
PARENT, DATA, CACHE = m2.PARENT, m2.DATA, m2.CACHE
SEAL, SEAL_RECORD, VALIDATION_RECORD = m2.SEAL, m2.SEAL_RECORD, m2.VALIDATION_RECORD
FRAMES_PER_EPISODE, CHUNK_SIZE, SEED = m2.FRAMES_PER_EPISODE, m2.CHUNK_SIZE, m2.SEED
TRAIN, VAL = m2.TRAIN, m2.VAL
TRAINING_ROWS, COMPLETE_STARTS, TOTAL_FRAMES = m2.TRAINING_ROWS, m2.COMPLETE_STARTS, m2.TOTAL_FRAMES
CAMERAS, INSTRUCTIONS = m2.CAMERAS, m2.INSTRUCTIONS
EPISODES = 540
planned_instructions, task_index_map, approved_instruction = (
    m2.planned_instructions, m2.task_index_map, m2.approved_instruction)
indices, sample_batches = m2.indices, m2.sample_batches
require_fields, digest_json, is_mountpoint = m2.require_fields, m2.digest_json, m2.is_mountpoint
parent_fingerprint, semantic_gate = m2.parent_fingerprint, m2.semantic_gate

# The pilot recipe (fixed: docs/superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md).
UPDATES, WARMUP, LR = 10_000, 200, 1e-5
JOINT_DIM, MAX_EXTRAS = 9, 10
STATE_WIDTH = JOINT_DIM+MAX_EXTRAS  # 19; the model pads proprioception to 20
DATASET_OBS = {'observation.state': (JOINT_DIM,), **{key: (3, 512, 512) for key in CAMERAS}}
OBS = {'observation.state': (STATE_WIDTH,), **{key: (3, 512, 512) for key in CAMERAS}}
def checkpoint_directory(updates=UPDATES):
    if type(updates) is not int or not 1 <= updates <= UPDATES:
        raise ValueError(f'Updates must be an integer in [1,{UPDATES}]')
    return f'step_{updates:06d}'


CHECKPOINT = checkpoint_directory(UPDATES)
CHECKPOINT_FILES = {'model.safetensors', 'config.json', 'checkpoint_result.json',
                    'policy_preprocessor.json', 'policy_postprocessor.json',
                    'policy_preprocessor_step_7_normalizer_processor.safetensors',
                    'policy_postprocessor_step_0_unnormalizer_processor.safetensors'}

# The start: the bundle's P2 checkpoint 101,520 (weights only), pinned file by file from A0's prepared record.
START_UPDATE = 101_520
START = ROOT/'outputs/downloads_20261017/props_p2/checkpoints/step_101520'
START_MODEL_SHA256 = '59d02d0a578f3b70590534837f65b802764e68d02e05ae513c334ed99975932e'
START_FILES = {
    'checkpoint_result.json': '855f4ab517d8b60950b385beaaf23404169286e365297e1b1c05ecb3bfcc6408',
    'config.json': '3cf7c26be80728faaf787d33b270e216fd37c6b6116ab709cffd0cbea94d24a4',
    'policy_postprocessor.json': 'baa5bcae49df96c477ea99da386588471c0a76e3d51f6d74859451cd1e743032',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors':
        'd589904817bf9edebee527f9c85b395e36e9b48e332c23e5c7e46f0a6c316ac9',
    'policy_preprocessor.json': '128c6ee39e84e13d40fd25cdb8006186ab3de761bfa237881bcdca200af8c6dd',
    'policy_preprocessor_step_7_normalizer_processor.safetensors':
        'd589904817bf9edebee527f9c85b395e36e9b48e332c23e5c7e46f0a6c316ac9',
}
START_NORMALISER = 'policy_preprocessor_step_7_normalizer_processor.safetensors'

# The box table of the sealed S0 sidecar as training reads it: (540, 452, 10) float32, pinned like M2's constants.
BOX_TABLE_SHA256 = '2475c91ef951c1fe2fd2ea60feea2008c7aa260bfdf0940b7564c010d72dfbfe'

# Explicit consumed helpers. The list is exact: `tests/test_fork_hygiene.py` asserts it equals the repo modules that
# importing the contract and the runner load (plus this package's tests and README).
SOURCE_FILES = (
    'experiments/xvla_trial/contract.py', 'experiments/xvla_trial/preflight.py',
    'experiments/xvla_trial/assets.py', 'experiments/xvla_single_task/contract.py',
    'experiments/props_p2_data/__init__.py', 'experiments/props_p2_data/plan.py',
    'experiments/props_p2_data/contract.py', 'experiments/props_p2_data/validate.py',
    'experiments/props_scene/__init__.py', 'experiments/props_scene/constants.py',
    'experiments/fixed_layout/__init__.py', 'experiments/fixed_layout/contract.py',
    'experiments/vla_inputs_signals/__init__.py', 'experiments/vla_inputs_signals/contract.py',
    'experiments/vla_inputs_signals/sidecar.py',
    'experiments/xvla_p2_force/__init__.py', 'experiments/xvla_p2_force/contract.py',
    'experiments/xvla_p2_force/forces.py',
    'experiments/xvla_p2_pilot/__init__.py', 'experiments/xvla_p2_pilot/contract.py',
    'experiments/xvla_p2_pilot/runner.py',
    'experiments/xvla_p2_pilot/tests/test_contract.py',
    'experiments/xvla_p2_pilot/tests/test_runner.py',
    'experiments/xvla_p2_pilot/tests/test_fork_hygiene.py',
    'experiments/xvla_p2_pilot/README.md',
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


def source_hashes():
    return {name: sha256(source_path(name)) for name in SOURCE_FILES}


# --- the extra-state builders -------------------------------------------------------------------

Builder = namedtuple('Builder', 'name width table')


def _frozen(array):
    array.flags.writeable = False
    return array


def table_digest(table):
    """sha256 over the shape and the exact bytes of a float32 table."""
    table = np.ascontiguousarray(table, dtype=np.float32)
    return hashlib.sha256(json.dumps(list(table.shape)).encode()+table.tobytes()).hexdigest()


def load_boxes(root=forces.SIDECAR, verify=True):
    """(540, 452, 10) float64: each sealed frame's `target_box`, front camera then gripper camera, (u_min, v_min, u_max,
    v_max, in_frame) each, as S0 recorded them. Every episode file is checked by S0's own validator."""
    root = Path(root)
    if verify:
        forces.verify_sidecar(root)
    names = sorted(path.name for path in (root/'episodes').iterdir())
    if names != [f'episode_{i:03d}.npz' for i in range(EPISODES)]:
        raise ValueError('Sidecar episodes are missing, duplicated or unexpected')
    values = np.empty((EPISODES, FRAMES_PER_EPISODE, MAX_EXTRAS))
    for index, name in enumerate(names):
        with np.load(root/'episodes'/name, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        s0_sidecar.check_arrays(arrays, index, FRAMES_PER_EPISODE)
        box = np.asarray(arrays['target_box'], dtype=np.float64)
        if box.shape != (FRAMES_PER_EPISODE, 2, 5) or not np.isfinite(box).all():
            raise ValueError(f'Sidecar episode {index} has no finite (452, 2, 5) target box')
        values[index] = box.reshape(FRAMES_PER_EPISODE, MAX_EXTRAS)
    return values


@lru_cache(maxsize=1)
def _box_table():
    table = _frozen(load_boxes().astype(np.float32))
    if table_digest(table) != BOX_TABLE_SHA256:
        raise ValueError('The target-box table differs from the pinned one')
    return table


@lru_cache(maxsize=1)
def _control_table():
    return _frozen(np.zeros((EPISODES, FRAMES_PER_EPISODE, 0), dtype=np.float32))


BUILDERS = {
    'control': Builder('control', 0, _control_table),
    'force': Builder('force', forces.N_EXTRA, forces.table),
    'box': Builder('box', MAX_EXTRAS, _box_table),
}


def builder_record(builder):
    """What the run record says about a builder: its width and exactly what its values come from."""
    record = dict(name=builder.name, width=builder.width, state_width=STATE_WIDTH)
    if builder.name == 'force':
        record.update(transform_source={'experiments/xvla_p2_force/forces.py':
                                        sha256(source_path('experiments/xvla_p2_force/forces.py'))},
                      constants_sha256=forces.constants_digest(forces.constants()), names=list(forces.EXTRA_NAMES))
    elif builder.name == 'box':
        record.update(source='S0 target_box, raw, front camera then gripper camera', table_sha256=table_digest(builder.table()))
    return record


# --- variants (data, hashed into the run record) -----------------------------------------------

VARIANT_SCHEMA = 'vla_inputs_pilot_variant_v1'
VARIANT_KEYS = {'schema', 'name', 'state_builder', 'loss_term', 'loss_weight', 'note'}
LOSS_TERMS = (None,)  # a loss term is code: it needs a package version, not a data file
_NAME = re.compile(r'[a-z][a-z0-9_]{0,31}')


def _object(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate key in a variant file')
    return dict(pairs)


def _constant(value):
    raise ValueError(f'{value} is not allowed in a variant file')


def check_variant_spec(spec, name):
    if not isinstance(spec, dict) or set(spec) != VARIANT_KEYS:
        raise ValueError(f'A variant file has exactly the keys {sorted(VARIANT_KEYS)}')
    if spec['schema'] != VARIANT_SCHEMA:
        raise ValueError(f'Variant schema must be {VARIANT_SCHEMA!r}')
    if spec['name'] != name or not isinstance(spec['note'], str):
        raise ValueError('Variant name must equal its file name and its note must be text')
    if spec['state_builder'] not in BUILDERS:
        raise ValueError(f'Unknown state builder {spec["state_builder"]!r}; known: {sorted(BUILDERS)}')
    if spec['loss_term'] not in LOSS_TERMS:
        raise ValueError('No loss terms exist in this package version; a new one needs a new package version')
    weight = spec['loss_weight']
    if type(weight) not in (int, float) or weight != 0:
        raise ValueError('The loss weight is 0 while there is no loss term')


def load_variant(name, directory=VARIANTS):
    """{'name', 'path', 'sha256', 'spec'} of `<directory>/<name>.json`, strictly validated."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError('A variant is named by a short lower-case identifier, never a path')
    path = Path(directory)/f'{name}.json'
    if not path.is_file():
        raise ValueError(f'No variant file {path}')
    raw = path.read_bytes()
    try:
        spec = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Variant file {path.name} is not valid JSON: {exc}') from None
    check_variant_spec(spec, name)
    resolved = path.resolve()
    shown = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)
    return dict(name=name, path=shown, sha256=hashlib.sha256(raw).hexdigest(), spec=spec)


# --- state rows ---------------------------------------------------------------------------------

def _index(value, bound):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
        raise ValueError(f'Index {value!r} is outside 0..{bound-1}')
    return int(value)


def extras_row(builder, episode_index, frame_index, zero=False):
    """(10,) float32: the builder's extras of one sealed frame, zero padded; all zero if `zero`."""
    row = np.zeros(MAX_EXTRAS, dtype=np.float32)
    episode, frame = _index(episode_index, EPISODES), _index(frame_index, FRAMES_PER_EPISODE)
    if builder.width and not zero:
        row[:builder.width] = builder.table()[episode, frame]
    return row


def expected_extras(builder, rows, zero=False):
    """(n, 10) float32 for dataset row indices (episode*452 + frame)."""
    return np.stack([extras_row(builder, *divmod(_index(row, EPISODES*FRAMES_PER_EPISODE), FRAMES_PER_EPISODE), zero)
                     for row in rows])


def joined_state(state, episode_index, frame_index, builder, zero=False):
    """(19,) float32: the dataset's 9 joint positions, then the builder's extras of the same (episode, frame), zero padded."""
    state = torch.as_tensor(state)
    if state.dtype != torch.float32 or state.shape != (JOINT_DIM,) or not torch.isfinite(state).all():
        raise ValueError('Finite float32 dataset state of exactly 9 joint positions required')
    return torch.cat([state, torch.from_numpy(extras_row(builder, episode_index, frame_index, zero))])


class VariantDataset:
    """The dataset with each row's state joined to a builder's extras, by (episode, frame) and nothing else.

    The builder's table is verified (seal, pinned constants) when it is built. A row that is not at its sealed position
    (`index == episode*452 + frame`) is an error. With `zero_extras` every extra is zero (the zero-equivalence gate).
    Everything but `__getitem__` is the wrapped dataset's.
    """

    def __init__(self, ds, builder, zero_extras=False):
        builder.table()
        self.ds, self.builder, self.zero_extras = ds, builder, zero_extras

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = dict(self.ds[i])
        episode, frame = int(row['episode_index']), int(row['frame_index'])
        if (episode, frame) != divmod(int(i), FRAMES_PER_EPISODE):
            raise ValueError(f'Dataset row {int(i)} is at (episode, frame) ({episode}, {frame}), not its sealed position')
        row['observation.state'] = joined_state(row['observation.state'], episode, frame, self.builder, self.zero_extras)
        return row

    def __getattr__(self, name):
        if name in ('ds', 'builder', 'zero_extras'):
            raise AttributeError(name)
        return getattr(self.ds, name)


def _summary(values):
    stacked = torch.stack(values)
    return {'mean': stacked.mean(0), 'std': stacked.std(0, unbiased=False), 'min': stacked.amin(0), 'max': stacked.amax(0)}


def selected_stats(rows, selected, task_of_episode, builder, zero=False):
    """Training-row statistics: 19 state values (the 9 as P2 computes them, then the 10 padded extras), 8 action values.

    `rows` are raw dataset rows (9-value state). The first 9 entries and the action's are computed exactly as P2's, on their
    own, so they equal P2's byte for byte; the padding has mean, std, min and max 0.
    """
    if selected != list(range(TRAINING_ROWS)):
        raise ValueError('Statistics require every training row exactly once')
    values = {key: [] for key in ('observation.state', 'extras', 'action')}
    for i in selected:
        row = rows[i]
        episode, frame = divmod(i, FRAMES_PER_EPISODE)
        if (int(row['episode_index']) != episode or int(row['frame_index']) != frame
                or int(row['task_index']) != task_of_episode[episode]):
            raise ValueError('Raw training membership/position/instruction mismatch')
        for key, size in (('observation.state', JOINT_DIM), ('action', 8)):
            value = torch.as_tensor(row[key])
            if value.dtype != torch.float32 or value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('Finite unpadded real FP32 state9/action8 required')
            values[key].append(value)
        values['extras'].append(torch.from_numpy(extras_row(builder, episode, frame, zero)))
    state, extras = _summary(values['observation.state']), _summary(values['extras'])
    return {'observation.state': {name: torch.cat([state[name], extras[name]]) for name in state},
            'action': _summary(values['action'])}


def require_start_stats(stats):
    """The action statistics must equal those saved with the start checkpoint (same action space, byte for byte)."""
    from safetensors.torch import load_file
    saved = load_file(str(START/START_NORMALISER))
    action = stats.get('action') or {}
    for key in ('mean', 'std', 'min', 'max'):
        if key not in action or not torch.equal(torch.as_tensor(action[key]).cpu(), saved[f'action.{key}']):
            raise ValueError(f'Action {key} differs from the start checkpoint\'s saved statistics')


def observation(frame):
    """The shared `observation` check for the 19-value state: real inputs and the approved instruction only."""
    if set(frame) != set(OBS)|{'task'} or frame['task'] != single_task.TASK:
        raise ValueError('Inference accepts only real observations and the approved instruction')
    for key, shape in OBS.items():
        value = frame[key]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Invalid physical observation')
        if key in CAMERAS and (value.min() < 0 or value.max() > 1):
            raise ValueError('RGB must be float32 [0,1], without LIBERO rotation')
    return dict(frame)


def check_model_inputs(policy, batch, raw_state, rows, builder, zero=False):
    """What the model will actually see: 9 joint values, the builder's extras exactly as built, zeros after.

    `raw_state` is the (n, 19) state of the collated rows and `rows` their dataset row indices. The expected extras are
    recomputed from the builder's table for those rows, so a loader that left them raw, shifted or truncated them fails.
    """
    inputs = policy._build_model_inputs(batch)
    size = raw_state.shape[0]
    if raw_state.shape != (size, STATE_WIDTH) or len(rows) != size:
        raise ValueError('Model input precheck needs one 19-value state and one dataset row index per sample')
    expected = torch.from_numpy(expected_extras(builder, rows, zero))
    expected_mask = torch.tensor([[True, True, False]]*size, device=inputs['image_mask'].device)
    if (inputs['image_input'].shape != (size, 3, 3, 224, 224)
            or inputs['proprio'].shape != (size, 20) or inputs['input_ids'].shape != (size, 64)
            or not torch.equal(inputs['image_mask'], expected_mask)
            or torch.count_nonzero(inputs['image_input'][:, 2]) != 0
            or torch.count_nonzero(inputs['domain_id']) != 0
            or not torch.equal(inputs['proprio'][:, :JOINT_DIM].cpu(), raw_state[:, :JOINT_DIM].cpu())
            or not torch.equal(inputs['proprio'][:, JOINT_DIM:STATE_WIDTH].cpu(), expected)
            or not torch.equal(raw_state[:, JOINT_DIM:].cpu(), expected)
            or torch.count_nonzero(inputs['proprio'][:, STATE_WIDTH:]) != 0
            or any(not torch.isfinite(x).all() for x in inputs.values())):
        raise ValueError('Model image/proprio/token/domain precheck failed')
    if batch.get('action') is not None and batch['action'].shape != (size, 30, 8):
        raise ValueError('Real action batch shape changed')
    return dict(image_shape=list(inputs['image_input'].shape), image_mask=inputs['image_mask'].tolist(),
                proprio_shape=list(inputs['proprio'].shape), state_width=STATE_WIDTH, extra_width=builder.width,
                real_action_dim=8, state_identity_exact=True, extras_exact=True, extras_forced_zero=bool(zero),
                domain=0, token_shape=list(inputs['input_ids'].shape), privileged_inputs=builder.width > 0,
                libero_rotation=False)


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


# --- the start checkpoint, the binding ----------------------------------------------------------

def start_binding():
    """The bundle's checkpoint 101,520 as pinned: exactly its files, each at its hash, the weights at their marker's."""
    names = {item.name for item in START.iterdir()} if START.is_dir() else set()
    if names != set(START_FILES)|{'model.safetensors'} or any((START/name).is_symlink() for name in names):
        raise ValueError('The start checkpoint is not exactly the pinned bundle checkpoint')
    files = {name: sha256(START/name) for name in START_FILES}
    marker = json.loads((START/'checkpoint_result.json').read_text())
    model = sha256(START/'model.safetensors')
    if (files != START_FILES or model != START_MODEL_SHA256 or marker.get('model_sha256') != model
            or marker.get('update') != START_UPDATE or marker.get('status') != 'complete'
            or marker.get('optimizer_saved') is not False):
        raise ValueError('The start checkpoint differs from the pinned one')
    return dict(path=START.relative_to(ROOT).as_posix(), update=START_UPDATE, model_sha256=model, files=files)


def claim_output(path):
    """Exclusive token permits a newly created empty Docker output bind mount only."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Preserve existing output; no overwrite; a crashed pilot is re-run in a new directory')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose': 'xvla_p2_pilot', 'resume': False, 'required_image': IMAGE}, stream)


def current_binding():
    """Cheap, repeatable rehash of the parent's small files, the props P2 sources, the sidecar, the start and the assets."""
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
    if table_digest(BUILDERS['box'].table()) != BOX_TABLE_SHA256:
        raise ValueError('The target-box table differs from the pinned one')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                plan_sha256=SEAL_RECORD['plan_sha256'], provenance_sha256=SEAL_RECORD['provenance_sha256'],
                validation_sha256=digest_json(validation), props_p2_provenance=current,
                sidecar=forces.binding(), box_table_sha256=BOX_TABLE_SHA256, start=start_binding(),
                sources=source_hashes(), assets=require_assets(), required_image=IMAGE,
                recipe_sha256=digest_json(recipe()))


def variant_record(variant):
    """The part of a run's record that names its variant: the file, its hash and what its builder reads."""
    return dict(name=variant['name'], path=variant['path'], sha256=variant['sha256'], spec=variant['spec'],
                builder=builder_record(BUILDERS[variant['spec']['state_builder']]))


def load_data(builder, zero=False):
    """Dataset-only reader joined to `builder`'s extras. Never library/global stats, pilot manifest or old adapter."""
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
    if ({k: tuple(v.shape) for k, v in features.items() if k.startswith('observation.')} != DATASET_OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != TOTAL_FRAMES):
        raise ValueError('Real dataset state/action/task schema mismatch')
    task_of_episode = tuple(task_index_map(ds)[x] for x in planned_instructions())
    rows, starts = indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    for i in range(TOTAL_FRAMES):
        if (int(ds.hf_dataset[i]['episode_index']) != i//FRAMES_PER_EPISODE
                or int(ds.hf_dataset[i]['frame_index']) != i % FRAMES_PER_EPISODE
                or int(ds.hf_dataset[i]['task_index']) != task_of_episode[i//FRAMES_PER_EPISODE]):
            raise ValueError('Raw episode/frame/instruction membership mismatch')
    stats = selected_stats(ds.hf_dataset, rows, task_of_episode, builder, zero)
    require_start_stats(stats)
    return VariantDataset(ds, builder, zero), starts, stats


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


# --- the recipe ---------------------------------------------------------------------------------

def learning_rate(update):
    """Linear warm-up from 0 to `LR` over `WARMUP` updates, then constant, up to `UPDATES`."""
    if type(update) is not int or not 1 <= update <= UPDATES:
        raise ValueError(f'Update must be integer in [1,{UPDATES}]')
    return LR*min(update, WARMUP)/WARMUP


def set_learning_rate(optimizer, update):
    """Port of the shared setter bound to the pilot schedule; same group checks and vlm scale."""
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
    """Shared config with the state feature widened to 19; LeRobot's state normalisation stays `IDENTITY`."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    config = _make_config(device)
    config.scheduler_decay_steps = UPDATES
    config.input_features = {**config.input_features,
                             'observation.state': PolicyFeature(type=FeatureType.STATE, shape=OBS['observation.state'])}
    config.validate_features()
    if config.normalization_mapping['STATE'].value != 'IDENTITY' or STATE_WIDTH > config.max_state_dim:
        raise ValueError('The 19 state values must pass through IDENTITY normalisation into the 20 proprio slots')
    return config


def recipe(updates=UPDATES):
    """The pilot recipe. `updates` below `UPDATES` is a dry run: a prefix of the same schedule, a different purpose."""
    checkpoint_directory(updates)
    return dict(purpose='xvla_p2_pilot' if updates == UPDATES else 'xvla_p2_pilot_dry', recipe_version=RECIPE_VERSION,
                seed=SEED, max_optimizer_updates=updates,
                microbatch=4, accumulation=2, effective_batch=8, chunk_size=CHUNK_SIZE, n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=LR, vlm_scale=.1, soft_prompt_scale=1., weight_decay=1e-4,
                               vlm_weight_decay=1e-5, betas=[.9, .95], eps=1e-8, clip=10.),
                schedule=dict(kind='linear_warmup_then_constant', warmup=WARMUP, lr=LR, total=UPDATES, vlm_scale=.1),
                start=dict(update=START_UPDATE, model_sha256=START_MODEL_SHA256,
                           source='bundle props_p2 checkpoint, weights only'),
                optimizer_state='fresh', optimizer_storage='none',
                state=dict(width=STATE_WIDTH, joints=JOINT_DIM, extras=MAX_EXTRAS),
                train_episodes=list(TRAIN), validation_episodes=list(VAL), autotune=False, retries=False)


# --- the output ---------------------------------------------------------------------------------

def require_marker(marker, variant, updates=UPDATES):
    require_fields(marker, dict(status='complete', update=updates, reload_fp32_weights_exact=True, optimizer_saved=False,
                                start_update=START_UPDATE, start_model_sha256=START_MODEL_SHA256,
                                variant=variant['name'], variant_sha256=variant['sha256']),
                   'Strict FP32 checkpoint marker of this pilot required')
    error = marker.get('reload_max_action_error')
    if type(error) not in (int, float) or not 0 <= error <= 1e-5:
        raise ValueError('Absolute reload action error must be finite and <=1e-5')


def verify_checkpoint(report, output, variant, updates=UPDATES):
    """The pilot's one checkpoint: its marker, the files the evaluator loads, no optimizer state, nothing else."""
    key = checkpoint_directory(updates)
    markers = report.get('checkpoints', [])
    if [m.get('update') for m in markers] != [updates] or set(report.get('checkpoint_files', {})) != {key}:
        raise ValueError('Exactly the final checkpoint is expected')
    present = {item.name for item in (output/'checkpoints').iterdir()}
    if present != {key}:
        raise ValueError('Unlisted/missing checkpoint')
    marker, directory = markers[0], output/'checkpoints'/key
    require_marker(marker, variant, updates)
    files = inventory(directory)
    if (files != report['checkpoint_files'][key] or set(files) != CHECKPOINT_FILES
            or files['model.safetensors'] != marker.get('model_sha256')
            or json.loads((directory/'checkpoint_result.json').read_text()) != marker):
        raise ValueError('Checkpoint/processor byte inventory changed, or it holds optimizer state')
