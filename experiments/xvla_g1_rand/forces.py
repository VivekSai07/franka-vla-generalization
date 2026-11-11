"""The 8 extra state values of G1: the sealed G1 signals, the clip and the standardisation.

Pure: numpy and S0's array checker only; imports no simulator or model stack, so the admission image (training) and the robosuite
image (rollout) run the very same code.

Each frame's extras are `wrist_wrench` (Fx, Fy, Fz, Tx, Ty, Tz) then `finger_force` (left, right), taken from the per-episode
`signals/episode_NNNN.npz` of the sealed G1 dataset by explicit (episode, frame) keys. They are clipped to the 0.1 / 99.9
percentiles of the training rows (episodes 0-2699), then standardised with the mean and standard deviation of those clipped
training rows. The constants are derived here from the sealed signals (every call checks the seal record first) and pinned by
`CONSTANTS_SHA256`: they come from demonstrations, never from rollouts, and are fixed before training.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.vla_inputs_signals import sidecar as s0_sidecar

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT/'outputs/gen_g1_data/g1_20261110'
SEAL_RECORD = {
    'fingerprint': '845ac94063fd367cc315b8d287fc141df76ea171f9227f5794eef0bd8e4e161e',
    'provenance_sha256': '350bf9fdf8b2084e33f234d02274c098b303be6630db381f09e8e6c793a2139d',
    'plan_sha256': '3b57cfac88ee5f828a4dc35103addf0d9a90e3a75591e9da6460cf31f9061bc3',
    'substitutions_sha256': '807b504ddcf8c291ec785e9085fe9542dcc2642edc507f2c08b6a18566ce7d35',
    'image': 'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e',
    'episodes': 3000,
}
CONSTANTS_SHA256 = 'b6e229658c04d6462cfe96bf71425b314608066835a4d21b33054ce5df2a2d15'  # pinned 2026-11-11 before any training
EPISODES, TRAIN_EPISODES, FRAMES, N_EXTRA = 3000, 2700, 452, 8
EXTRA_NAMES = tuple(s0_sidecar.FORCE_NAMES)
PERCENTILES = (0.1, 99.9)


def verify_seal(root=SEALED):
    """The sealed root's `seal.json` is the one this package is bound to (its full fingerprint is recomputed by `--check`)."""
    sealed = json.loads((Path(root)/'seal.json').read_text())
    if sealed != SEAL_RECORD:
        raise ValueError('The G1 dataset is not the pinned sealed one')
    return sealed


def load_raw(root=SEALED, verify=True):
    """{'values': (3000, 452, 8) float64, 'train': (3000,) bool} from the sealed signals; any structural defect raises.

    Wrist wrench then finger force. Every file is checked by S0's own validator (identity, frame index 0..451, shapes,
    finiteness), so a missing, duplicated or shifted row cannot be joined; the split recorded in each file must be the plan's
    (the first 2,700 episodes train, the last 300 validate).
    """
    root = Path(root)
    if verify:
        verify_seal(root)
    names = sorted(path.name for path in (root/'signals').iterdir())
    if names != [f'episode_{i:04d}.npz' for i in range(EPISODES)]:
        raise ValueError('Signal files are missing, duplicated or unexpected')
    values, train = np.empty((EPISODES, FRAMES, N_EXTRA)), np.zeros(EPISODES, dtype=bool)
    for index, name in enumerate(names):
        with np.load(root/'signals'/name, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        s0_sidecar.check_arrays(arrays, index, FRAMES)
        split = str(arrays['split'])
        if split != ('train' if index < TRAIN_EPISODES else 'validation'):
            raise ValueError(f'Signal episode {index} has split {split!r}')
        values[index] = np.concatenate([arrays['wrist_wrench'], arrays['finger_force']], axis=1)
        train[index] = split == 'train'
    return dict(values=values, train=train)


def clip_bounds(raw, train):
    """(low, high), (8,) float64 each: the 0.1 and 99.9 percentiles of the training rows."""
    rows = np.asarray(raw, dtype=np.float64)[np.asarray(train, dtype=bool)].reshape(-1, N_EXTRA)
    if not len(rows):
        raise ValueError('No training rows')
    low, high = (np.percentile(rows, q, axis=0) for q in PERCENTILES)
    return low, high


def derive_constants(raw, train, low, high):
    """Clip bounds with the mean and standard deviation of the clipped training rows (sums are exactly rounded).

    `math.fsum` makes the bytes independent of numpy's reduction order, so every image derives the same constants.
    """
    raw, train = np.asarray(raw, dtype=np.float64), np.asarray(train, dtype=bool)
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != N_EXTRA or train.shape != raw.shape[:1] or not train.any():
        raise ValueError('Expected (episodes, frames, 8) values and a training mask over the episodes')
    if (low.shape, high.shape) != ((N_EXTRA,), (N_EXTRA,)) or not (np.isfinite(low).all() and np.isfinite(high).all()
                                                                     and (low < high).all()):
        raise ValueError('Clip bounds must be finite and ordered, one pair per extra')
    rows = np.clip(raw[train].reshape(-1, N_EXTRA), low, high)
    if not np.isfinite(rows).all():
        raise ValueError('Training rows must be finite')
    mean = np.array([math.fsum(rows[:, j].tolist())/len(rows) for j in range(N_EXTRA)])
    std = np.array([math.sqrt(math.fsum(((rows[:, j]-mean[j])**2).tolist())/len(rows)) for j in range(N_EXTRA)])
    if not (std > 0).all():
        raise ValueError('An extra has no spread over the training rows')
    return dict(low=low.copy(), high=high.copy(), mean=mean, std=std)


def constants_digest(constants):
    exact = {key: [x.hex() for x in np.asarray(constants[key], dtype=np.float64).tolist()]
             for key in ('low', 'high', 'mean', 'std')}
    return hashlib.sha256(json.dumps(exact, sort_keys=True).encode()).hexdigest()


def constants_json(constants):
    return {key: np.asarray(constants[key], dtype=np.float64).tolist() for key in ('low', 'high', 'mean', 'std')}


def standardise(values, constants):
    """Clip to the training range, then (x - mean) / std, as float32. Raw values are float64, finite, width 8."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != N_EXTRA or not np.isfinite(values).all():
        raise ValueError('Expected finite values with 8 extras per row')
    clipped = np.clip(values, constants['low'], constants['high'])
    return ((clipped-constants['mean'])/constants['std']).astype(np.float32)


def derived(root=SEALED, verify=True):
    """(raw, constants) derived from the signals under `root`; the caller compares the digest with the pinned one."""
    raw = load_raw(root, verify)
    low, high = clip_bounds(raw['values'], raw['train'])
    return raw, derive_constants(raw['values'], raw['train'], low, high)


@lru_cache(maxsize=1)
def _loaded():
    raw, constants = derived(SEALED)
    if constants_digest(constants) != CONSTANTS_SHA256:
        raise ValueError('Derived extra-state constants differ from the pinned ones')
    table = standardise(raw['values'], constants)
    for array in (*constants.values(), table):
        array.flags.writeable = False
    return constants, table


def constants():
    return _loaded()[0]


def table():
    """(3000, 452, 8) float32: every sealed frame's standardised extras."""
    return _loaded()[1]


def clipping_rates():
    """Per extra, the fraction of rows outside the clip range, for training and validation episodes.

    Reported, never used to move the ranges: they are fixed by the training rows before any policy is trained.
    """
    raw, k = load_raw(SEALED, verify=False), constants()
    out = {}
    for split, mask in (('train', raw['train']), ('validation', ~raw['train'])):
        rows = raw['values'][mask].reshape(-1, N_EXTRA)
        out[split] = dict(rows=len(rows), fraction_clipped=((rows < k['low']).mean(0)+(rows > k['high']).mean(0)).tolist())
    return out


def _index(value, bound):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
        raise ValueError(f'Index {value!r} is outside 0..{bound-1}')
    return int(value)


def extra_state(episode_index, frame_index):
    """(8,) float32: the standardised extras of one sealed frame."""
    return table()[_index(episode_index, EPISODES), _index(frame_index, FRAMES)].copy()


def expected_extras(rows):
    """(n, 8) float32 for dataset row indices (episode*452 + frame)."""
    flat = table().reshape(-1, N_EXTRA)
    return np.stack([flat[_index(row, EPISODES*FRAMES)] for row in rows])


def binding():
    """What the training binding records about the signals: the pinned seal, its file digest and the derived constants."""
    seal = verify_seal(SEALED)
    return dict(seal=seal, seal_sha256=hashlib.sha256((SEALED/'seal.json').read_bytes()).hexdigest(),
                constants_sha256=constants_digest(constants()))
