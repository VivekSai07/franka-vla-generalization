"""The 8 extra state values of M2: the sealed S0 sidecar, the clip and the standardisation.

Pure: numpy and S0's contract only; imports no simulator or model stack, so the admission image (training) and the
robosuite image (rollout) run the very same code.

Each frame's extras are `wrist_wrench` (Fx, Fy, Fz, Tx, Ty, Tz) then `finger_force` (left, right), taken from the sealed S0
sidecar by explicit (episode, frame) keys. They are clipped to the 0.1 / 99.9 percentiles of the training rows that S0's
sealed `report.json` records, then standardised with the mean and standard deviation of those clipped training rows. The
constants are derived from the sealed sidecar (every call verifies its seal first) and pinned by `CONSTANTS_SHA256`: they
come from demonstrations, never from rollouts, and are fixed before training.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.vla_inputs_signals import contract as s0_contract
from experiments.vla_inputs_signals import sidecar as s0_sidecar

ROOT = Path(__file__).resolve().parents[2]
SIDECAR = ROOT/'outputs/vla_inputs_signals/p2_signals_20261030'
SIDECAR_RECORD = {
    'episodes': 540,
    'fingerprint': '7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c',
    'frames': 244080,
    'image': 'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e',
    'p2_seal': 'b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898',
    'provenance_sha256': '577d94e3f27a208e93e5097cc47410c5e04244e2c6bd0b4bc3d4936304c65e07',
    'runtime_sha256': 'f41bde6f3fc2d201d2c53f214ae9c4e79fce1fca7b9dd1616d979e994d1eff5f',
}
CONSTANTS_SHA256 = '98d8682198b5a154a095e8aedd2735dfe32d3cf2ab38356c49430752e1b02824'
EPISODES, FRAMES, N_EXTRA = 540, 452, 8
EXTRA_NAMES = tuple(s0_sidecar.FORCE_NAMES)
SPLITS = ('train', 'validation')


def verify_sidecar(root=SIDECAR):
    """Recompute everything S0's seal claims, then require it to be the sealed sidecar this package is bound to."""
    sealed = s0_sidecar.verify(root, EPISODES, FRAMES)
    if sealed != SIDECAR_RECORD:
        raise ValueError('The S0 sidecar is not the pinned sealed one')
    return sealed


def load_raw(root=SIDECAR, verify=True):
    """{'values': (540, 452, 8) float64, 'train': (540,) bool} from the sidecar's episodes; any structural defect raises.

    Wrist wrench then finger force. Episode files are checked by S0's own validator (identity, frame index 0..451,
    shapes, finiteness), so a missing, duplicated or shifted row cannot be joined.
    """
    root = Path(root)
    if verify:
        verify_sidecar(root)
    names = sorted(path.name for path in (root/'episodes').iterdir())
    if names != [f'episode_{i:03d}.npz' for i in range(EPISODES)]:
        raise ValueError('Sidecar episodes are missing, duplicated or unexpected')
    values, train = np.empty((EPISODES, FRAMES, N_EXTRA)), np.zeros(EPISODES, dtype=bool)
    for index, name in enumerate(names):
        with np.load(root/'episodes'/name, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        s0_sidecar.check_arrays(arrays, index, FRAMES)
        split = str(arrays['split'])
        if split not in SPLITS:
            raise ValueError(f'Sidecar episode {index} has split {split!r}')
        values[index] = np.concatenate([arrays['wrist_wrench'], arrays['finger_force']], axis=1)
        train[index] = split == 'train'
    return dict(values=values, train=train)


def clip_bounds(root=SIDECAR):
    """(low, high), (8,) float64 each: S0's report of the 0.1 and 99.9 percentiles of the training rows."""
    report = json.loads((Path(root)/'report.json').read_text())
    if report.get('force_names') != list(EXTRA_NAMES):
        raise ValueError('S0 report columns are not the extras in the sealed order')
    low, high = (np.array(report['clip_percentiles'][key], dtype=np.float64) for key in ('p0.1', 'p99.9'))
    if low.shape != (N_EXTRA,) or high.shape != (N_EXTRA,):
        raise ValueError('S0 clip percentiles must have one value per extra')
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
    """Clip to the sealed range, then (x - mean) / std, as float32. Raw values are float64, finite, width 8."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != N_EXTRA or not np.isfinite(values).all():
        raise ValueError('Expected finite values with 8 extras per row')
    clipped = np.clip(values, constants['low'], constants['high'])
    return ((clipped-constants['mean'])/constants['std']).astype(np.float32)


@lru_cache(maxsize=1)
def _loaded():
    raw = load_raw(SIDECAR)
    low, high = clip_bounds(SIDECAR)
    constants = derive_constants(raw['values'], raw['train'], low, high)
    if constants_digest(constants) != CONSTANTS_SHA256:
        raise ValueError('Derived extra-state constants differ from the pinned ones')
    table = standardise(raw['values'], constants)
    for array in (*constants.values(), table):
        array.flags.writeable = False
    return constants, table


def constants():
    return _loaded()[0]


def table():
    """(540, 452, 8) float32: every sealed frame's standardised extras."""
    return _loaded()[1]


def clipping_rates():
    """Per extra, the fraction of rows outside the sealed clip range, for training and validation episodes.

    Reported, never used to move the ranges: they are fixed by S0's report before any policy is trained.
    """
    raw, k = load_raw(SIDECAR, verify=False), constants()
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
    """What the training binding records about the sidecar: its pinned seal, report, source and derived constants."""
    seal = json.loads((SIDECAR/'seal.json').read_text())
    if seal != SIDECAR_RECORD:
        raise ValueError('The S0 sidecar seal is not the pinned one')
    provenance = json.loads((SIDECAR/'provenance.json').read_text())
    current = s0_contract.provenance()
    if provenance != current or s0_sidecar.digest(current) != SIDECAR_RECORD['provenance_sha256']:
        raise ValueError('S0 source provenance drifted from the sealed sidecar')
    return dict(seal=seal, seal_sha256=hashlib.sha256((SIDECAR/'seal.json').read_bytes()).hexdigest(),
                report_sha256=hashlib.sha256((SIDECAR/'report.json').read_bytes()).hexdigest(),
                constants_sha256=constants_digest(constants()), s0_provenance=current)
