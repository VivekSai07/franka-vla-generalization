"""The target's box in both cameras for every sealed frame: S0's `target_box`, read from the sealed sidecar.

Pure: numpy and S0's contract only (no simulator, no model stack), so the admission image (the training loader) and the robosuite
image (the rollout parity tests) run the very same code. The sidecar's seal is verified (by M2's `forces`, which pins it) before
any episode file is opened, the table is pinned by `TABLE_SHA256`, and every training row must have `in_frame` 1 in both cameras
(S0's finding; the overlay is never silently absent from a training frame).
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.vla_inputs_signals import sidecar as s0_sidecar
from experiments.xvla_p2_force import forces

EPISODES, FRAMES, TRAIN_EPISODES = forces.EPISODES, forces.FRAMES, 480
SIDECAR_EPISODES = forces.SIDECAR/'episodes'
TABLE_SHA256 = 'd168977668b8d3bb61acdc1aac1216564c5707c12b674740360fe82454a2a1d7'


def load_raw(root=forces.SIDECAR, verify=True):
    """(540, 452, 2, 5) float64: `target_box` of every episode file, front camera then gripper camera; any defect raises."""
    root = Path(root)
    if verify:
        forces.verify_sidecar(root)
    names = sorted(path.name for path in (root/'episodes').iterdir())
    if names != [f'episode_{i:03d}.npz' for i in range(EPISODES)]:
        raise ValueError('Sidecar episodes are missing, duplicated or unexpected')
    table = np.empty((EPISODES, FRAMES, 2, 5))
    for index, name in enumerate(names):
        with np.load(root/'episodes'/name, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        s0_sidecar.check_arrays(arrays, index, FRAMES)
        box = arrays['target_box']
        if box.shape != (FRAMES, 2, 5) or box.dtype != np.float64:
            raise ValueError(f'Sidecar episode {index} target_box is not (452, 2, 5) float64')
        table[index] = box
    return table


def check_training_in_frame(table):
    """Every training row has the target in frame in both cameras, so a training frame always carries its band."""
    if not (table[:TRAIN_EPISODES, :, :, 4] == 1.0).all():
        raise ValueError('A training frame has a target box that is not in frame; S0 found none')


def table_digest(table):
    return hashlib.sha256(np.ascontiguousarray(table, dtype='<f8').tobytes()).hexdigest()


@lru_cache(maxsize=1)
def _loaded():
    table = load_raw()
    if not (np.isfinite(table).all() and ((table >= 0) & (table <= 1)).all()):
        raise ValueError('Sidecar boxes must be finite and in [0, 1]')
    check_training_in_frame(table)
    if table_digest(table) != TABLE_SHA256:
        raise ValueError('Sidecar target boxes differ from the pinned table')
    table.flags.writeable = False
    return table


def table():
    """(540, 452, 2, 5) float64: every sealed frame's boxes, exactly as stored."""
    return _loaded()


def _index(value, bound):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
        raise ValueError(f'Index {value!r} is outside 0..{bound-1}')
    return int(value)


def target_box(episode_index, frame_index):
    """(2, 5) float64 copy: the box of one sealed frame in the front and gripper cameras."""
    return table()[_index(episode_index, EPISODES), _index(frame_index, FRAMES)].copy()


def expected_boxes(rows):
    """(n, 2, 5) float64 for dataset row indices (episode*452 + frame)."""
    flat = table().reshape(-1, 2, 5)
    return np.stack([flat[_index(row, EPISODES*FRAMES)] for row in rows])


def binding():
    """What the training binding records about the boxes beyond the sidecar seal M2's `forces.binding()` already pins."""
    table()
    return dict(target_box_sha256=TABLE_SHA256)


@lru_cache(maxsize=1)
def labels():
    """(objects, segments): each episode's target object name and its (540, 452) int8 segment labels, for the sample sheets.

    S0's segments: 0 before_close, 1 closing, 2 closed_and_lifted, 3 after_release. Read after `table()` has verified the seal.
    """
    table()
    objects, segments = [], np.empty((EPISODES, FRAMES), dtype=np.int8)
    for index in range(EPISODES):
        with np.load(SIDECAR_EPISODES/f'episode_{index:03d}.npz', allow_pickle=False) as stored:
            objects.append(str(stored['object']))
            segments[index] = stored['segment']
    return tuple(objects), segments


def first_frame(segment, value):
    """Index of the first frame whose segment label is `value`; raises when the episode never reaches it."""
    hits = np.flatnonzero(np.asarray(segment) == value)
    if not len(hits):
        raise ValueError(f'No frame has segment {value}')
    return int(hits[0])
