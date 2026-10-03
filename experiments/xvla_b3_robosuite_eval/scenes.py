"""Restore a reserved B3 scene from the sealed bank. Robosuite image (MuJoCo 3.4.0).

The bank is never re-derived: a scene is the stored `initial_state` on the model for its colours,
checked against the stored `qpos` and the manifest's scene hash.
"""
import hashlib
import json

import mujoco
import numpy as np

from data_collection.benchmark import restore_state
from experiments.b3_robosuite_data.scenes import build_model, present_colours  # noqa: F401
from experiments.xvla_b3_robosuite_eval import protocol


def scene_sha256(qpos, initial_state):
    """The hash `b3_robosuite_data.scenes.make_bank` stored: sorted keys, name then raw bytes."""
    digest = hashlib.sha256()
    for name, array in sorted({'qpos': qpos, 'initial_state': initial_state}.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def restore(key):
    """(model, data) for `key = f'{layout_id}:{condition}'`; raises on any disagreement with the bank."""
    manifest = json.loads(protocol.BANK_JSON.read_text())
    if key not in manifest:
        raise ValueError(f'{key!r} is not a banked scene')
    entry = manifest[key]
    with np.load(protocol.BANK_NPZ) as bank:
        qpos, state = bank[f'{key}:qpos'], bank[f'{key}:initial_state']
    if scene_sha256(qpos, state) != entry['scene_sha256']:
        raise ValueError(f'bank arrays for {key} differ from their manifest hash')
    model = build_model(tuple(entry['colours']))
    if present_colours(model) != sorted(entry['colours']):
        raise ValueError(f'model for {key} holds other cubes than the bank')
    data = mujoco.MjData(model)
    restore_state(model, data, state)
    if not np.array_equal(data.qpos, qpos):
        raise ValueError(f'restored qpos for {key} differs from the bank')
    return model, data
