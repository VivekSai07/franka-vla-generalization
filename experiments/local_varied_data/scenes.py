"""Freeze all 77 initializations; forwarding only, no settling or dynamics."""
import hashlib
from pathlib import Path
from shutil import disk_usage

import mujoco
import numpy as np

from data_collection.benchmark import restore_state, synchronize_observation
from data_collection.pilot_integrity import scene_digest
from experiments.fixed_layout import source as original
from experiments.fixed_layout.record import file_hash
from experiments.local_position_probe.scenes import (
    exact, arrays_from_data, require_arrays, cube_qpos, validate_geometry,
)
from . import plan as p


def provenance(check=lambda: None):
    check()
    result = original.provenance()
    paths = [p.ROOT / p.SPEC, p.ROOT / p.PLAN,
             p.ROOT / 'experiments/local_position_probe/scenes.py',
             p.ROOT / 'experiments/local_position_probe/contract.py']
    paths += sorted(Path(__file__).parent.rglob('*.py'))
    paths += [Path(__file__).with_name('README.md')]
    for path in paths:
        if path.is_symlink(): raise ValueError('Symlink source forbidden')
        result['files'][str(path.relative_to(p.ROOT))] = file_hash(path, check)
    result['files'] = dict(sorted(result['files'].items()))
    check()
    return result


def scene_hash(arrays):
    return scene_digest(arrays['qpos'])


def old_offsets():
    result = {(0., 0., 0., 0.)}
    for axis in range(4):
        for value in (-.010, -.005, .005, .010):
            offset = [0.]*4; offset[axis] = value; result.add(tuple(offset))
    return result


def validate_layouts(layouts):
    seen = set()
    for layout in layouts:
        values = layout['offsets_m']
        if (type(values) is not list or len(values) != 4 or any(type(v) is not float for v in values)
                or not np.isfinite(values).all() or np.any(np.abs(values) > .010)):
            raise ValueError('Offsets must be four finite float coordinates within 10mm')
        key = tuple(values)
        if key in seen: raise ValueError('Exact duplicate four-coordinate geometry')
        seen.add(key)
        if key in old_offsets() and not (layout['layout_id'] == 'center' and key == (0.,)*4):
            raise ValueError('Exact old position-probe overlap')


def validate_scene(model, source, layout, arrays):
    validate_layouts([layout]); require_arrays(model, source); require_arrays(model, arrays)
    base = mujoco.MjData(model); restore_state(model, base, source['initial_state'])
    if any(not exact(v, source[k]) for k, v in arrays_from_data(model, base).items()):
        raise ValueError('Source reset inconsistent')
    indices = [cube_qpos(model, color)+axis for color in ('red', 'green') for axis in (0, 1)]
    expected_qpos = source['qpos'].copy(); expected_qpos[indices] += layout['offsets_m']
    prefix = int(mujoco.mjtState.mjSTATE_INTEGRATION) & (int(mujoco.mjtState.mjSTATE_QPOS)-1)
    state_indices = np.asarray(indices)+mujoco.mj_stateSize(model, prefix)
    expected_state = source['initial_state'].copy(); expected_state[state_indices] = expected_qpos[indices]
    if not exact(arrays['qpos'], expected_qpos) or not exact(arrays['initial_state'], expected_state):
        raise ValueError('Full integration change must be ONLY the four requested cube XY coordinates')
    data = mujoco.MjData(model); restore_state(model, data, arrays['initial_state'])
    if any(not exact(v, arrays[k]) for k, v in arrays_from_data(model, data).items()):
        raise ValueError('Derived scene arrays inconsistent')
    for key in ('initial_ctrl', 'camera_xpos', 'camera_xmat'):
        if not exact(arrays[key], source[key]): raise ValueError('Camera/control intervention forbidden')
    expected_xyz = source['body_xpos'].copy()
    for n, color in enumerate(('red', 'green')):
        expected_xyz[model.body(f'target_cube_{color}').id, :2] += layout['offsets_m'][2*n:2*n+2]
    if not np.allclose(data.xpos, expected_xyz, atol=2e-16, rtol=0): raise ValueError('World offsets differ')
    validate_geometry(model, data, source)


def derive_scene(model, source, layout) -> dict:
    validate_layouts([layout]); require_arrays(model, source)
    data = mujoco.MjData(model); restore_state(model, data, source['initial_state'])
    for n, color in enumerate(('red', 'green')):
        start = cube_qpos(model, color); data.qpos[start:start+2] += layout['offsets_m'][2*n:2*n+2]
    synchronize_observation(model, data)
    arrays = arrays_from_data(model, data)
    validate_scene(model, source, layout, arrays)
    return arrays


def make_bank(model, source, plan, check=lambda: None):
    p.validate_plan(plan); validate_layouts(plan['layouts'])
    # Independently derive the old seventeen geometric hashes, not any policy bank.
    old = set()
    for offsets in old_offsets():
        qpos = source['qpos'].copy()
        for n, color in enumerate(('red', 'green')):
            start = cube_qpos(model, color); qpos[start:start+2] += offsets[2*n:2*n+2]
        old.add(scene_digest(qpos))
    bank, seen = {}, set()
    for layout in plan['layouts']:
        check()
        arrays = derive_scene(model, source, layout); digest = scene_hash(arrays)
        if digest in seen: raise ValueError('Exact duplicate full scene hash')
        if digest in old and layout['layout_id'] != 'center': raise ValueError('Full scene overlaps old probe')
        seen.add(digest); bank[layout['layout_id']] = arrays
    return bank


def bank_manifest(plan, bank):
    return {x['layout_id']: dict(layout=x, scene_hash=scene_hash(bank[x['layout_id']]),
                arrays_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in bank[x['layout_id']].items()})
            for x in plan['layouts']}


def preflight() -> dict:
    """Read-only CLI path: no output, dataset, expert.run, renderer or step."""
    plan = p.make_plan(); before = provenance()
    expert = original.Expert(); source = original.load_reset(expert)
    bank = make_bank(expert.model, source, plan)
    if provenance() != before: raise ValueError('Source/runtime changed during preflight')
    destination = original.storage_path(p.OUTPUT); free = disk_usage(destination).free
    return dict(reset_valid=True, unique_layouts=len(bank), expert_slots=60, evaluation_layouts=24,
                policy_seeds=plan['policy_seeds'], output=str(p.OUTPUT), output_exists=p.OUTPUT.exists(),
                storage_path=str(destination), free_bytes=free, storage_eligible=free >= p.START_BYTES,
                collection_ready=False, scope='77_initializations_only_no_steps_no_render_no_expert_execution',
                plan=plan, bank=bank_manifest(plan, bank), provenance=before,
                pending=['independent review and main fresh preflight', 'actual 60 expert and 120 control replay gates'])