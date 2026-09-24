"""Broad-workspace scene bank; cube XY absolute (envelope F7) and 7-DoF home jitter."""
import hashlib
from pathlib import Path
from shutil import disk_usage

import mujoco
import numpy as np

from data_collection.benchmark import restore_state, synchronize_observation
from data_collection.pilot_integrity import scene_digest
from experiments.fixed_layout import source as base_source_module
from experiments.fixed_layout.record import file_hash
from experiments.local_position_probe.scenes import (
    exact, arrays_from_data, require_arrays, cube_qpos, require_no_robot_cube_contacts,
)
from . import plan as p


def provenance(check=lambda: None):
    check()
    result = base_source_module.provenance()
    paths = [p.ROOT / p.SPEC, p.ROOT / p.PLAN,
             p.ROOT / 'experiments/local_position_probe/scenes.py',
             p.ROOT / 'experiments/local_position_probe/contract.py']
    paths += sorted(Path(__file__).parent.rglob('*.py'))
    paths += [Path(__file__).with_name('README.md')]
    for path in paths:
        if path.is_symlink():
            raise ValueError('Symlink source forbidden')
        result['files'][str(path.relative_to(p.ROOT))] = file_hash(path, check)
    result['files'] = dict(sorted(result['files'].items()))
    check()
    return result


def scene_hash(arrays):
    return scene_digest(arrays['qpos'])


def _validate_home_jitter(jitter):
    home = jitter.get('home_pose_jitter_deg')
    if (type(home) is not list or len(home) != 7 or any(type(v) is not float for v in home)
            or not np.isfinite(home).all() or np.any(np.abs(home) > p.HOME_JITTER_DEG_MAX + 1e-12)):
        raise ValueError('home_pose_jitter_deg must be 7 finite floats within +/-1.0 degrees')


def _quad(values, label):
    if type(values) is not list or len(values) != 4 or any(type(v) is not float for v in values) \
            or not np.isfinite(values).all():
        raise ValueError(f'{label} must be four finite floats')


def _in_region(xy, first_axis):
    lo, hi = p.AXIS_LOW, p.AXIS_HIGH
    return lo[first_axis] <= xy[0] <= hi[first_axis] and lo[first_axis+1] <= xy[1] <= hi[first_axis+1]


def _validate_xy(layout):
    """Colour is decorrelated from position here, so validate REGION occupancy, not colour."""
    region = layout.get('region_xy_m')
    if region is not None:
        _quad(region, 'Region XY')
        if layout['layout_id'] != 'base' and not (_in_region(region[:2], 0) and _in_region(region[2:], 2)):
            raise ValueError('Region XY outside envelope F7 rectangles')
    xy = layout.get('cube_xy_m')
    if xy is None:
        return
    _quad(xy, 'Cube XY')
    if layout['layout_id'] == 'base':
        return
    red, green = xy[:2], xy[2:]
    if not ((_in_region(red, 0) and _in_region(green, 2))
            or (_in_region(red, 2) and _in_region(green, 0))):
        raise ValueError('Cubes must occupy one region-A and one region-B slot')


def validate_layouts(layouts):
    seen = set()
    for layout in layouts:
        _validate_xy(layout)
        key = tuple(layout.get('region_xy_m') or layout['cube_xy_m'])
        if key in seen:
            raise ValueError('Exact duplicate cube-XY layout')
        seen.add(key)


def _set_cube_xy(model, data, xy):
    for n, color in enumerate(('red', 'green')):
        start = cube_qpos(model, color)
        data.qpos[start:start+2] = xy[2*n:2*n+2]


def _set_arm_jitter(data, home_jitter_deg):
    rad = np.deg2rad(np.asarray(home_jitter_deg, dtype=np.float64))
    data.qpos[:7] += rad


def derive_scene(model, base_source, layout, jitter) -> dict:
    """Return arrays for a single episode reset: absolute cube XY + arm-jitter home pose."""
    _validate_xy(layout); _validate_home_jitter(jitter); require_arrays(model, base_source)
    data = mujoco.MjData(model)
    restore_state(model, data, base_source['initial_state'])
    if any(not exact(v, base_source[k]) for k, v in arrays_from_data(model, data).items()):
        raise ValueError('Base source reset did not restore bit-exactly')
    _set_cube_xy(model, data, layout['cube_xy_m'])
    _set_arm_jitter(data, jitter['home_pose_jitter_deg'])
    synchronize_observation(model, data)
    arrays = arrays_from_data(model, data)
    _validate_derived(model, base_source, layout, jitter, arrays)
    return arrays


def _validate_derived(model, base_source, layout, jitter, arrays):
    require_arrays(model, arrays)
    if not exact(arrays['initial_ctrl'], base_source['initial_ctrl']):
        raise ValueError('Control changed in derived scene')
    expected_qpos = base_source['qpos'].copy()
    for n, color in enumerate(('red', 'green')):
        start = cube_qpos(model, color)
        expected_qpos[start:start+2] = layout['cube_xy_m'][2*n:2*n+2]
    expected_qpos[:7] = base_source['qpos'][:7] + np.deg2rad(np.asarray(jitter['home_pose_jitter_deg'], np.float64))
    if not exact(arrays['qpos'], expected_qpos):
        raise ValueError('Derived qpos deviates from expected cube-XY + arm-jitter combination')
    check = mujoco.MjData(model); restore_state(model, check, arrays['initial_state'])
    if any(not exact(v, arrays[k]) for k, v in arrays_from_data(model, check).items()):
        raise ValueError('Derived scene arrays inconsistent with initial_state')
    data = mujoco.MjData(model); restore_state(model, data, arrays['initial_state'])
    _validate_broad_geometry(model, data, base_source)


def _validate_broad_geometry(model, data, base_source):
    positions = np.array([data.xpos[model.body(f'target_cube_{c}').id] for c in ('red', 'green')])
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(positions).all():
        raise ValueError('Nonfinite derived scene')
    for color in ('red', 'green'):
        body = model.body(f'target_cube_{color}').id
        if data.xpos[body, 2] != base_source['body_xpos'][body, 2]:
            raise ValueError(f'Cube {color} settled height changed')
    if np.linalg.norm(positions[0, :2] - positions[1, :2]) < .15 - 1e-5:
        raise ValueError('Cubes closer than 15 cm; envelope F7 violated')
    require_no_robot_cube_contacts(model, data)


def eval_bank_key(layout_id, arrangement):
    """Reserved layouts are banked once per arrangement; the evaluator never builds a scene itself."""
    if arrangement not in p.ARRANGEMENTS:
        raise ValueError(f'Unapproved arrangement: {arrangement!r}')
    return f'{layout_id}:{arrangement}'


def make_bank(model, base_source, plan, check=lambda: None) -> dict:
    """200 per-episode + 80 evaluation layouts x 2 arrangements + 1 base = 361 identities."""
    p.validate_plan(plan); validate_layouts(plan['layouts'])
    layouts = {x['layout_id']: x for x in plan['layouts']}
    layouts['base'] = plan['base_layout']
    bank, seen = {}, set()

    def add(key, layout, jitter):
        check()
        arrays = derive_scene(model, base_source, layout, jitter)
        digest = scene_hash(arrays)
        if digest in seen:
            raise ValueError(f'Duplicate scene digest for {key}')
        seen.add(digest); bank[key] = arrays

    for episode in plan['episodes']:
        i = episode['episode_index']
        layout = layouts[episode['layout_id']]
        add(f'episode_{i:03d}', layout, dict(home_pose_jitter_deg=episode['home_pose_jitter_deg']))

    for layout_id in plan['evaluation_layout_ids']:
        eval_jit = plan['evaluation_holdout'][layout_id]
        for arrangement in p.ARRANGEMENTS:
            placed = dict(layouts[layout_id], cube_xy_m=list(eval_jit['arrangements'][arrangement]))
            add(eval_bank_key(layout_id, arrangement), placed,
                dict(home_pose_jitter_deg=eval_jit['home_pose_jitter_deg']))

    add('base', plan['base_layout'], dict(home_pose_jitter_deg=[0.0]*7))
    if len(bank) != p.UNIQUE_BANK_IDENTITIES:
        raise ValueError(f'Bank size {len(bank)} != {p.UNIQUE_BANK_IDENTITIES}')
    return bank


def bank_manifest(plan, bank):
    manifest = {}
    layouts = {x['layout_id']: x for x in plan['layouts']}
    layouts['base'] = plan['base_layout']
    for episode in plan['episodes']:
        i = episode['episode_index']; key = f'episode_{i:03d}'
        manifest[key] = dict(kind='episode', episode_index=i, layout_id=episode['layout_id'],
                             layout=layouts[episode['layout_id']],
                             cell=episode['cell'], arrangement=episode['arrangement'],
                             pick=episode['pick'], place=episode['place'],
                             instruction=episode['instruction'], task_index=episode['task_index'],
                             home_pose_jitter_deg=episode['home_pose_jitter_deg'],
                             scene_hash=scene_hash(bank[key]),
                             arrays_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in bank[key].items()})
    for layout_id in plan['evaluation_layout_ids']:
        eh = plan['evaluation_holdout'][layout_id]
        for arrangement in p.ARRANGEMENTS:
            key = eval_bank_key(layout_id, arrangement)
            placed = dict(layouts[layout_id], cube_xy_m=list(eh['arrangements'][arrangement]))
            manifest[key] = dict(kind='evaluation', layout_id=layout_id, arrangement=arrangement,
                                 layout=placed,
                                 home_pose_jitter_deg=eh['home_pose_jitter_deg'],
                                 hover_z_jitter_m=eh['hover_z_jitter_m'],
                                 scene_hash=scene_hash(bank[key]),
                                 arrays_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in bank[key].items()})
    manifest['base'] = dict(kind='base', layout=plan['base_layout'], home_pose_jitter_deg=[0.0]*7,
                            scene_hash=scene_hash(bank['base']),
                            arrays_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in bank['base'].items()})
    return dict(sorted(manifest.items()))


def preflight() -> dict:
    """Read-only CLI path: no expert.run, renderer, dataset or physics step."""
    plan = p.make_plan(); before = provenance()
    base_expert = base_source_module.Expert()
    source = base_source_module.load_reset(base_expert)
    bank = make_bank(base_expert.model, source, plan)
    if provenance() != before:
        raise ValueError('Source/runtime changed during preflight')
    destination = base_source_module.storage_path(p.OUTPUT); free = disk_usage(destination).free
    return dict(reset_valid=True, unique_bank_identities=len(bank), expert_slots=p.ADMIT_COUNT,
                evaluation_layouts=p.EVAL_HOLDOUT_COUNT, policy_seeds=plan['policy_seeds'],
                output=str(p.OUTPUT), output_exists=p.OUTPUT.exists(), storage_path=str(destination),
                free_bytes=free, storage_eligible=free >= p.START_BYTES, collection_ready=False,
                scope='281_bank_only_no_steps_no_render_no_expert_execution',
                plan=plan, bank=bank_manifest(plan, bank), provenance=before,
                pending=['independent review and main fresh preflight',
                         'actual 200 expert admits with feasibility gate and replay budget'])
