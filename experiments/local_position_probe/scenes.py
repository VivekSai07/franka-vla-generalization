"""Translate a sealed full reset, forward geometry only, preserve warmstart exactly."""
import hashlib
import json

import mujoco
import numpy as np

from data_collection.benchmark import capture_state, restore_state, restore_timing, synchronize_observation
from data_collection.pilot_integrity import scene_digest, validate_scene_geometry
from src.robots.franka_panda.config import ACTIVE_JOINTS, SCENE_PATH
from . import contract as c


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def arrays_from_data(model, data):
    return dict(initial_state=capture_state(model, data), qpos=data.qpos.copy(), initial_ctrl=data.ctrl.copy(),
                body_xpos=data.xpos.copy(), camera_xpos=data.cam_xpos.copy(), camera_xmat=data.cam_xmat.copy())


def require_arrays(model, arrays):
    shapes = dict(initial_state=(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),),
                  qpos=(model.nq,), initial_ctrl=(model.nu,), body_xpos=(model.nbody, 3),
                  camera_xpos=(model.ncam, 3), camera_xmat=(model.ncam, 9))
    if set(arrays) != set(shapes): raise ValueError('Reset array inventory changed')
    for key, shape in shapes.items():
        a = arrays[key]
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid finite full-precision reset array: {key}')


def cube_qpos(model, color):
    body = model.body(f'target_cube_{color}').id
    joint = int(model.body_jntadr[body])
    if (model.body_jntnum[body] != 1 or joint < 0 or model.jnt_bodyid[joint] != body
            or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError('Cube must have exactly one free joint on its own body')
    return int(model.jnt_qposadr[joint])


def require_no_robot_cube_contacts(model, data):
    cubes = {model.body(f'target_cube_{color}').id for color in ('red', 'green', 'blue', 'yellow')}
    joints = model.actuator_trnid[list(ACTIVE_JOINTS), 0]
    roots = {int(model.body_rootid[model.jnt_bodyid[j]]) for j in joints}
    robot = {body for body in range(1, model.nbody) if int(model.body_rootid[body]) in roots}
    if not robot or robot & cubes: raise ValueError('Ambiguous robot/cube body identity')
    for contact in data.contact[:data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bodies & cubes and bodies & robot: raise ValueError('Initial robot-cube contact forbidden')


def validate_geometry(model, data, source):
    validate_scene_geometry(model, data, ['red', 'green'])
    for color in ('red', 'green'):
        body = model.body(f'target_cube_{color}').id
        position = data.xpos[body]
        if not (.20 <= position[0] <= .60 and -.40 <= position[1] <= .40):
            raise ValueError('Translated cube outside exact workspace bounds')
        if position[2] != source['body_xpos'][body, 2]: raise ValueError('Settled cube height changed')
    if np.linalg.norm(data.xpos[model.body('target_cube_red').id, :2]-data.xpos[model.body('target_cube_green').id, :2]) < .15:
        raise ValueError('Cube clearance below 15cm')
    require_no_robot_cube_contacts(model, data)


def validate_scene(model, source_reset, condition, arrays):
    c.validate_condition(condition)
    require_arrays(model, source_reset); require_arrays(model, arrays)
    for color in ('red', 'green'): cube_qpos(model, color)
    source = mujoco.MjData(model); restore_state(model, source, source_reset['initial_state'])
    for key, value in arrays_from_data(model, source).items():
        if not exact(value, source_reset[key]): raise ValueError(f'Source reset inconsistent: {key}')
    expected_qpos = source.qpos.copy()
    expected_state = source_reset['initial_state'].copy()
    index = None
    if condition['object'] is not None:
        qindex = cube_qpos(model, condition['object'])+condition['axis']
        # MuJoCo serializes selected state components in bit order. Query prefix size, not guessed offset.
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        prefix = spec & (int(mujoco.mjtState.mjSTATE_QPOS)-1)
        index = mujoco.mj_stateSize(model, prefix)+qindex
        expected_qpos[qindex] += condition['delta_m']
        expected_state[index] = expected_qpos[qindex]
    if not exact(arrays['initial_state'], expected_state) or not exact(arrays['qpos'], expected_qpos):
        raise ValueError('Full integration diff must be ONLY the expected cube XY qpos scalar')
    data = mujoco.MjData(model); restore_state(model, data, arrays['initial_state'])
    for key, actual in arrays_from_data(model, data).items():
        if not exact(actual, arrays[key]): raise ValueError(f'Derived reset inconsistent: {key}')
    for key in ('initial_ctrl', 'camera_xpos', 'camera_xmat'):
        if not exact(arrays[key], source_reset[key]): raise ValueError(f'Control/camera changed: {key}')
    expected_positions = source_reset['body_xpos'].copy()
    if condition['object'] is not None:
        body = model.body(f"target_cube_{condition['object']}").id
        expected_positions[body, condition['axis']] += condition['delta_m']
    if not np.allclose(data.xpos, expected_positions, rtol=0, atol=2e-16):
        raise ValueError('Actual world translation differs from requested XY intervention')
    validate_geometry(model, data, source_reset)
    return dict(changed_integration_index=index, state_size=len(expected_state), qpos_size=model.nq,
                no_steps=True, no_resettle=True, no_initial_robot_cube_contacts=True,
                arrays_sha256={k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in arrays.items()})


def derive_scene(model, source_reset, condition) -> dict:
    c.validate_condition(condition); require_arrays(model, source_reset)
    for color in ('red', 'green'): cube_qpos(model, color)
    data = mujoco.MjData(model); restore_state(model, data, source_reset['initial_state'])
    if condition['object'] is not None:
        index = cube_qpos(model, condition['object'])+condition['axis']
        data.qpos[index] += condition['delta_m']
    synchronize_observation(model, data)
    arrays = arrays_from_data(model, data)
    checks = validate_scene(model, source_reset, condition, arrays)
    return dict(condition=condition.copy(), arrays=arrays, scene_hash=scene_digest(data.qpos), checks=checks)


def load_source():
    path = c.PARENT/'reset.npz'
    if c.sha256(path) != c.RESET_SHA256: raise ValueError('Sealed source reset bytes changed')
    with np.load(path, allow_pickle=False) as archive: source = {key: archive[key].copy() for key in archive.files}
    manifest = json.loads((c.PARENT/'configuration.json').read_text())
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    if restore_timing(model, manifest)['substeps_per_control'] != 20: raise ValueError('Baseline cadence changed')
    if manifest['camera_names'] != ['front_cam', 'gripper_cam'] or manifest['camera_size'] != [512, 512]:
        raise ValueError('Baseline cameras changed')
    for key, actual in (('camera_pos', model.cam_pos), ('camera_quat', model.cam_quat), ('camera_fovy', model.cam_fovy)):
        if not np.array_equal(actual, manifest[key]): raise ValueError('Camera calibration changed')
    if scene_digest(source['qpos']) != c.SOURCE_SCENE_SHA256: raise ValueError('Wrong source scene identity')
    validate_scene(model, source, c.make_plan()['conditions'][0], source)
    return model, manifest, source


def bank_summary(derived):
    return {key: value for key, value in derived.items() if key != 'arrays'}