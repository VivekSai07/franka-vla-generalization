"""B3 scenes: one cached model per set of cube colours, a deterministic reset, and the reserved bank.

Runs in the robosuite image (MuJoCo 3.4.0). Nothing here draws random numbers; every position
comes from the plan.
"""
import functools
import hashlib

import mujoco
import numpy as np

from experiments.b3_robosuite_data.plan import COLOURS, MIN_CLEARANCE_M, PLAN, SPEC  # noqa: F401
from experiments.robosuite_scene.scene import build_sim

CUBE_REST_Z_M = 0.02
CUBE_SETTLED_Z_M = 0.021  # measured: 0.02097 for every cube over 40 planned resets
XY_TOLERANCE_M = 1e-3
Z_TOLERANCE_M = 1e-3
SETTLE_SECONDS = 0.2
HOME_JOINT_TOLERANCE_RAD = 5e-3


@functools.lru_cache(maxsize=None)
def _built(colours):
    env, model, _ = build_sim(colours=colours)
    return env, model  # the env must stay alive or robosuite frees its sim


def build_model(colours):
    """Compiled model holding exactly these cubes; one per sorted colour tuple."""
    return _built(tuple(sorted(colours)))[1]


def present_colours(model):
    return sorted(c for c in COLOURS
                  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_cube_{c}") >= 0)


def reset(model, cube_xy_m, home_pose_jitter_deg):
    """Arm at Q_HOME plus jitter, cubes at their planned xy, settled 0.2 s; no RNG."""
    from data_collection.benchmark import capture_state, synchronize_observation
    from src.robots.franka_panda.config import Q_HOME

    if sorted(cube_xy_m) != present_colours(model):
        raise ValueError(f"cubes {sorted(cube_xy_m)} do not match the model's {present_colours(model)}")
    if len(home_pose_jitter_deg) != 7:
        raise ValueError("home_pose_jitter_deg must have 7 values")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for colour, (x, y) in cube_xy_m.items():
        joint = model.joint(int(model.body(f"target_cube_{colour}").jntadr[0]))
        adr = int(joint.qposadr[0])
        data.qpos[adr:adr + 3] = (x, y, CUBE_REST_Z_M)
        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    home = np.asarray(Q_HOME[:7], dtype=np.float64) + np.deg2rad(np.asarray(home_pose_jitter_deg, dtype=np.float64))
    data.qpos[:9] = Q_HOME
    data.qpos[:7] = home
    data.ctrl[:] = 0.0
    data.ctrl[:7] = home
    data.ctrl[-1] = 255
    mujoco.mj_forward(model, data)
    for _ in range(round(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return {"qpos": data.qpos.copy(), "initial_state": capture_state(model, data)}


def cube_positions(model, data):
    return {c: data.xpos[model.body(f"target_cube_{c}").id].copy() for c in present_colours(model)}


def validate_geometry(model, data, cube_xy_m):
    """The settled scene is the planned one: positions, rest height, spacing and no stray contact."""
    positions = cube_positions(model, data)
    if sorted(cube_xy_m) != sorted(positions):
        raise ValueError("planned cubes differ from the model's cubes")
    for colour, planned in cube_xy_m.items():
        position = positions[colour]
        if not np.isfinite(position).all():
            raise ValueError(f"cube {colour} is not finite")
        if float(np.hypot(*(position[:2] - np.asarray(planned)))) > XY_TOLERANCE_M:
            raise ValueError(f"cube {colour} moved from its planned xy")
        if abs(float(position[2]) - CUBE_SETTLED_Z_M) > Z_TOLERANCE_M:
            raise ValueError(f"cube {colour} is not at rest height (z={position[2]:.4f})")
    colours = sorted(positions)
    for i, a in enumerate(colours):
        for b in colours[i + 1:]:
            if float(np.linalg.norm(positions[a][:2] - positions[b][:2])) < MIN_CLEARANCE_M - 1e-3:
                raise ValueError(f"cubes {a} and {b} are closer than the minimum clearance")
    cube_bodies = {model.body(f"target_cube_{c}").id for c in colours}
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bodies & cube_bodies and len(bodies & cube_bodies) == 2:
            raise ValueError("two cubes touch at reset")
        if bodies & cube_bodies:
            other = (bodies - cube_bodies).pop() if bodies - cube_bodies else None
            if other is not None and model.body(other).name not in ("world", "table", ""):
                raise ValueError(f"cube touches {model.body(other).name} at reset")


def _scene_hash(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def bank_key(layout_id, condition):
    return f"{layout_id}:{condition}"


def make_bank(plan):
    """Settled start states for every reserved (layout, condition): (arrays, manifest)."""
    arrays, manifest = {}, {}
    for layout_id, layout in plan["evaluation_holdout"].items():
        for condition, distractor_xy in layout["conditions"].items():
            cube_xy = {**layout["cube_xy_m"], **distractor_xy}
            model = build_model(tuple(cube_xy))
            scene = reset(model, cube_xy, layout["home_pose_jitter_deg"])
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            from data_collection.benchmark import restore_state
            restore_state(model, data, scene["initial_state"])
            validate_geometry(model, data, cube_xy)
            key = bank_key(layout_id, condition)
            arrays[f"{key}:qpos"] = scene["qpos"]
            arrays[f"{key}:initial_state"] = scene["initial_state"]
            manifest[key] = dict(
                layout_id=layout_id, condition=condition, tier=layout["tier"], pick=layout["pick"],
                place=layout["place"], side=layout["side"], edge=layout["edge"],
                colours=sorted(cube_xy), cube_xy_m=dict(sorted(cube_xy.items())),
                home_pose_jitter_deg=list(layout["home_pose_jitter_deg"]),
                scene_sha256=_scene_hash({"qpos": scene["qpos"], "initial_state": scene["initial_state"]}))
    return arrays, dict(sorted(manifest.items()))
