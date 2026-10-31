"""Per-frame signals computed from a forwarded `(model, data)` without writing either.

Runs in the robosuite image. Everything here is evaluated at the observation instant: after
`synchronize_observation`, before the command. The evaluator (M0) calls the same functions on a live rollout.
"""
import copy

import mujoco
import numpy as np

HAND = "robot0_right_hand"
FINGERS = ("left_finger", "right_finger")


def scratch(model, data):
    """A full copy of `data`; anything that writes derived fields (`mj_rnePostConstraint`) writes only this."""
    return copy.copy(data)


def wrist_wrench(model, data):
    """(force N, torque N*m) the parent link exerts on the hand, at the hand origin, in the hand frame.

    This is MuJoCo's force/torque-sensor convention (child body's interaction wrench, expressed in the site
    frame), with a site at the hand origin. Gravity of the hand, the fingers and anything held is included: it is
    the raw reading of a wrist sensor, not a gravity-compensated external load.
    """
    hand = model.body(HAND).id
    work = scratch(model, data)
    mujoco.mj_rnePostConstraint(model, work)
    wrench = np.zeros(6)
    mujoco.mju_transformSpatial(wrench, work.cfrc_int[hand], 1, work.xpos[hand],
                                work.subtree_com[model.body_rootid[hand]], work.xmat[hand])
    return np.concatenate([wrench[3:], wrench[:3]])


def _below(model, body, ancestor):
    while body != 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def finger_force(model, data):
    """(left, right) sum of normal contact forces in N between each finger and anything outside the robot."""
    robot_root = model.body_rootid[model.body(HAND).id]
    fingers = [model.body(name).id for name in FINGERS]
    out, force = np.zeros(2), np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        for mine, other in (bodies, bodies[::-1]):
            if model.body_rootid[other] == robot_root:
                continue
            for k, finger in enumerate(fingers):
                if _below(model, mine, finger):
                    mujoco.mj_contactForce(model, data, i, force)
                    out[k] += force[0]
    return out


CAMERAS = ("front_cam", "gripper_cam")
IMAGE = 512
RENDERED_GROUPS = 3  # the renderer draws geom groups 0-2 by default
_CUBE = np.array(np.meshgrid((-1, 1), (-1, 1), (-1, 1), indexing="ij")).reshape(3, 8).T
_CUBE_EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1])


def _half_extent(model, geom):
    """Half-extents of the geom's local axis-aligned cube: exact for a box, a documented loose bound otherwise."""
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return size.copy()
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([size[0], size[0], size[1] + size[0]])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.array([size[0], size[0], size[1]])
    raise ValueError(f"geom {geom}: type {mujoco.mjtGeom(kind).name} has no projected bound")


def _geometry(model, data, body):
    """World-space vertices and edges of the rendered geoms of `body`: mesh surface edges, cube edges for primitives."""
    verts, edges, offset = [], [], 0
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_group[geom] >= RENDERED_GROUPS:
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start + count]
            faces = model.mesh_face[model.mesh_faceadr[mesh]:][:model.mesh_facenum[mesh]]
            edge = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        else:
            local, edge = _CUBE * _half_extent(model, geom), _CUBE_EDGES
        verts.append(data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T)
        edges.append(edge + offset)
        offset += len(local)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    return np.concatenate(verts), np.concatenate(edges)


def _project(model, data, geometry, camera):
    verts, edges = geometry
    cam = model.camera(camera).id
    local = (verts - data.cam_xpos[cam]) @ data.cam_xmat[cam].reshape(3, 3)  # x right, y up, looking along -z
    depth = -local[:, 2]
    near = model.vis.map.znear * model.stat.extent
    front = depth > near
    a, b = edges[:, 0], edges[:, 1]
    crossing = front[a] != front[b]
    t = (near - depth[a[crossing]]) / (depth[b[crossing]] - depth[a[crossing]])
    cut = local[a[crossing]] + t[:, None] * (local[b[crossing]] - local[a[crossing]])
    points = np.concatenate([local[front], cut])
    if not len(points):
        return np.zeros(5)
    d = np.concatenate([depth[front], np.full(len(cut), near)])
    f = IMAGE / 2 / np.tan(np.radians(model.cam_fovy[cam]) / 2)
    u, v = IMAGE / 2 + f * points[:, 0] / d, IMAGE / 2 - f * points[:, 1] / d
    low = np.maximum([u.min(), v.min()], 0.0)
    high = np.minimum([u.max(), v.max()], float(IMAGE))
    if (high <= low).any():
        return np.zeros(5)
    return np.array([low[0], low[1], high[0], high[1], IMAGE]) / IMAGE


def box(model, data, body_name, camera):
    """(u_min, v_min, u_max, v_max, in_frame) of the body's rendered geoms in `camera`, /512, clipped to [0, 1].

    Geometry is clipped at the near plane before the perspective division. `in_frame` is projected geometric
    validity (a positive-area box meeting the image), not visibility: occlusion is ignored. Not in frame: all zeros.
    """
    return _project(model, data, _geometry(model, data, model.body(body_name).id), camera)


def target_box(model, data, target):
    """(2, 5): `box` of `prop_<target>` in front_cam and gripper_cam."""
    geometry = _geometry(model, data, model.body(f"prop_{target}").id)
    return np.stack([_project(model, data, geometry, camera) for camera in CAMERAS])


def props_box(model, data, names):
    """(len(names), 2, 5): `target_box` for every prop in `names`."""
    return np.stack([target_box(model, data, name) for name in names])
