"""Prop geometry from the compiled collision geoms, for grasp planning and success.

Every prop is a single collision mesh, which MuJoCo collides as its convex hull, so the vertices
are the hull's (a box geom contributes its corners, which the hand-built test models use). All
numbers are metres. Bodies rest upright (body z is world up) apart from yaw.
"""
import mujoco
import numpy as np

_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


def _body_vertices(model, name):
    body = model.body(f"prop_{name}").id
    chunks = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not model.geom_contype[g]:
            continue
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            adr, num = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[adr:adr + num].astype(np.float64)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = _CORNERS * model.geom_size[g]
        else:
            continue
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        chunks.append(local @ rotation.reshape(3, 3).T + model.geom_pos[g])
    if not chunks:
        raise ValueError(f"prop_{name} has no collision mesh")
    return np.vstack(chunks)


def bbox(model, name):
    """(centre, half-extents) of the collision hull's bounding box in the prop's body frame."""
    vertices = _body_vertices(model, name)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return (low + high) / 2, (high - low) / 2


def extents(model, name):
    return bbox(model, name)[1]


def _world_vertices(model, data, name):
    body = model.body(f"prop_{name}").id
    return _body_vertices(model, name) @ data.xmat[body].reshape(3, 3).T + data.xpos[body]


def bottom_z(model, data, name):
    return float(_world_vertices(model, data, name)[:, 2].min())


def top_z(model, data, name):
    return float(_world_vertices(model, data, name)[:, 2].max())


def centre_world(model, data, name):
    """World position of the hull's bounding-box centre (not the body origin, which is the centre of mass)."""
    body = model.body(f"prop_{name}").id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ bbox(model, name)[0]


def horizontal_width(model, data, name, yaw):
    """Width of the hull along the horizontal direction at world yaw `yaw` (radians)."""
    projection = _world_vertices(model, data, name)[:, :2] @ np.array([np.cos(yaw), np.sin(yaw)])
    return float(projection.max() - projection.min())


def _wrap_half_turn(angle):
    return float((angle + np.pi / 2) % np.pi - np.pi / 2)


def narrow_axis_yaw(model, data, name):
    """World yaw (radians, in [-pi/2, pi/2)) of the body's narrower horizontal axis."""
    half = extents(model, name)
    axis = 0 if half[0] <= half[1] else 1
    direction = data.xmat[model.body(f"prop_{name}").id].reshape(3, 3)[:, axis]
    return _wrap_half_turn(float(np.arctan2(direction[1], direction[0])))


def axis_yaws(model, data, name):
    """World yaws (radians, mod pi) of the body's horizontal x and y axes."""
    rotation = data.xmat[model.body(f"prop_{name}").id].reshape(3, 3)
    return tuple(float(np.arctan2(rotation[1, k], rotation[0, k]) % np.pi) for k in (0, 1))


def footprint_widths(model, data, name):
    """Hull widths along the body x axis, its 45-degree diagonal, the body y axis and the other diagonal.

    A round footprint has four equal widths; a square one has faces (0, 2) 1.41 times narrower than its
    diagonals (1, 3), which is why aspect alone cannot tell the two apart.
    """
    psi = axis_yaws(model, data, name)[0]
    return tuple(horizontal_width(model, data, name, psi + k * np.pi / 4) for k in range(4))
