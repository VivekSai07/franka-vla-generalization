"""Prop geometry from the compiled collision meshes, for grasp planning.

Every prop is a single collision mesh, which MuJoCo collides as its convex hull, so the vertices
are the hull's. All numbers are metres. Bodies rest upright (body z is world up) apart from yaw.
"""
import mujoco
import numpy as np


def _body_vertices(model, name):
    body = model.body(f"prop_{name}").id
    chunks = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not model.geom_contype[g] or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[g])
        adr, num = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        chunks.append(model.mesh_vert[adr:adr + num].astype(np.float64) @ rotation.reshape(3, 3).T + model.geom_pos[g])
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


def aspect(model, name):
    """Long horizontal half-extent over the short one; 1.0 for a round or square footprint."""
    half = extents(model, name)[:2]
    return float(half.max() / half.min())
