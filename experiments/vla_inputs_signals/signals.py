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
