"""Bin success: where the prop is, whether the gripper has let go, and the 30-tick streak.

The prop's position is its body origin, which MuJoCo puts at the centre of mass. Imports no robosuite,
so it runs in either image.
"""
import numpy as np

from experiments.props_scene import constants as c


def in_bin(model, data, name, bin_colour):
    if bin_colour not in c.BIN_RGBA:
        raise ValueError(f"bin must be one of {sorted(c.BIN_RGBA)}, got {bin_colour!r}")
    prop = model.body(f"prop_{name}").id
    delta = data.xpos[prop] - data.xpos[model.body(f"bin_{bin_colour}").id]
    margin = c.BIN_INNER_M / 2 - max(abs(float(delta[0])), abs(float(delta[1])))
    z = float(delta[2])
    inside = margin > 0.0 and z < c.BIN_WALL_HEIGHT_M
    fingers = {model.body("left_finger").id, model.body("right_finger").id}
    touching = False
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if prop in bodies and bodies & fingers:
            touching = True
            break
    open_ = bool(data.ctrl[-1] > c.GRIPPER_OPEN_CTRL)
    return {"inside": bool(inside), "xy_margin_m": float(margin), "z_m": z, "gripper_open": open_,
            "finger_contact": touching, "success_tick": bool(inside and open_ and not touching)}


def placement(model, data, name, bin_colour):
    """Diagnostic, never part of success: where the prop rests, for objects taller than the bin is deep.

    A 15 cm box standing on the bin floor has its centre above the 7 cm wall top, so `in_bin` calls it
    outside; this records that it is upright, centred and resting on the floor.
    """
    from experiments.props_scene import geometry
    prop = model.body(f"prop_{name}").id
    delta = data.xpos[prop] - data.xpos[model.body(f"bin_{bin_colour}").id]
    margin = c.BIN_INNER_M / 2 - max(abs(float(delta[0])), abs(float(delta[1])))
    bottom = geometry.bottom_z(model, data, name)
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[prop].reshape(3, 3)[2, 2], -1.0, 1.0))))
    return {"bottom_z_m": bottom, "top_z_m": geometry.top_z(model, data, name), "tilt_deg": tilt,
            "xy_margin_m": float(margin), "resting_in_bin": bool(margin > 0.0 and bottom < c.BIN_WALL_HEIGHT_M)}


def success_streak(stable, tick):
    stable = stable + 1 if tick["success_tick"] else 0
    return stable, stable >= c.SUCCESS_TICKS
