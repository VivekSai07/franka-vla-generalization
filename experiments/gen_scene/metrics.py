"""Where a trial's target ended, measured against the bin it is in. Pure numpy; imports no simulator module (takes a model/data).

`props_scene.metrics.placement` compares the prop's lowest point with the wall height as an ABSOLUTE z, which is only right when
the table's top is at z = 0. With the table raised or lowered it calls a prop resting in a bin "not in it". Here the height is
taken from the bin's own origin, so it is the same test at every table height; at height 0 it returns the same numbers.
"""
import numpy as np

from experiments.props_scene import constants as c


def placement(model, data, name, bin_colour):
    """Tilt, top height and bin test of the resting prop, relative to the bin."""
    from experiments.props_scene import geometry
    prop = model.body(f"prop_{name}").id
    bin_origin = data.xpos[model.body(f"bin_{bin_colour}").id]
    delta = data.xpos[prop] - bin_origin
    margin = c.BIN_INNER_M / 2 - max(abs(float(delta[0])), abs(float(delta[1])))
    bottom = geometry.bottom_z(model, data, name)
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[prop].reshape(3, 3)[2, 2], -1.0, 1.0))))
    return {"bottom_z_m": bottom, "top_z_m": geometry.top_z(model, data, name), "tilt_deg": tilt,
            "xy_margin_m": float(margin), "resting_in_bin": bool(margin > 0.0 and bottom - float(bin_origin[2]) < c.BIN_WALL_HEIGHT_M)}
