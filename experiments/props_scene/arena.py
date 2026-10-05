import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from robosuite.utils.mjcf_utils import new_element

from experiments.props_scene import constants as c
from experiments.robosuite_scene.arena import CopiedCameraTableArena

class BinsTableArena(CopiedCameraTableArena):
    """The B3 table and front camera plus two static open bins, one per slot."""

    def __init__(self, blue_slot):
        if blue_slot not in c.SLOT_XY:
            raise ValueError(f"blue_slot must be one of {sorted(c.SLOT_XY)}, got {blue_slot!r}")
        super().__init__()
        orange_slot = "right" if blue_slot == "left" else "left"
        for colour, slot in (("blue", blue_slot), ("orange", orange_slot)):
            self.worldbody.append(_bin(colour, c.SLOT_XY[slot]))


def _geom(name, pos, half, rgba):
    # group 1 so the contract's "collision geoms out of the render" pass leaves it drawn
    return new_element("geom", name, type="box", pos=" ".join(repr(float(v)) for v in pos),
                       size=" ".join(repr(float(v)) for v in half),
                       rgba=" ".join(repr(float(v)) for v in rgba), group="1", contype="1", conaffinity="1")


def _bin(colour, xy):
    half_inner = c.BIN_INNER_M / 2
    wall = c.BIN_WALL_THICKNESS_M
    outer = half_inner + wall
    height = c.BIN_WALL_HEIGHT_M
    rgba = c.BIN_RGBA[colour]
    body = new_element("body", f"bin_{colour}", pos=f"{xy[0]!r} {xy[1]!r} 0.0")
    floor = c.BIN_FLOOR_THICKNESS_M
    body.append(_geom(f"bin_{colour}_floor", (0, 0, floor / 2), (outer, outer, floor / 2), rgba))
    for sign in (-1, 1):
        offset = half_inner + wall / 2
        body.append(_geom(f"bin_{colour}_wall_x{'p' if sign > 0 else 'n'}", (sign * offset, 0, height / 2),
                          (wall / 2, outer, height / 2), rgba))
        body.append(_geom(f"bin_{colour}_wall_y{'p' if sign > 0 else 'n'}", (0, sign * offset, height / 2),
                          (outer, wall / 2, height / 2), rgba))
    return body
