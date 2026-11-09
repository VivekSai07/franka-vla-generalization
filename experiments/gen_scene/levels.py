"""Every level of G0's five axes, and the fixed rotation that gives each layout one level per axis. Pure; admission image.

Every number below is pinned by `tests/test_levels.py` and fixed before any policy trial (G0 spec D4, D5, D7, D13). Placement of
added objects, the outer start and the `move` destination are drawn here, from streams named by the layout's identity, and stored in
the plan: nothing is drawn at run time.

Reading of the spec where it was not literal (recorded in the README): `bright` scales the specular as well as the diffuse
light; `dark` leaves specular alone; `tinted`'s warm/cool alternates within the level; a bin shift moves both bins apart (+) or
together (-) and flips to apart when together would bring a bin within 2 cm of an object.
"""
import hashlib
import math
from collections import defaultdict

import numpy as np

from experiments.gen_scene import pools
from experiments.gen_scene.conditions import Condition, primitive_name
from experiments.props_p2_data import plan as p2
from experiments.props_scene import constants as psc

VERSION = "gen_scene_levels_v1"
MASTER = 20261109

AXES = ("lighting", "background", "distractors", "spatial", "disturbance")
LEVELS = {
    "lighting": ("dark", "bright", "tinted", "low_angle"),
    "background": ("table", "room", "both"),
    "distractors": ("three", "four", "five"),
    "spatial": ("table_up5", "table_down5", "table_up10", "outer_start"),
    "disturbance": ("move", "drop"),
}
OFFSETS = {"lighting": 0, "background": 1, "distractors": 2, "spatial": 1, "disturbance": 1}  # level of layout n = LEVELS[(n + offset) % len]

DARK = dict(light_diffuse_scale=0.3, ambient_scale=0.3)
BRIGHT = dict(light_diffuse_scale=2.5, light_specular_scale=2.5)
WARM, COOL = (1.0, 0.75, 0.5), (0.6, 0.75, 1.0)
LOW_ANGLE_ELEVATION_DEG = 20.0
LOW_ANGLE_HEADLIGHT_SCALE = 0.3
LOW_ANGLE_DIR = (0.0, -math.cos(math.radians(LOW_ANGLE_ELEVATION_DEG)), -math.sin(math.radians(LOW_ANGLE_ELEVATION_DEG)))

TABLE_OFFSETS_M = {"table_up5": 0.05, "table_down5": -0.05, "table_up10": 0.10}
OUTER_START_X_M, OUTER_START_Y_M = (0.42, 0.50), psc.START_Y_M
OUTER_BIN_SHIFT_M = 0.04

DISTRACTOR_PROPS = 3                       # prop distractors at every distractor level; primitives come on top
PRIMITIVES = {"three": 0, "four": 1, "five": 2}
DISTRACTOR_REGION_X_M, DISTRACTOR_REGION_Y_M = (0.25, 0.52), (-0.20, 0.20)
REGION_RADIUS_MIN_M = psc.START_RADIUS_MIN_M

MOVE_DISTANCE_M = 0.08                     # the target is moved this far
MOVE_TRIGGER_M = 0.06                      # when the hand first comes this close horizontally, before any close command
MOVE_REGION_X_M, MOVE_REGION_Y_M = (0.25, 0.46), (-0.16, 0.16)
MOVE_GAP_M = 0.01                          # the last-resort rule: footprint circles at least this far apart
DROP_LIFT_M = 0.05                         # `drop` fires when the target first rises this far above its start
DROP_OPEN_TICKS = 10                       # the gripper command is forced open this many ticks
NOMINAL_TICKS, DISTURBANCE_TICKS = psc.MAX_TICKS, 1200
LIFT_SUCCESS_M = 0.05                      # `grasp_lift`: the target rose at least this far at any tick

BIN_OUTER_HALF_M = psc.BIN_INNER_M / 2 + psc.BIN_WALL_THICKNESS_M
BIN_CLEARANCE_M = 0.02
DRAW_LIMIT = 20_000
RESTARTS, TRIES_PER_OBJECT = 400, 300       # distractor placement: objects are placed one at a time, the set restarts when one is stuck
PRIMITIVE_YAW_MAX_DEG = 90.0
PAIR_GAP_M = p2.PAIR_GAP_M


def ticks_for(condition):
    return DISTURBANCE_TICKS if condition.disturbance != "none" else NOMINAL_TICKS


def stream(identity):
    value = int.from_bytes(hashlib.sha256(f"{VERSION}:{MASTER}:{identity}".encode()).digest()[:4], "big")
    return np.random.RandomState(value)


def _uniform(rng, low, high):
    return float(low + (high - low) * rng.uniform())


# --- geometry of the objects ---------------------------------------------------------------------------------------------

def extent(name):
    """(footprint radius, height) of a prop or a primitive named `<shape>_<colour>`."""
    if name in p2.FOOTPRINT_RADIUS_M:
        return p2.FOOTPRINT_RADIUS_M[name], p2.HEIGHT_M[name]
    return pools.primitive_extent(name.split("_")[0])


def min_distance(a, b):
    ra, rb = extent(a)[0], extent(b)[0]
    return max(ra + rb + PAIR_GAP_M, p2.FINGER_REACH_M + max(ra, rb))


def hand_clear(a, pose_a, b, pose_b):
    """P2's rule (`props_p2_data.plan.hand_clear`) for any object `b`: grasping the prop `a`, no hand part comes down onto `b`."""
    grasp_z = p2.HEIGHT_M[a] - min(p2.GRASP_BELOW_TOP_M, p2.HEIGHT_M[a] / 2)
    dx, dy = pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]
    rb, hb = extent(b)
    for half_along, half_across, underside in p2.HAND_PARTS_M.values():
        if hb <= grasp_z + underside - p2.HAND_MARGIN_M:
            continue
        for axis in p2.finger_axes_deg(a, pose_a[2]):
            c, s = math.cos(math.radians(axis)), math.sin(math.radians(axis))
            along, across = dx * c + dy * s, -dx * s + dy * c
            if abs(along) <= half_along + rb + p2.HAND_MARGIN_M and abs(across) <= half_across + rb + p2.HAND_MARGIN_M:
                return False
    return True


def bins_clear(name, pose, bin_shift_m=0.0):
    """True if the object's footprint circle is BIN_CLEARANCE_M or more from both bins' outer boxes."""
    radius = extent(name)[0]
    for sign in (1, -1):
        cx, cy = psc.SLOT_XY["left"][0], sign * (psc.SLOT_XY["left"][1] + bin_shift_m)
        dx, dy = max(abs(pose[0] - cx) - BIN_OUTER_HALF_M, 0.0), max(abs(pose[1] - cy) - BIN_OUTER_HALF_M, 0.0)
        if math.hypot(dx, dy) < radius + BIN_CLEARANCE_M:
            return False
    return True


def spaced(items, bin_shift_m=0.0):
    """`items`: [(name, pose)]. Pairwise far enough apart; every prop graspable without the hand touching another object;
    no object against a bin. Props are checked in both directions (like P2's), primitives are only ever the neighbour."""
    for name, pose in items:
        if not bins_clear(name, pose, bin_shift_m):
            return False
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a, pa), (b, pb) = items[i], items[j]
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < min_distance(a, b):
                return False
            if a in p2.HEIGHT_M and not hand_clear(a, pa, b, pb):
                return False
            if b in p2.HEIGHT_M and not hand_clear(b, pb, a, pa):
                return False
    return True


def free(items, bin_shift_m=0.0):
    """Only physical clearance: no object against a bin, every pair at least P2's `min_distance` apart."""
    for name, pose in items:
        if not bins_clear(name, pose, bin_shift_m):
            return False
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a, pa), (b, pb) = items[i], items[j]
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < min_distance(a, b):
                return False
    return True


def physical(items, bin_shift_m=0.0):
    """Only that nothing overlaps: no object against a bin, footprint circles at least MOVE_GAP_M apart."""
    for name, pose in items:
        if not bins_clear(name, pose, bin_shift_m):
            return False
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a, pa), (b, pb) = items[i], items[j]
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < extent(a)[0] + extent(b)[0] + MOVE_GAP_M:
                return False
    return True


MOVE_RULES = (spaced, free, physical)


def _pose(rng, x_range, y_range, name):
    yaw_max = p2.YAW_MAX_DEG_BY_OBJECT[name] if name in p2.YAW_MAX_DEG_BY_OBJECT else PRIMITIVE_YAW_MAX_DEG
    while True:
        x, y = _uniform(rng, *x_range), _uniform(rng, *y_range)
        yaw = _uniform(rng, -yaw_max, yaw_max)
        if math.hypot(x, y) >= REGION_RADIUS_MIN_M:
            return (x, y, yaw)


def _fixed(layout):
    return [(layout["object"], tuple(layout["start"])), *((d["object"], tuple(d["pose"])) for d in layout["distractors"])]


# --- levels --------------------------------------------------------------------------------------------------------------

def lighting(level, j):
    """`j`: how many layouts of this level came before (tinted alternates warm, cool)."""
    if level == "dark":
        return Condition(axis="lighting", level=level, **DARK)
    if level == "bright":
        return Condition(axis="lighting", level=level, **BRIGHT)
    if level == "tinted":
        return Condition(axis="lighting", level=level, tint=WARM if j % 2 == 0 else COOL)
    if level == "low_angle":
        return Condition(axis="lighting", level=level, light_dir=LOW_ANGLE_DIR, light_shadows=True,
                         headlight_scale=LOW_ANGLE_HEADLIGHT_SCALE)
    raise ValueError(level)


def background(level, j, manifest=None):
    """Test-pool textures, each used in turn so a level covers the pool evenly."""
    both = pools.pools(manifest)
    table, room = both["table"]["test"], both["room"]["test"]
    values = {}
    if level in ("table", "both"):
        values["table_texture"] = table[j % len(table)]
    if level in ("room", "both"):
        values["floor_texture"], values["wall_texture"] = room[j % len(room)], room[(j + len(room) // 2) % len(room)]
    if not values:
        raise ValueError(level)
    return Condition(axis="background", level=level, pool="test", **values)


def distractors(level, layout, identity):
    """Prop distractors to DISTRACTOR_PROPS (the layout's own are kept), then primitives from test pool B."""
    rng = stream(f"{identity}:distractors")
    base = [d["object"] for d in layout["distractors"]]
    candidates = [o for o in p2.OBJECTS if o != layout["object"] and o not in base]
    added = [candidates[i] for i in rng.permutation(len(candidates))[:DISTRACTOR_PROPS - len(base)]]
    combos = [(s, c) for s in pools.PRIMITIVE_SHAPES["B"] for c in pools.PRIMITIVE_COLOURS["B"]]
    prims = [combos[i] for i in rng.permutation(len(combos))[:PRIMITIVES[level]]]
    fixed = _fixed(layout)
    names = [*added, *(primitive_name(*p) for p in prims)]
    for _ in range(RESTARTS):
        placed = []
        for name in names:
            for _ in range(TRIES_PER_OBJECT):
                pose = _pose(rng, DISTRACTOR_REGION_X_M, DISTRACTOR_REGION_Y_M, name)
                if spaced(fixed + placed + [(name, pose)]):
                    placed.append((name, pose))
                    break
            else:
                break
        else:
            poses = [pose for _, pose in placed]
            return Condition(axis="distractors", level=level, pool="test",
                             extra_props=tuple((n, *p) for n, p in zip(added, poses[:len(added)])),
                             primitives=tuple((s, c, *p) for (s, c), p in zip(prims, poses[len(added):])))
    raise RuntimeError(f"{identity}: no spaced placement for the {level} distractors in {RESTARTS} restarts")


def spatial(level, layout, identity, j):
    if level in TABLE_OFFSETS_M:
        return Condition(axis="spatial", level=level, table_offset_m=TABLE_OFFSETS_M[level])
    if level != "outer_start":
        raise ValueError(level)
    rng = stream(f"{identity}:outer_start")
    others = [(d["object"], tuple(d["pose"])) for d in layout["distractors"]]
    sign = 1.0 if j % 2 == 0 else -1.0     # even: bins apart; odd: bins together, unless that touches an object
    for _ in range(DRAW_LIMIT):
        pose = _pose(rng, OUTER_START_X_M, OUTER_START_Y_M, layout["object"])
        for shift in (sign * OUTER_BIN_SHIFT_M, OUTER_BIN_SHIFT_M):
            if spaced([(layout["object"], pose), *others], shift):
                return Condition(axis="spatial", level=level, start=pose, bin_shift_m=shift)
    raise RuntimeError(f"{identity}: no outer start in {DRAW_LIMIT} draws")


def disturbance(level, layout, identity):
    if level == "drop":
        return Condition(axis="disturbance", level=level, disturbance="drop")
    if level != "move":
        raise ValueError(level)
    rng = stream(f"{identity}:move")
    others = [(d["object"], tuple(d["pose"])) for d in layout["distractors"]]
    start = layout["start"]
    for rule in MOVE_RULES:   # P2's full spacing first, then only P2's distance, then only non-overlap; the expert pass (D8) decides the rest
        for _ in range(DRAW_LIMIT):
            angle = _uniform(rng, 0.0, 2 * math.pi)
            x, y = start[0] + MOVE_DISTANCE_M * math.cos(angle), start[1] + MOVE_DISTANCE_M * math.sin(angle)
            if (MOVE_REGION_X_M[0] <= x <= MOVE_REGION_X_M[1] and MOVE_REGION_Y_M[0] <= y <= MOVE_REGION_Y_M[1]
                    and math.hypot(x, y) >= REGION_RADIUS_MIN_M and rule([(layout["object"], (x, y, start[2])), *others])):
                return Condition(axis="disturbance", level=level, disturbance="move", move_to=(x, y))
    raise RuntimeError(f"{identity}: no free spot for the move in {DRAW_LIMIT} draws")


def level_of(axis, n):
    levels = LEVELS[axis]
    return levels[(n + OFFSETS[axis]) % len(levels)]


def assign(layouts, manifest=None):
    """{layout_id: {axis: Condition}} for the layouts in tier order. Layout n gets level (n + offset) % levels on each axis, so
    every level is used equally often; `j` counts the layouts that already had that level."""
    counts, out = defaultdict(int), {}
    for n, layout in enumerate(layouts):
        conditions = {}
        for axis in AXES:
            level = level_of(axis, n)
            j = counts[(axis, level)]
            counts[(axis, level)] += 1
            identity = layout["identity"]
            if axis == "lighting":
                condition = lighting(level, j)
            elif axis == "background":
                condition = background(level, j, manifest)
            elif axis == "distractors":
                condition = distractors(level, layout, identity)
            elif axis == "spatial":
                condition = spatial(level, layout, identity, j)
            else:
                condition = disturbance(level, layout, identity)
            conditions[axis] = condition.check(manifest)
        out[layout["layout_id"]] = conditions
    return out
