"""P2 collection plan: targets, distractors, spaced layouts, 120 reserved layouts and the expert re-test.

Pure Python + numpy; imports only `experiments.props_scene.constants` (also pure), so a training
contract can import it in the admission image. Balance is by construction (shuffled decks), never by
sampling. Spec: docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md.
"""
import hashlib
import itertools
import json
import math

import numpy as np

from experiments.props_scene import constants as psc

VERSION = "props_p2_v1"
MASTER = 20261013
SPEC = "docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md"
PLAN = "docs/superpowers/plans/2026-10-13-props-p2-collection.md"

OBJECTS = psc.OBJECTS
BIN_COLOURS = psc.BINS
SLOTS = ("left", "right")
PAIRS = tuple(itertools.product(OBJECTS, BIN_COLOURS))
HELD_OUT_PAIRS = (("milk", "blue"), ("bread", "orange"))
TRAINED_PAIRS = tuple(pair for pair in PAIRS if pair not in HELD_OUT_PAIRS)
CELLS = tuple((obj, bin_, slot) for obj, bin_ in PAIRS for slot in SLOTS)
TRAINED_CELLS = tuple((obj, bin_, slot) for obj, bin_ in TRAINED_PAIRS for slot in SLOTS)
HELD_OUT_CELLS = tuple((obj, bin_, slot) for obj, bin_ in HELD_OUT_PAIRS for slot in SLOTS)

DISTRACTOR_COUNTS = (1, 2)
TRAIN_PER_CELL_PER_COUNT = 12
VALIDATION_PER_CELL = 3
TRAIN_COUNT = TRAIN_PER_CELL_PER_COUNT * len(DISTRACTOR_COUNTS) * len(TRAINED_CELLS)
VALIDATION_COUNT = VALIDATION_PER_CELL * len(TRAINED_CELLS)
ADMIT_COUNT = TRAIN_COUNT + VALIDATION_COUNT
EVAL_REPORT_PER_HELD_OUT_CELL = 5
GATE_PER_CELL_PER_COUNT = 6

START_X_M = psc.START_X_M
START_Y_M = psc.START_Y_M
START_RADIUS_MIN_M = psc.START_RADIUS_MIN_M
YAW_MAX_DEG_BY_OBJECT = dict(psc.YAW_MAX_DEG_BY_OBJECT)
HOME_JITTER_DEG_MAX = psc.HOME_JITTER_DEG_MAX

# Compiled collision-hull bounding boxes (measured 2026-10-13): half-diagonal, rounded up to the mm.
FOOTPRINT_RADIUS_M = {"can": 0.036, "bread": 0.032, "lemon": 0.040, "milk": 0.029, "bottle": 0.041, "cereal": 0.053}
HEIGHT_M = {"can": 0.080, "bread": 0.048, "lemon": 0.040, "milk": 0.144, "bottle": 0.160, "cereal": 0.150}
PAIR_GAP_M = 0.03
FINGER_REACH_M = 0.065   # open finger gap 0.08 / 2 plus pad, from the target's centre
CARRY_BOTTOM_M = 0.19    # tallest prop plus 3 cm
# Hand parts that come down beside the grasp (measured 2026-10-13 on the compiled scene, finger-axis frame):
# (half-length along the finger axis, half-width across it, underside height above the grasp point).
# The first draft spaced only the fingers; the gate then found the housing clipping tall neighbours (20/288).
HAND_PARTS_M = {"housing": (0.104, 0.032, 0.042), "link7": (0.088, 0.044, 0.108)}
HAND_MARGIN_M = 0.01
GRASP_BELOW_TOP_M = psc.GRASP_BELOW_TOP_M
ROUND_PROPS = ("can", "bottle")   # the expert's finger axis is fixed at 90 deg for these
SQUARE_PROPS = ("milk",)          # the expert picks either face; both axes are checked

EXPECTED_EPISODE_FRAMES = 452
ATTEMPTS_PER_LAYOUT = 3
MAX_SUBSTITUTIONS = 3
LAYOUT_DRAW_LIMIT = 100_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed(identity):
    return int.from_bytes(hashlib.sha256(f"{VERSION}:{MASTER}:{identity}".encode()).digest()[:4], "big")


def _stream(identity, seeds=None):
    value = seed(identity)
    if seeds is not None:
        seeds[identity] = value
    return np.random.RandomState(value)


def _uniform(stream, low, high):
    return float(low + (high - low) * stream.uniform())


def instruction(obj, bin_colour):
    return f"put the {obj} in the {bin_colour} bin"


def min_distance(a, b):
    """Centre-to-centre floor for two objects. Symmetric, so spacing never singles out the target."""
    ra, rb = FOOTPRINT_RADIUS_M[a], FOOTPRINT_RADIUS_M[b]
    return max(ra + rb + PAIR_GAP_M, FINGER_REACH_M + max(ra, rb))


def finger_axes_deg(obj, yaw_deg):
    """World yaw(s) of the finger axis the expert will use (measured against `PropsExpert.plan_grasp`)."""
    if obj in ROUND_PROPS:
        return [90.0]
    if obj in SQUARE_PROPS:
        return [90.0 + yaw_deg, yaw_deg]
    return [90.0 + yaw_deg]


def hand_clear(a, pose_a, b, pose_b):
    """True if, grasping `a`, no hand part comes down onto `b`."""
    grasp_z = HEIGHT_M[a] - min(GRASP_BELOW_TOP_M, HEIGHT_M[a] / 2)
    dx, dy, rb = pose_b[0] - pose_a[0], pose_b[1] - pose_a[1], FOOTPRINT_RADIUS_M[b]
    for half_along, half_across, underside in HAND_PARTS_M.values():
        if HEIGHT_M[b] <= grasp_z + underside - HAND_MARGIN_M:
            continue
        for axis in finger_axes_deg(a, pose_a[2]):
            c, s = math.cos(math.radians(axis)), math.sin(math.radians(axis))
            along, across = dx * c + dy * s, -dx * s + dy * c
            if abs(along) <= half_along + rb + HAND_MARGIN_M and abs(across) <= half_across + rb + HAND_MARGIN_M:
                return False
    return True


def spaced(names, poses):
    """Every pair far enough apart, and every object graspable without the hand touching another."""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b, pa, pb = names[i], names[j], poses[i], poses[j]
            if (math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < min_distance(a, b)
                    or not hand_clear(a, pa, b, pb) or not hand_clear(b, pb, a, pa)):
                return False
    return True


def _draw_pose(stream, obj):
    while True:
        x, y = _uniform(stream, *START_X_M), _uniform(stream, *START_Y_M)
        yaw = _uniform(stream, -YAW_MAX_DEG_BY_OBJECT[obj], YAW_MAX_DEG_BY_OBJECT[obj])
        if math.hypot(x, y) >= START_RADIUS_MIN_M:
            return [x, y, yaw]


def _draw_layout(target, distractors, identity, seeds=None):
    """Target start and distractor poses, all in the region and pairwise spaced; whole layout redrawn on a clash."""
    stream = _stream(f"{identity}:pose", seeds)
    names = [target, *distractors]
    for draws in range(1, LAYOUT_DRAW_LIMIT + 1):
        poses = [_draw_pose(stream, name) for name in names]
        if spaced(names, poses):
            return {"start": poses[0], "distractors": [{"object": n, "pose": q} for n, q in zip(distractors, poses[1:])],
                    "layout_draws": draws}
    raise RuntimeError(f"no spaced layout for {names} in {LAYOUT_DRAW_LIMIT} draws")


def _jitter(identity, seeds=None):
    stream = _stream(f"{identity}:home", seeds)
    return [_uniform(stream, -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX) for _ in range(7)]


def _shuffled(cards, identity, seeds):
    order = _stream(identity, seeds).permutation(len(cards))
    return [cards[i] for i in order]


def _options(target, count):
    return [list(combo) for combo in itertools.combinations([o for o in OBJECTS if o != target], count)]


def _deck(target, count, n, identity, seeds):
    """n distractor sets for one target and count: every option cycled before any repeats, then shuffled."""
    options = _options(target, count)
    cards = (options * (n // len(options) + 1))
    return _shuffled(cards, identity, seeds)[:n]


def _pick(target, count, identity, seeds):
    if count == 0:
        return []
    options = _options(target, count)
    return options[int(_stream(f"{identity}:distractors", seeds).randint(len(options)))]


def _layout_entry(identity, cell, distractors, seeds):
    obj, bin_colour, slot = cell
    expert_seed = seed(f"{identity}:expert")
    seeds[f"{identity}:expert"] = expert_seed
    return dict(object=obj, bin=bin_colour, blue_slot=slot, cell=list(cell), distractor_count=len(distractors),
                instruction=instruction(obj, bin_colour), **_draw_layout(obj, distractors, identity, seeds),
                home_pose_jitter_deg=_jitter(identity, seeds), seed=expert_seed)


def attempt_draw(episode, k):
    """Seed and home jitter for attempt k of a slot; k=0 is the planned draw."""
    if type(k) is not int or k < 0:
        raise ValueError("attempt index must be a non-negative integer")
    if k == 0:
        return {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    prefix = f"attempt:{episode['episode_index']}:{k}"
    return {"seed": seed(f"{prefix}:expert"), "home_pose_jitter_deg": _jitter(prefix)}


def resample_start(episode, r):
    """The same slot and the same objects with a freshly drawn spaced layout; r=0 is the planned one."""
    if type(r) is not int or r < 0:
        raise ValueError("resample index must be a non-negative integer")
    if r == 0:
        return episode
    prefix = f"resample:{episode['episode_index']}:{r}"
    names = [d["object"] for d in episode["distractors"]]
    return {**episode, "layout_id": f"{episode['layout_id']}_r{r}", **_draw_layout(episode["object"], names, prefix),
            "home_pose_jitter_deg": _jitter(prefix)}


def _training_episodes(seeds):
    decks = {(cell, n): _deck(cell[0], n, TRAIN_PER_CELL_PER_COUNT, f"deck:train:{'-'.join(cell)}:{n}", seeds)
             for cell in TRAINED_CELLS for n in DISTRACTOR_COUNTS}
    entries = [(cell, n) for cell in TRAINED_CELLS for n in DISTRACTOR_COUNTS for _ in range(TRAIN_PER_CELL_PER_COUNT)]
    episodes = []
    for local, (cell, n) in enumerate(_shuffled(entries, "order:train", seeds)):
        identity = f"episode:train:{local}"
        episode = _layout_entry(identity, cell, decks[(cell, n)].pop(), seeds)
        episodes.append(dict(episode_index=local, split="train", layout_id=f"train_{local:03d}", **episode))
    return episodes


def _validation_episodes(start, seeds):
    cells = [cell for cell in TRAINED_CELLS for _ in range(VALIDATION_PER_CELL)]
    counts = _shuffled([1, 2] * (len(cells) // 2), "counts:validation", seeds)
    episodes = []
    for local, (cell, n) in enumerate(zip(_shuffled(cells, "order:validation", seeds), counts)):
        identity = f"episode:validation:{local}"
        episode = _layout_entry(identity, cell, _pick(cell[0], n, identity, seeds), seeds)
        episodes.append(dict(episode_index=start + local, split="validation", layout_id=f"validation_{local:03d}",
                             **episode))
    return episodes


def _reserved(seeds):
    plan = []  # (tier, cell, count)
    for tier in ("screen", "confirm"):
        counts = _shuffled([1, 2] * (len(TRAINED_CELLS) // 2), f"counts:{tier}", seeds)
        plan += [(tier, cell, n) for cell, n in zip(TRAINED_CELLS, counts)]
    plan += [("report", cell, n) for n in DISTRACTOR_COUNTS for cell in TRAINED_CELLS]
    held = [cell for cell in HELD_OUT_CELLS for _ in range(EVAL_REPORT_PER_HELD_OUT_CELL)]
    plan += [("report", cell, n) for cell, n in zip(held, _shuffled([1, 2] * (len(held) // 2), "counts:report_held", seeds))]
    plan += [("control", cell, 0) for cell in TRAINED_CELLS]
    holdout, local = {}, {}
    for tier, cell, n in plan:
        index = local.get(tier, 0)
        local[tier] = index + 1
        key = f"{tier}_{index:03d}"
        identity = f"evaluation:{tier}:{index}"
        entry = _layout_entry(identity, cell, _pick(cell[0], n, identity, seeds), seeds)
        holdout[key] = dict(tier=tier, layout_id=key, held_out=(cell[0], cell[1]) in HELD_OUT_PAIRS, **entry)
    return holdout


def _gate_attempts(seeds):
    attempts = []
    for cell in CELLS:
        for n in DISTRACTOR_COUNTS:
            for _ in range(GATE_PER_CELL_PER_COUNT):
                index = len(attempts)
                identity = f"gate:{index}"
                entry = _layout_entry(identity, cell, _pick(cell[0], n, identity, seeds), seeds)
                attempts.append(dict(episode_index=index, split="gate", layout_id=f"gate_{index:03d}", **entry))
    return attempts


def shard_slots(k, shards, total=ADMIT_COUNT):
    """Contiguous slot block of shard k; the blocks partition 0..total-1 in order."""
    if total % shards or not 0 <= k < shards:
        raise ValueError(f"shard {k} of {shards} does not partition {total} slots")
    size = total // shards
    return list(range(k * size, (k + 1) * size))


def make_plan() -> dict:
    seeds = {}
    episodes = _training_episodes(seeds)
    episodes += _validation_episodes(len(episodes), seeds)
    holdout = _reserved(seeds)
    gate = _gate_attempts(seeds)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; stop without redraw")
    return dict(
        version=VERSION, master_seed=MASTER, rng="numpy.random.RandomState", rng_seeds=seeds,
        seed_derivation="SHA256(version:master:identity) first four bytes big endian",
        spec=SPEC, plan=PLAN, objects=list(OBJECTS), bin_colours=list(BIN_COLOURS), slots=list(SLOTS),
        held_out_pairs=[list(x) for x in HELD_OUT_PAIRS], trained_pairs=[list(x) for x in TRAINED_PAIRS],
        trained_cells=[list(c) for c in TRAINED_CELLS], held_out_cells=[list(c) for c in HELD_OUT_CELLS],
        start_region=dict(x=list(START_X_M), y=list(START_Y_M), radius_min_m=START_RADIUS_MIN_M,
                          yaw_max_deg_by_object=YAW_MAX_DEG_BY_OBJECT),
        spacing=dict(footprint_radius_m=FOOTPRINT_RADIUS_M, pair_gap_m=PAIR_GAP_M, finger_reach_m=FINGER_REACH_M,
                     hand_parts_m=HAND_PARTS_M, hand_margin_m=HAND_MARGIN_M, height_m=HEIGHT_M),
        carry_bottom_m=CARRY_BOTTOM_M, episodes=episodes, train_ids=list(range(TRAIN_COUNT)),
        validation_ids=list(range(TRAIN_COUNT, ADMIT_COUNT)), evaluation_holdout=holdout,
        **{f"evaluation_{tier}_ids": sorted(k for k, v in holdout.items() if v["tier"] == tier)
           for tier in ("screen", "report", "control", "confirm")},
        gate_attempts=gate, home_pose_jitter_deg_max=float(HOME_JITTER_DEG_MAX),
        attempts_per_layout=ATTEMPTS_PER_LAYOUT, max_substitutions=MAX_SUBSTITUTIONS,
        expected_episode_frames=EXPECTED_EPISODE_FRAMES, fps=30, state_dim=9, action_dim=8,
    )


def validate_plan(plan):
    if digest(plan) != digest(make_plan()):
        raise ValueError("Noncanonical props P2 plan; no count/seed/layout overrides")
