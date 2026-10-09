"""P1 instruction-sorting collection plan: cells, held-out pairs, starts and reserved layouts.

Pure Python + numpy; imports only `experiments.props_scene.constants` (also pure), so a training
contract can import it in the admission image. Balance is enforced by construction (every trained
cell gets exactly the same count), never by sampling. Spec:
docs/superpowers/specs/2026-10-05-props-sorting-design.md, section "P1 — instruction sorting,
one object, no distractors".
"""
import hashlib
import itertools
import json
import math

import numpy as np

from experiments.props_scene import constants as psc

VERSION = "props_p1_v1"
MASTER = 20261006
SPEC = "docs/superpowers/specs/2026-10-05-props-sorting-design.md"
PLAN = "docs/superpowers/plans/2026-10-06-props-p1-collection.md"

OBJECTS = psc.OBJECTS
BIN_COLOURS = psc.BINS
SLOTS = ("left", "right")

PAIRS = tuple(itertools.product(OBJECTS, BIN_COLOURS))
HELD_OUT_PAIRS = (("milk", "blue"), ("bread", "orange"))
TRAINED_PAIRS = tuple(p for p in PAIRS if p not in HELD_OUT_PAIRS)
CELLS = tuple((obj, bin_, slot) for obj, bin_ in PAIRS for slot in SLOTS)
TRAINED_CELLS = tuple((obj, bin_, slot) for obj, bin_ in TRAINED_PAIRS for slot in SLOTS)
HELD_OUT_CELLS = tuple((obj, bin_, slot) for obj, bin_ in HELD_OUT_PAIRS for slot in SLOTS)

TRAIN_PER_CELL = 24
VALIDATION_PER_CELL = 3
TRAIN_COUNT = TRAIN_PER_CELL * len(TRAINED_CELLS)
VALIDATION_COUNT = VALIDATION_PER_CELL * len(TRAINED_CELLS)
ADMIT_COUNT = TRAIN_COUNT + VALIDATION_COUNT

EVAL_SCREEN_COUNT = len(TRAINED_CELLS)
EVAL_REPORT_PER_HELD_OUT_CELL = 5
EVAL_REPORT_COUNT = len(TRAINED_CELLS) + EVAL_REPORT_PER_HELD_OUT_CELL * len(HELD_OUT_CELLS)
EVAL_CONFIRM_COUNT = len(TRAINED_CELLS)
EVAL_COUNT = EVAL_SCREEN_COUNT + EVAL_REPORT_COUNT + EVAL_CONFIRM_COUNT

HOME_JITTER_DEG_MAX = psc.HOME_JITTER_DEG_MAX
START_X_M = psc.START_X_M
START_Y_M = psc.START_Y_M
START_RADIUS_MIN_M = psc.START_RADIUS_MIN_M
YAW_MAX_DEG_BY_OBJECT = dict(psc.YAW_MAX_DEG_BY_OBJECT)
EXPECTED_EPISODE_FRAMES = 452
ATTEMPTS_PER_LAYOUT = 3
MAX_SUBSTITUTIONS = 3


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


def cell_name(cell):
    return "-".join(cell)


def _draw_start(obj, identity, seeds=None):
    """Reject draws inside the reach limit, as `props_scene.benchmark` does; cap yaw per object."""
    stream = _stream(f"{identity}:pose", seeds)
    cap = YAW_MAX_DEG_BY_OBJECT[obj]
    for _ in range(100_000):
        x = _uniform(stream, *START_X_M)
        y = _uniform(stream, *START_Y_M)
        if math.hypot(x, y) >= START_RADIUS_MIN_M:
            return [x, y, _uniform(stream, -cap, cap)]
    raise RuntimeError("could not draw a start pose clear of the reach limit")


def _jitter(identity, seeds=None):
    stream = _stream(f"{identity}:home", seeds)
    return [_uniform(stream, -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX) for _ in range(7)]


def _shuffled(cards, identity, seeds):
    order = _stream(identity, seeds).permutation(len(cards))
    return [cards[i] for i in order]


def attempt_draw(episode, k):
    """Seed and home jitter for attempt k of a slot; k=0 is the planned draw."""
    if type(k) is not int or k < 0:
        raise ValueError("attempt index must be a non-negative integer")
    if k == 0:
        return {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    prefix = f"attempt:{episode['episode_index']}:{k}"
    return {"seed": seed(f"{prefix}:expert"), "home_pose_jitter_deg": _jitter(prefix)}


def resample_start(episode, r):
    """The same slot with a freshly drawn start pose and jitter; r=0 is the planned one."""
    if type(r) is not int or r < 0:
        raise ValueError("resample index must be a non-negative integer")
    if r == 0:
        return episode
    prefix = f"resample:{episode['episode_index']}:{r}"
    return {**episode, "layout_id": f"{episode['layout_id']}_r{r}",
            "start": _draw_start(episode["object"], prefix), "home_pose_jitter_deg": _jitter(prefix)}


def _cell_entries(split, count_per_cell, seeds):
    """`count_per_cell` copies of every trained cell, shuffled once so collection order mixes cells."""
    entries = [cell for cell in TRAINED_CELLS for _ in range(count_per_cell)]
    return _shuffled(entries, f"order:{split}", seeds)


def _episode(index, local, split, cell, seeds):
    obj, bin_colour, slot = cell
    identity = f"episode:{split}:{index}"
    expert_seed = seed(f"{identity}:expert")
    seeds[f"{identity}:expert"] = expert_seed
    return dict(episode_index=index, split=split, object=obj, bin=bin_colour, blue_slot=slot,
                cell=list(cell), layout_id=f"{split}_{local:03d}", instruction=instruction(obj, bin_colour),
                start=_draw_start(obj, identity, seeds), home_pose_jitter_deg=_jitter(identity, seeds),
                seed=expert_seed)


def _holdout_layout(tier, local, cell, seeds):
    obj, bin_colour, slot = cell
    identity = f"evaluation:{tier}:{local}"
    expert_seed = seed(f"{identity}:expert")
    seeds[f"{identity}:expert"] = expert_seed
    return dict(tier=tier, object=obj, bin=bin_colour, blue_slot=slot, cell=list(cell),
                instruction=instruction(obj, bin_colour), start=_draw_start(obj, identity, seeds),
                home_pose_jitter_deg=_jitter(identity, seeds), seed=expert_seed)


def _evaluation_holdout(seeds):
    holdout = {}
    for tier in ("screen", "confirm"):
        for local, cell in enumerate(TRAINED_CELLS):
            holdout[f"{tier}_{local:03d}"] = _holdout_layout(tier, local, cell, seeds)
    local = 0
    for cell in TRAINED_CELLS:
        holdout[f"report_{local:03d}"] = _holdout_layout("report", local, cell, seeds)
        local += 1
    for cell in HELD_OUT_CELLS:
        for _ in range(EVAL_REPORT_PER_HELD_OUT_CELL):
            holdout[f"report_{local:03d}"] = _holdout_layout("report", local, cell, seeds)
            local += 1
    return holdout


def make_plan() -> dict:
    seeds = {}
    episodes = []
    for split, count_per_cell, count in (("train", TRAIN_PER_CELL, TRAIN_COUNT),
                                         ("validation", VALIDATION_PER_CELL, VALIDATION_COUNT)):
        entries = _cell_entries(split, count_per_cell, seeds)
        for local, cell in enumerate(entries):
            episodes.append(_episode(len(episodes), local, split, cell, seeds))
        assert len(entries) == count
    holdout = _evaluation_holdout(seeds)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; stop without redraw")
    return dict(
        version=VERSION, master_seed=MASTER, rng="numpy.random.RandomState", rng_seeds=seeds,
        seed_derivation="SHA256(version:master:identity) first four bytes big endian",
        spec=SPEC, plan=PLAN, objects=list(OBJECTS), bin_colours=list(BIN_COLOURS), slots=list(SLOTS),
        held_out_pairs=[list(p) for p in HELD_OUT_PAIRS], trained_pairs=[list(p) for p in TRAINED_PAIRS],
        cells=[list(c) for c in CELLS], trained_cells=[list(c) for c in TRAINED_CELLS],
        held_out_cells=[list(c) for c in HELD_OUT_CELLS],
        start_region=dict(x=list(START_X_M), y=list(START_Y_M), radius_min_m=START_RADIUS_MIN_M,
                          yaw_max_deg_by_object=YAW_MAX_DEG_BY_OBJECT),
        episodes=episodes, train_ids=list(range(TRAIN_COUNT)), validation_ids=list(range(TRAIN_COUNT, ADMIT_COUNT)),
        evaluation_holdout=holdout,
        evaluation_screen_ids=sorted(k for k, v in holdout.items() if v["tier"] == "screen"),
        evaluation_report_ids=sorted(k for k, v in holdout.items() if v["tier"] == "report"),
        evaluation_confirm_ids=sorted(k for k, v in holdout.items() if v["tier"] == "confirm"),
        home_pose_jitter_deg_max=float(HOME_JITTER_DEG_MAX), attempts_per_layout=ATTEMPTS_PER_LAYOUT,
        max_substitutions=MAX_SUBSTITUTIONS, expected_episode_frames=EXPECTED_EPISODE_FRAMES,
        fps=30, state_dim=9, action_dim=8,
    )


def validate_plan(plan):
    if digest(plan) != digest(make_plan()):
        raise ValueError("Noncanonical props P1 plan; no count/seed/layout overrides")
