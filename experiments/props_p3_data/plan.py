"""P3 collection plan: one object per episode, a fixed prompt, food/drinks to blue/orange, 288 reserved layouts.

Pure Python + numpy; imports only `experiments.props_scene.constants` (also pure), so a training
contract can import it in the admission image. Balance is by construction (shuffled decks), never by
sampling. Spec: docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md.
"""
import hashlib
import json
import math

import numpy as np

from experiments.props_scene import constants as psc

VERSION = "props_p3_v1"
MASTER = 20261020
SPEC = "docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md"
PLAN = "docs/superpowers/plans/2026-10-19-props-p3-collection.md"

PROMPT = "sort the item"
OBJECTS = psc.OBJECTS
BIN_COLOURS = psc.BINS
SLOTS = ("left", "right")
CATEGORY = {"can": "drinks", "milk": "drinks", "bottle": "drinks", "bread": "food", "lemon": "food", "cereal": "food"}
CONTAINER = {"drinks": "orange", "food": "blue"}
TRAINED_OBJECTS = ("can", "milk", "bread", "lemon")
HELD_OUT_OBJECTS = ("bottle", "cereal")


def container_of(obj):
    return CONTAINER[CATEGORY[obj]]


CELLS = tuple((obj, container_of(obj), slot) for obj in OBJECTS for slot in SLOTS)
TRAINED_CELLS = tuple(c for c in CELLS if c[0] in TRAINED_OBJECTS)
HELD_OUT_CELLS = tuple(c for c in CELLS if c[0] in HELD_OUT_OBJECTS)

TRAIN_PER_CELL = 60
VALIDATION_PER_CELL = 8
TRAIN_COUNT = TRAIN_PER_CELL * len(TRAINED_CELLS)
VALIDATION_COUNT = VALIDATION_PER_CELL * len(TRAINED_CELLS)
ADMIT_COUNT = TRAIN_COUNT + VALIDATION_COUNT
SCREEN_PER_CELL = 3
CONFIRM_PER_CELL = 3
REPORT_PER_TRAINED_CELL = 10
REPORT_PER_HELD_OUT_CELL = 40
GATE_PER_CELL = 24

START_X_M = psc.START_X_M
START_Y_M = psc.START_Y_M
START_RADIUS_MIN_M = psc.START_RADIUS_MIN_M
YAW_MAX_DEG_BY_OBJECT = dict(psc.YAW_MAX_DEG_BY_OBJECT)
HOME_JITTER_DEG_MAX = psc.HOME_JITTER_DEG_MAX

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


def _draw_pose(identity, obj, seeds=None):
    stream = _stream(f"{identity}:pose", seeds)
    while True:
        x, y = _uniform(stream, *START_X_M), _uniform(stream, *START_Y_M)
        yaw = _uniform(stream, -YAW_MAX_DEG_BY_OBJECT[obj], YAW_MAX_DEG_BY_OBJECT[obj])
        if math.hypot(x, y) >= START_RADIUS_MIN_M:
            return [x, y, yaw]


def _jitter(identity, seeds=None):
    stream = _stream(f"{identity}:home", seeds)
    return [_uniform(stream, -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX) for _ in range(7)]


def _shuffled(cards, identity, seeds):
    order = _stream(identity, seeds).permutation(len(cards))
    return [cards[i] for i in order]


def _layout_entry(identity, cell, seeds):
    obj, bin_colour, slot = cell
    expert_seed = seed(f"{identity}:expert")
    seeds[f"{identity}:expert"] = expert_seed
    return dict(object=obj, category=CATEGORY[obj], bin=bin_colour, blue_slot=slot, cell=list(cell), instruction=PROMPT,
                start=_draw_pose(identity, obj, seeds), home_pose_jitter_deg=_jitter(identity, seeds), seed=expert_seed)


def attempt_draw(episode, k):
    """Seed and home jitter for attempt k of a slot; k=0 is the planned draw."""
    if type(k) is not int or k < 0:
        raise ValueError("attempt index must be a non-negative integer")
    if k == 0:
        return {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    prefix = f"attempt:{episode['episode_index']}:{k}"
    return {"seed": seed(f"{prefix}:expert"), "home_pose_jitter_deg": _jitter(prefix)}


def resample_start(episode, r):
    """The same slot and object with a freshly drawn start; r=0 is the planned one."""
    if type(r) is not int or r < 0:
        raise ValueError("resample index must be a non-negative integer")
    if r == 0:
        return episode
    prefix = f"resample:{episode['episode_index']}:{r}"
    return {**episode, "layout_id": f"{episode['layout_id']}_r{r}", "start": _draw_pose(prefix, episode["object"]),
            "home_pose_jitter_deg": _jitter(prefix)}


def _training_episodes(seeds):
    entries = [cell for cell in TRAINED_CELLS for _ in range(TRAIN_PER_CELL)]
    episodes = []
    for local, cell in enumerate(_shuffled(entries, "order:train", seeds)):
        episodes.append(dict(episode_index=local, split="train", layout_id=f"train_{local:03d}",
                             **_layout_entry(f"episode:train:{local}", cell, seeds)))
    return episodes


def _validation_episodes(start, seeds):
    entries = [cell for cell in TRAINED_CELLS for _ in range(VALIDATION_PER_CELL)]
    episodes = []
    for local, cell in enumerate(_shuffled(entries, "order:validation", seeds)):
        episodes.append(dict(episode_index=start + local, split="validation", layout_id=f"validation_{local:03d}",
                             **_layout_entry(f"episode:validation:{local}", cell, seeds)))
    return episodes


def _reserved(seeds):
    deck = {
        "screen": [c for c in TRAINED_CELLS for _ in range(SCREEN_PER_CELL)],
        "report": ([c for c in TRAINED_CELLS for _ in range(REPORT_PER_TRAINED_CELL)]
                   + [c for c in HELD_OUT_CELLS for _ in range(REPORT_PER_HELD_OUT_CELL)]),
        "confirm": [c for c in TRAINED_CELLS for _ in range(CONFIRM_PER_CELL)],
    }
    holdout = {}
    for tier, cells in deck.items():
        for index, cell in enumerate(_shuffled(cells, f"order:{tier}", seeds)):
            key = f"{tier}_{index:03d}"
            entry = _layout_entry(f"evaluation:{tier}:{index}", cell, seeds)
            holdout[key] = dict(tier=tier, layout_id=key, held_out=cell[0] in HELD_OUT_OBJECTS, **entry)
    return holdout


def _gate_attempts(seeds):
    attempts = []
    for cell in CELLS:
        for _ in range(GATE_PER_CELL):
            index = len(attempts)
            attempts.append(dict(episode_index=index, split="gate", layout_id=f"gate_{index:03d}",
                                 **_layout_entry(f"gate:{index}", cell, seeds)))
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
        spec=SPEC, plan=PLAN, prompt=PROMPT, objects=list(OBJECTS), bin_colours=list(BIN_COLOURS), slots=list(SLOTS),
        category=dict(CATEGORY), container=dict(CONTAINER), trained_objects=list(TRAINED_OBJECTS),
        held_out_objects=list(HELD_OUT_OBJECTS), trained_cells=[list(c) for c in TRAINED_CELLS],
        held_out_cells=[list(c) for c in HELD_OUT_CELLS],
        start_region=dict(x=list(START_X_M), y=list(START_Y_M), radius_min_m=START_RADIUS_MIN_M,
                          yaw_max_deg_by_object=YAW_MAX_DEG_BY_OBJECT),
        episodes=episodes, train_ids=list(range(TRAIN_COUNT)), validation_ids=list(range(TRAIN_COUNT, ADMIT_COUNT)),
        evaluation_holdout=holdout,
        **{f"evaluation_{tier}_ids": sorted(k for k, v in holdout.items() if v["tier"] == tier)
           for tier in ("screen", "report", "confirm")},
        gate_attempts=gate, home_pose_jitter_deg_max=float(HOME_JITTER_DEG_MAX),
        attempts_per_layout=ATTEMPTS_PER_LAYOUT, max_substitutions=MAX_SUBSTITUTIONS,
        expected_episode_frames=EXPECTED_EPISODE_FRAMES, fps=30, state_dim=9, action_dim=8,
    )


def validate_plan(plan):
    if digest(plan) != digest(make_plan()):
        raise ValueError("Noncanonical props P3 plan; no count/seed/layout overrides")
