"""G1 collection plan: 3,000 randomized slots (2,700 train, 300 validation) and 96 expert-gate attempts. Pure; admission image.

Every slot is one P2 instruction cell and one randomized scene, drawn from the TRAIN pools only: lights, table/floor/wall textures,
table height, bin shift, target start region, 0-5 distractors (props of the six and pool A primitives) and a +-10 degree home pose.
Everything a scene changes is a `gen_scene.Condition(pool="train")` (revision R2 of the G1 spec); the layout carries the target, the
used start pose, the prop distractors and the home jitter, the condition carries the rest, so `gen_scene.compose.model_for(layout,
condition)` rebuilds the scene. Balance is by construction (shuffled decks), never by sampling, for the cells, the distractor counts,
the five table-height strata and the start region; the continuous values inside a stratum are drawn per slot from a named stream.

Spec: docs/superpowers/specs/2026-11-06-g1-randomized-data-design.md (D1-D10, revision R1-R4 and the 2026-11-10 mapping note).
`gen_scene` and `gen_eval` are frozen by G0's core hash and are imported read-only; nothing here is a fork of them.
"""
import hashlib
import json
import math

import numpy as np

from experiments.gen_scene import levels, pools
from experiments.gen_scene.conditions import Condition
from experiments.props_p2_data import plan as p2
from experiments.props_scene import constants as psc

VERSION = "gen_g1_v1"
NAMESPACE = "gen:g1"
MASTER = 20261110
SPEC = "docs/superpowers/specs/2026-11-06-g1-randomized-data-design.md"
PLAN_DOC = "docs/superpowers/plans/2026-11-06-g1-randomized-data.md"

OBJECTS = p2.OBJECTS
SLOTS = p2.SLOTS
TRAINED_CELLS = p2.TRAINED_CELLS
HELD_OUT_PAIRS = p2.HELD_OUT_PAIRS
TRAIN_PER_CELL = 135
VALIDATION_PER_CELL = 15
TRAIN_COUNT = TRAIN_PER_CELL * len(TRAINED_CELLS)
VALIDATION_COUNT = VALIDATION_PER_CELL * len(TRAINED_CELLS)
ADMIT_COUNT = TRAIN_COUNT + VALIDATION_COUNT
GATE_ATTEMPTS = 96
GATE_PER_OBJECT = GATE_ATTEMPTS // len(OBJECTS)

# R1: distractors per episode (props of the six and pool A primitives), as shares of the 3,000 slots.
COUNT_WEIGHTS = {0: 0.05, 1: 0.15, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.10}
PROP_SHARE = 0.6               # each distractor is a prop with this probability while props of the six remain, else a pool A primitive
# R2: ranges (Condition fields).
DIFFUSE_SCALE, SPECULAR_SCALE, AMBIENT_SCALE, HEADLIGHT_SCALE = (0.25, 2.8), (0.5, 2.8), (0.2, 1.5), (0.3, 1.5)
TINT = (0.55, 1.0)
LIGHT_DIR_SHARE, ELEVATION_DEG, SHADOW_SHARE = 0.5, (15.0, 80.0), 0.5
TEXTURE_SHARE = 0.85           # each of table, floor, wall: a train-pool texture with this probability, else P2's default
TABLE_OFFSET_M = (-0.07, 0.12)
QUINTILES = 5
BIN_SHIFT_M = (-0.05, 0.05)
OUTER_SHARE = 0.3              # target start on G0's outer ring (x 0.42-0.50) instead of P2's region
HOME_JITTER_DEG_MAX = 10.0
DISTRACTOR_REGION_X_M, DISTRACTOR_REGION_Y_M = levels.DISTRACTOR_REGION_X_M, levels.DISTRACTOR_REGION_Y_M
REGION_X_M = {"p2": psc.START_X_M, "outer": levels.OUTER_START_X_M}
EXPECTED_EPISODE_FRAMES = p2.EXPECTED_EPISODE_FRAMES
ATTEMPTS_PER_LAYOUT = p2.ATTEMPTS_PER_LAYOUT
MAX_SUBSTITUTIONS = p2.MAX_SUBSTITUTIONS
PLACEMENT_RESTARTS = 2000

PLAN_SHA256 = "3b57cfac88ee5f828a4dc35103addf0d9a90e3a75591e9da6460cf31f9061bc3"  # pinned 2026-11-10; test_plan


def digest(value):
    return p2.digest(value)


def seed(identity):
    return int.from_bytes(hashlib.sha256(f"{VERSION}:{MASTER}:{identity}".encode()).digest()[:4], "big")


def _stream(identity, seeds=None):
    value = seed(identity)
    if seeds is not None:
        seeds[identity] = value
    return np.random.RandomState(value)


def _u(rng, low, high):
    return float(low + (high - low) * rng.uniform())


def instruction(obj, bin_colour):
    return p2.instruction(obj, bin_colour)


def count_deck(total):
    """{distractor count: slots} for `total` slots; raises unless every share is a whole number of slots."""
    deck = {n: total * w for n, w in COUNT_WEIGHTS.items()}
    if any(abs(v - round(v)) > 1e-9 for v in deck.values()) or round(sum(deck.values())) != total:
        raise ValueError(f"{total} slots do not split into whole distractor counts")
    return {n: round(v) for n, v in deck.items()}


def _shuffled(cards, identity, seeds):
    order = _stream(identity, seeds).permutation(len(cards))
    return [cards[i] for i in order]


def _jitter(identity, seeds=None):
    rng = _stream(f"{identity}:home", seeds)
    return [_u(rng, -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX) for _ in range(7)]


def quintile_of(table_offset_m):
    low, high = TABLE_OFFSET_M
    return min(int((table_offset_m - low) / (high - low) * QUINTILES), QUINTILES - 1)


# --- one randomized scene ----------------------------------------------------------------------------------------------

def _light(rng):
    diffuse, specular = _u(rng, *DIFFUSE_SCALE), _u(rng, *SPECULAR_SCALE)
    ambient, headlight = _u(rng, *AMBIENT_SCALE), _u(rng, *HEADLIGHT_SCALE)
    tint = tuple(_u(rng, *TINT) for _ in range(3))
    elevation, azimuth = math.radians(_u(rng, *ELEVATION_DEG)), math.radians(_u(rng, 0.0, 360.0))
    redirect, shadows = rng.uniform() < LIGHT_DIR_SHARE, bool(rng.uniform() < SHADOW_SHARE)
    direction = (math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), -math.sin(elevation))
    return dict(light_diffuse_scale=diffuse, light_specular_scale=specular, ambient_scale=ambient, headlight_scale=headlight,
                tint=tint, light_dir=direction if redirect else None, light_shadows=shadows)


def _surfaces(rng, manifest):
    both = pools.pools(manifest)
    out = {}
    for field, surface in (("table_texture", "table"), ("floor_texture", "room"), ("wall_texture", "room")):
        pool = both[surface]["train"]
        use, index = rng.uniform() < TEXTURE_SHARE, int(rng.randint(len(pool)))
        out[field] = pool[index] if use else None
    return out


def _kinds(rng, target, count):
    """(props, primitives): `count` distractors, each a prop of the six while any remain (with PROP_SHARE), else a pool A primitive."""
    props = [o for o in OBJECTS if o != target]
    props = [props[i] for i in rng.permutation(len(props))]
    combos = [(s, c) for s in pools.PRIMITIVE_SHAPES["A"] for c in pools.PRIMITIVE_COLOURS["A"]]
    combos = [combos[i] for i in rng.permutation(len(combos))]
    chosen_props, chosen_primitives = [], []
    for _ in range(count):
        if props and rng.uniform() < PROP_SHARE:
            chosen_props.append(props.pop())
        else:
            chosen_primitives.append(combos.pop())
    return chosen_props, chosen_primitives


def _place(rng, target, region, shift, props, primitives):
    """(start, distractor poses, restarts): target start in `region`, then each distractor in the wide region, all spaced (props
    both ways, primitives as neighbours) and clear of both bins at `shift`. The whole set restarts when one object is stuck."""
    names = [*props, *(f"{s}_{c}" for s, c in primitives)]
    for restart in range(1, PLACEMENT_RESTARTS + 1):
        start = levels._pose(rng, REGION_X_M[region], psc.START_Y_M, target)
        if not levels.spaced([(target, start)], shift):
            continue
        placed = []
        for name in names:
            for _ in range(levels.TRIES_PER_OBJECT):
                pose = levels._pose(rng, DISTRACTOR_REGION_X_M, DISTRACTOR_REGION_Y_M, name)
                if levels.spaced([(target, start), *placed, (name, pose)], shift):
                    placed.append((name, pose))
                    break
            else:
                break
        else:
            return start, [pose for _, pose in placed], restart
    raise RuntimeError(f"no spaced placement for {target} + {names} at region {region}, shift {shift} in {PLACEMENT_RESTARTS} restarts")


def draw_scene(identity, stratum, manifest=None):
    """The scene body of one slot: everything the layout and the condition carry. `stratum` fixes the cell, the distractor count, the
    table-height quintile and the start region; everything else is drawn from the stream `identity`."""
    obj, bin_colour, blue_slot = stratum["cell"]
    rng = _stream(f"{identity}:scene")
    light = _light(rng)
    surfaces = _surfaces(rng, manifest)
    low, high = TABLE_OFFSET_M
    width = (high - low) / QUINTILES
    offset = float(low + (stratum["quintile"] + rng.uniform()) * width)
    shift = _u(rng, *BIN_SHIFT_M)
    props, primitives = _kinds(rng, obj, stratum["distractor_count"])
    start, poses, restarts = _place(rng, obj, stratum["region"], shift, props, primitives)
    prop_poses, primitive_poses = poses[:len(props)], poses[len(props):]
    condition = Condition(axis="randomized", level="train", pool="train", **light, **surfaces, table_offset_m=offset,
                          bin_shift_m=shift, primitives=tuple((s, c, *pose) for (s, c), pose in zip(primitives, primitive_poses))).check(manifest)
    return dict(object=obj, bin=bin_colour, blue_slot=blue_slot, cell=list(stratum["cell"]),
                distractor_count=stratum["distractor_count"], prop_distractors=len(props), primitive_count=len(primitives),
                quintile=stratum["quintile"], region=stratum["region"], instruction=instruction(obj, bin_colour), start=list(start),
                distractors=[{"object": n, "pose": list(q)} for n, q in zip(props, prop_poses)], layout_draws=restarts,
                condition=condition.to_dict(), condition_sha256=condition.digest())


def condition_of(episode):
    return Condition.from_dict(episode["condition"])


def _episode(identity, stratum, seeds, manifest):
    expert_seed = seed(f"{identity}:expert")
    seeds[f"{identity}:expert"] = expert_seed
    return dict(**draw_scene(identity, stratum, manifest), home_pose_jitter_deg=_jitter(identity, seeds), seed=expert_seed)


def _strata(split, per_cell, seeds):
    """The cell, distractor count, quintile and start region of every slot of a split, each from its own shuffled deck."""
    total = per_cell * len(TRAINED_CELLS)
    cells = _shuffled([cell for cell in TRAINED_CELLS for _ in range(per_cell)], f"{split}:cells", seeds)
    counts = _shuffled([n for n, k in count_deck(total).items() for _ in range(k)], f"{split}:counts", seeds)
    quintiles = _shuffled([q for q in range(QUINTILES) for _ in range(total // QUINTILES)], f"{split}:quintiles", seeds)
    outer = round(total * OUTER_SHARE)
    regions = _shuffled(["outer"] * outer + ["p2"] * (total - outer), f"{split}:regions", seeds)
    return [dict(cell=cell, distractor_count=n, quintile=q, region=r) for cell, n, q, r in zip(cells, counts, quintiles, regions)]


def _slots(seeds, manifest):
    episodes = []
    for split, per_cell in (("train", TRAIN_PER_CELL), ("validation", VALIDATION_PER_CELL)):
        for local, stratum in enumerate(_strata(split, per_cell, seeds)):
            identity = f"{NAMESPACE}:{split}:{local}"
            episodes.append(dict(episode_index=len(episodes), split=split, layout_id=f"{split}_{local:04d}", identity=identity,
                                 **_episode(identity, stratum, seeds, manifest)))
    return episodes


def _gate_strata(seeds):
    """96 attempts, balanced on what the gate reports: 16 per target object, 16 per distractor count, 19-20 per quintile."""
    cells = []
    for obj in OBJECTS:
        own = [c for c in TRAINED_CELLS if c[0] == obj]
        cells += [own[i % len(own)] for i in range(GATE_PER_OBJECT)]
    cells = _shuffled(cells, "gate:cells", seeds)
    counts = _shuffled([n for n in COUNT_WEIGHTS for _ in range(GATE_ATTEMPTS // len(COUNT_WEIGHTS))], "gate:counts", seeds)
    quintiles = _shuffled([q for q in range(QUINTILES) for _ in range(GATE_ATTEMPTS // QUINTILES)]
                          + [0] * (GATE_ATTEMPTS % QUINTILES), "gate:quintiles", seeds)
    outer = round(GATE_ATTEMPTS * OUTER_SHARE)
    regions = _shuffled(["outer"] * outer + ["p2"] * (GATE_ATTEMPTS - outer), "gate:regions", seeds)
    return [dict(cell=cell, distractor_count=n, quintile=q, region=r) for cell, n, q, r in zip(cells, counts, quintiles, regions)]


def _gate(seeds, manifest):
    attempts = []
    for index, stratum in enumerate(_gate_strata(seeds)):
        identity = f"{NAMESPACE}:gate:{index}"
        attempts.append(dict(episode_index=index, split="gate", layout_id=f"gate_{index:03d}", identity=identity,
                             **_episode(identity, stratum, seeds, manifest)))
    return attempts


def attempt_draw(episode, k):
    """Seed and home jitter for attempt k of a slot; k=0 is the planned draw. Jitter is +-HOME_JITTER_DEG_MAX (P2's was +-1)."""
    if type(k) is not int or k < 0:
        raise ValueError("attempt index must be a non-negative integer")
    if k == 0:
        return {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    prefix = f"attempt:{episode['split']}:{episode['episode_index']}:{k}"
    return {"seed": seed(f"{prefix}:expert"), "home_pose_jitter_deg": _jitter(prefix)}


def resample_start(episode, r):
    """The same slot (cell, count, quintile, region) with a freshly drawn scene; r=0 is the planned one. Only the draw changes,
    so every stratum total of the admitted data is the plan's whatever is substituted."""
    if type(r) is not int or r < 0:
        raise ValueError("resample index must be a non-negative integer")
    if r == 0:
        return episode
    prefix = f"resample:{episode['split']}:{episode['episode_index']}:{r}"
    stratum = dict(cell=tuple(episode["cell"]), distractor_count=episode["distractor_count"], quintile=episode["quintile"],
                   region=episode["region"])
    return {**episode, "layout_id": f"{episode['layout_id']}_r{r}", **draw_scene(prefix, stratum), "home_pose_jitter_deg": _jitter(prefix)}


def shard_slots(k, shards, total=ADMIT_COUNT):
    """Contiguous slot block of shard k; the blocks partition 0..total-1 in order (the first `total % shards` are one slot longer)."""
    if not 0 <= k < shards <= total:
        raise ValueError(f"shard {k} of {shards} does not partition {total} slots")
    size, extra = divmod(total, shards)
    first = k * size + min(k, extra)
    return list(range(first, first + size + (1 if k < extra else 0)))


def make_plan(manifest=None) -> dict:
    seeds = {}
    episodes = _slots(seeds, manifest)
    gate = _gate(seeds, manifest)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; stop without redraw")
    return dict(
        version=VERSION, namespace=NAMESPACE, master_seed=MASTER, rng="numpy.random.RandomState", rng_seeds=seeds,
        seed_derivation="SHA256(version:master:identity) first four bytes big endian",
        spec=SPEC, plan=PLAN_DOC, objects=list(OBJECTS), slots=list(SLOTS), trained_cells=[list(c) for c in TRAINED_CELLS],
        held_out_pairs=[list(x) for x in HELD_OUT_PAIRS],
        ranges=dict(count_weights={str(n): w for n, w in COUNT_WEIGHTS.items()}, prop_share=PROP_SHARE, diffuse=DIFFUSE_SCALE,
                    specular=SPECULAR_SCALE, ambient=AMBIENT_SCALE, headlight=HEADLIGHT_SCALE, tint=TINT,
                    light_dir_share=LIGHT_DIR_SHARE, elevation_deg=ELEVATION_DEG, shadow_share=SHADOW_SHARE,
                    texture_share=TEXTURE_SHARE, table_offset_m=TABLE_OFFSET_M, quintiles=QUINTILES, bin_shift_m=BIN_SHIFT_M,
                    outer_share=OUTER_SHARE, home_jitter_deg_max=HOME_JITTER_DEG_MAX, region_x_m=REGION_X_M,
                    start_y_m=psc.START_Y_M, distractor_region_x_m=DISTRACTOR_REGION_X_M,
                    distractor_region_y_m=DISTRACTOR_REGION_Y_M, texture_manifest_sha256=pools.manifest_digest(manifest)),
        episodes=episodes, train_ids=list(range(TRAIN_COUNT)), validation_ids=list(range(TRAIN_COUNT, ADMIT_COUNT)),
        gate_attempts=gate, attempts_per_layout=ATTEMPTS_PER_LAYOUT, max_substitutions=MAX_SUBSTITUTIONS,
        expected_episode_frames=EXPECTED_EPISODE_FRAMES, fps=30, state_dim=9, action_dim=8,
    )


def validate_plan(plan):
    if digest(plan) != digest(make_plan()) or digest(plan) != PLAN_SHA256:
        raise ValueError("Noncanonical G1 plan; no count/seed/layout overrides")


def earlier_values():
    """Every seed and pose of the sealed P2 plan, the P2 re-test, M0, the pilot plan and G0's whole plan (all tiers). Needs the
    sealed P2 data mounted (M0's `forbidden_values`)."""
    from experiments.gen_eval import plan as g0
    forbidden = g0.forbidden_values()
    seeds, poses = set(forbidden["seeds"]), set(forbidden["poses"])
    earlier = g0.make_plan()
    seeds |= set(earlier["rng_seeds"].values())
    for layout in earlier["layouts"].values():
        seeds.add(layout["seed"])
        poses.add(tuple(layout["start"]))
        poses |= {tuple(d["pose"]) for d in layout["distractors"]}
        for axis in layout["axes"].values():
            condition = axis["condition"]
            if condition["start"] is not None:
                poses.add(tuple(condition["start"]))
            poses |= {tuple(x[1:]) for x in condition["extra_props"]} | {tuple(x[2:]) for x in condition["primitives"]}
    return {"seeds": seeds, "poses": poses}


def check_disjoint_from_earlier(plan, forbidden=None):
    """Raises ValueError if any slot or gate attempt shares a seed or pose with an earlier plan, or repeats one inside this plan."""
    from experiments.gen_eval import plan as g0
    layouts = {e["layout_id"]: e for e in (*plan["episodes"], *plan["gate_attempts"])}
    if len(layouts) != len(plan["episodes"]) + len(plan["gate_attempts"]):
        raise ValueError("Layout id collision")
    g0.check_unique(layouts)
    g0.check_disjoint(layouts, forbidden if forbidden is not None else earlier_values())
    if any(not e["identity"].startswith(f"{NAMESPACE}:") for e in layouts.values()):
        raise ValueError("An identity outside the gen:g1 namespace")
