"""B3 robosuite collection plan: every count, seed, layout and reserved evaluation layout, in advance.

Pure Python + numpy. Imports no robosuite, MuJoCo, torch or LeRobot, so a training contract can
import it in the admission image. Balance is enforced by construction (shuffled decks), never by
sampling. Spec: docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md section 1.
"""
import hashlib
import itertools
import json

import numpy as np

from experiments.paired_demonstrations.plan import instruction

VERSION = "b3_robosuite_data_v1"
MASTER = 20261003
SPEC = "docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md"
PLAN = "docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md"
EXPERT_VERSION = "b3_robosuite_expert_v1"

COLOURS = ("red", "green", "blue", "yellow")
HELD_OUT_PAIRS = (("yellow", "red"), ("green", "blue"))
TRAINED_PAIRS = tuple(sorted(p for p in itertools.permutations(COLOURS, 2) if p not in HELD_OUT_PAIRS))
SIDES = ("A", "B")  # region of the PICK cube; the place cube is in the other region
CELLS = tuple((pick, place, side) for pick, place in TRAINED_PAIRS for side in SIDES)
INSTRUCTIONS = tuple(sorted(instruction(p, q) for p, q in TRAINED_PAIRS))

# Regions as (x_low, x_high, |y|_low, |y|_high); A is y < 0, B is y > 0.
REGION_X = (0.20, 0.35)
REGION_ABS_Y = (0.20, 0.45)
REGION_SIGN = {"A": -1.0, "B": 1.0}
EDGE_BANDS = {"inner": ((0.20, 0.25), (0.20, 0.25)), "outer": ((0.32, 0.35), (0.40, 0.45))}
DISTRACTOR_X = (0.20, 0.35)
DISTRACTOR_Y = (-0.45, 0.45)

# Frozen by the pilot (ladder 0.08 -> 0.09 -> 0.10) and never changed after collection starts.
MIN_CLEARANCE_M = 0.08
NEAR_MAX_M = 0.12
FAR_CLEARANCE_M = 0.15
DISTRACTOR_DRAW_LIMIT = 500
DISTRACTOR_DISTURBANCE_M = 0.02

HOME_JITTER_DEG_MAX = 1.0
HOVER_Z_JITTER_M_MAX = 0.03
PHASES = ("pre_grasp", "pre_place")

TRAIN_PER_CELL = 25
VALIDATION_PER_CELL = 3
TRAIN_COUNT = TRAIN_PER_CELL * len(CELLS)
VALIDATION_COUNT = VALIDATION_PER_CELL * len(CELLS)
ADMIT_COUNT = TRAIN_COUNT + VALIDATION_COUNT
EVAL_SCREEN_COUNT = 20
EVAL_REPORT_COUNT = 40
EVAL_COUNT = EVAL_SCREEN_COUNT + EVAL_REPORT_COUNT
EVAL_EDGE_PER_GROUP = 5  # of 20 layouts in each of: screen, report trained, report held-out
EXPECTED_FRAMES = 528
ATTEMPTS_PER_LAYOUT = 3
MAX_SUBSTITUTIONS = 3

# One training cell: 5 none, 5+5 one-distractor far/near, 5+5 two-distractor far/near.
NEAR_SPLIT = {  # (count, parity of cell index) -> (near_pick, near_place) cards
    (1, 0): (3, 2), (1, 1): (2, 3), (2, 0): (2, 3), (2, 1): (3, 2),
}
EDGE_SPLIT = {0: (3, 2), 1: (2, 3)}  # parity -> (inner, outer) of 5 edge cards
VALIDATION_PROFILES = ((0, "none"), (1, "far"), (2, "near_pick"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seed(identity):
    return int.from_bytes(hashlib.sha256(f"{VERSION}:{MASTER}:{identity}".encode()).digest()[:4], "big")


def _stream(identity, seeds=None):
    value = seed(identity)
    if seeds is not None:
        seeds[identity] = value
    return np.random.RandomState(value)


def _uniform(stream, low, high):
    return float(low + (high - low) * stream.uniform())


def cell_name(cell):
    return "-".join(cell)


def _others(pick, place):
    return sorted(c for c in COLOURS if c not in (pick, place))


# --- decks ---------------------------------------------------------------------------------------

def _shuffled(cards, identity, seeds):
    order = _stream(identity, seeds).permutation(len(cards))
    return [cards[i] for i in order]


def _train_profile_cards(parity):
    cards = [(0, "none")] * 5 + [(1, "far")] * 5 + [(2, "far")] * 5
    for count in (1, 2):
        pick_n, place_n = NEAR_SPLIT[(count, parity)]
        cards += [(count, "near_pick")] * pick_n + [(count, "near_place")] * place_n
    return cards


def _train_edge_cards(parity):
    inner, outer = EDGE_SPLIT[parity]
    return ["interior"] * 20 + ["inner"] * inner + ["outer"] * outer


def _distractor_orders(pick, place, profiles, start):
    """Distractor colours per card; one-distractor colours and near/far order alternate."""
    others, orders = _others(pick, place), []
    one, near = start, start  # separate counters: each alternation must stay balanced on its own
    for count, placement in profiles:
        if count == 0:
            orders.append([])
        elif count == 1:
            orders.append([others[one % 2]])
            one += 1
        elif placement == "far":
            orders.append(list(others))
        else:  # near first
            orders.append([others[near % 2], others[(near + 1) % 2]])
            near += 1
    return orders


def _cell_entries(index, cell, split, seeds):
    pick, place, side = cell
    name = cell_name(cell)
    if split == "train":
        profiles = _shuffled(_train_profile_cards(index % 2), f"profiles:train:{name}", seeds)
        edges = _shuffled(_train_edge_cards(index % 2), f"edges:train:{name}", seeds)
        start = 0
    else:
        profiles = _shuffled(list(VALIDATION_PROFILES), f"profiles:validation:{name}", seeds)
        edges = ["interior"] * VALIDATION_PER_CELL
        start = index
    orders = _distractor_orders(pick, place, profiles, start)
    return [dict(split=split, cell=name, pick=pick, place=place, side=side, edge=edge,
                 profile=[count, placement], distractors=order)
            for (count, placement), edge, order in zip(profiles, edges, orders)]


_SMOKE = (  # riskiest first: FAILURES #29 made the first minutes of a run its smoke test
    lambda e: e["profile"] == [2, "near_pick"] and e["edge"] == "outer",
    lambda e: e["profile"] == [2, "near_place"],
    lambda e: e["profile"] == [1, "near_pick"] and e["edge"] == "inner",
    lambda e: e["profile"] == [0, "none"],
)


def _promote_riskiest(entries):
    entries = list(entries)
    for slot, matches in enumerate(_SMOKE):
        found = next((k for k in range(slot, len(entries)) if matches(entries[k])), None)
        if found is None:
            raise ValueError(f"No training episode matches smoke profile {slot}")
        entries[slot], entries[found] = entries[found], entries[slot]
    return entries


# --- geometry ------------------------------------------------------------------------------------

def _lhs(split, n, seeds):
    """4-D Latin hypercube over (pick x, pick |y|, place x, place |y|)."""
    low = (REGION_X[0], REGION_ABS_Y[0], REGION_X[0], REGION_ABS_Y[0])
    high = (REGION_X[1], REGION_ABS_Y[1], REGION_X[1], REGION_ABS_Y[1])
    names = ("pick_x", "pick_absy", "place_x", "place_absy")
    values = np.empty((n, 4))
    for axis in range(4):
        permutation = _stream(f"layout:{split}:{names[axis]}:permutation", seeds).permutation(n)
        jitter = _stream(f"layout:{split}:{names[axis]}:jitter", seeds).uniform(size=n)
        values[:, axis] = low[axis] + (high[axis] - low[axis]) * (permutation + jitter) / n
    return values


def _xy(side, x, abs_y):
    return [float(x), float(REGION_SIGN[side] * abs_y)]


def _other_side(side):
    return "B" if side == "A" else "A"


def _interior_xy(point, side):
    return _xy(side, point[0], point[1]), _xy(_other_side(side), point[2], point[3])


def _random_xy(stream, side, bands=None):
    (x_low, x_high), (y_low, y_high) = bands or (REGION_X, REGION_ABS_Y)
    x = _uniform(stream, x_low, x_high)
    return _xy(side, x, _uniform(stream, y_low, y_high))


def _edge_xy(prefix, side, edge, seeds):
    pick = _random_xy(_stream(f"{prefix}:edgepick", seeds), side, EDGE_BANDS[edge])
    place = _random_xy(_stream(f"{prefix}:edgeplace", seeds), _other_side(side))
    return pick, place


def _resample_xy(prefix, side, edge):
    if edge == "interior":
        return (_random_xy(_stream(f"{prefix}:pick"), side),
                _random_xy(_stream(f"{prefix}:place"), _other_side(side)))
    return _edge_xy(prefix, side, edge, None)


def _distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _in_band(xy):
    return DISTRACTOR_X[0] <= xy[0] <= DISTRACTOR_X[1] and DISTRACTOR_Y[0] <= xy[1] <= DISTRACTOR_Y[1]


def draw_distractors(prefix, pick_xy, place_xy, placement, order, seeds=None):
    """Rejection-sample the distractors in `order`; raises on exhaustion, never redraws silently.

    near_pick / near_place: the first distractor lies MIN_CLEARANCE_M..NEAR_MAX_M from that target
    and at least MIN_CLEARANCE_M from the other; every other distractor is at least
    FAR_CLEARANCE_M from both targets. All are at least MIN_CLEARANCE_M from one another.
    """
    placed = {}
    for n, colour in enumerate(order):
        stream = _stream(f"{prefix}:{colour}", seeds)
        near = n == 0 and placement in ("near_pick", "near_place")
        anchor, other = (pick_xy, place_xy) if placement == "near_pick" else (place_xy, pick_xy)
        for _ in range(DISTRACTOR_DRAW_LIMIT):
            if near:
                radius = _uniform(stream, MIN_CLEARANCE_M, NEAR_MAX_M)
                angle = _uniform(stream, 0.0, 2 * np.pi)
                xy = [float(anchor[0] + radius * np.cos(angle)), float(anchor[1] + radius * np.sin(angle))]
            else:
                xy = [_uniform(stream, *DISTRACTOR_X), _uniform(stream, *DISTRACTOR_Y)]
            if not _in_band(xy):
                continue
            if near:
                ok = (MIN_CLEARANCE_M <= _distance(xy, anchor) <= NEAR_MAX_M
                      and _distance(xy, other) >= MIN_CLEARANCE_M)
            else:
                ok = _distance(xy, pick_xy) >= FAR_CLEARANCE_M and _distance(xy, place_xy) >= FAR_CLEARANCE_M
            if ok and all(_distance(xy, p) >= MIN_CLEARANCE_M for p in placed.values()):
                placed[colour] = xy
                break
        else:
            raise ValueError(f"Distractor {colour} unplaceable for {prefix}; stop without redraw")
    return placed


def _cube_xy(pick, place, pick_xy, place_xy, distractor_xy):
    return dict(sorted({pick: pick_xy, place: place_xy, **distractor_xy}.items()))


# --- episodes ------------------------------------------------------------------------------------

def _jitter(prefix, seeds):
    home = [float(_uniform(_stream(f"homepose:{prefix}:{j}", seeds), -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX))
            for j in range(7)]
    hover = {phase: float(_uniform(_stream(f"hoverz:{prefix}:{phase}", seeds),
                                   -HOVER_Z_JITTER_M_MAX, HOVER_Z_JITTER_M_MAX)) for phase in PHASES}
    return home, hover


def _episode(index, local, entry, point, seeds):
    split = entry["split"]
    layout_id = f"{split}_{local:03d}"
    if entry["edge"] == "interior":
        pick_xy, place_xy = _interior_xy(point, entry["side"])
    else:
        pick_xy, place_xy = _edge_xy(f"episode:{index}", entry["side"], entry["edge"], seeds)
    distractor_xy = draw_distractors(f"distractor:{index}", pick_xy, place_xy, entry["profile"][1],
                                     entry["distractors"], seeds)
    home, hover = _jitter(f"episode:{index}", seeds)
    seeds[f"expert:{index}"] = seed(f"expert:{index}")
    text = instruction(entry["pick"], entry["place"])
    return dict(
        episode_index=index, split=split, cell=entry["cell"], pick=entry["pick"], place=entry["place"],
        side=entry["side"], colours=sorted({entry["pick"], entry["place"], *entry["distractors"]}),
        distractors=list(entry["distractors"]), profile=list(entry["profile"]), edge=entry["edge"],
        layout_id=layout_id, cube_xy_m=_cube_xy(entry["pick"], entry["place"], pick_xy, place_xy, distractor_xy),
        instruction=text, task_index=INSTRUCTIONS.index(text), seed=seeds[f"expert:{index}"],
        home_pose_jitter_deg=home, hover_z_offset=hover)


def attempt_draw(episode, k):
    """Seed and home jitter for attempt k of a slot; k=0 is the planned draw."""
    if type(k) is not int or k < 0:
        raise ValueError("attempt index must be a non-negative integer")
    if k == 0:
        return {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])}
    prefix = f"attempt:{episode['episode_index']}:{k}"
    home = [_uniform(_stream(f"{prefix}:home:{j}"), -HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX) for j in range(7)]
    return {"seed": seed(f"{prefix}:expert"), "home_pose_jitter_deg": home}


def resample_layout(episode, r):
    """The same slot with freshly drawn positions; r=0 is the planned layout."""
    if type(r) is not int or r < 0:
        raise ValueError("resample index must be a non-negative integer")
    if r == 0:
        return episode
    prefix = f"resample:{episode['episode_index']}:{r}"
    pick_xy, place_xy = _resample_xy(prefix, episode["side"], episode["edge"])
    distractor_xy = draw_distractors(f"{prefix}:distractor", pick_xy, place_xy, episode["profile"][1],
                                     episode["distractors"])
    return {**episode, "layout_id": f"{episode['layout_id']}_r{r}",
            "cube_xy_m": _cube_xy(episode["pick"], episode["place"], pick_xy, place_xy, distractor_xy)}


# --- reserved evaluation layouts -----------------------------------------------------------------

def _evaluation_groups():
    """(tier, pick, place, side) per layout: 20 screen, 20 report trained, 20 report held-out."""
    trained = [(p, q, s) for p, q in TRAINED_PAIRS for s in SIDES]
    held_out = [(p, q, s) for p, q in HELD_OUT_PAIRS for s in SIDES for _ in range(5)]
    return ([("screen", "trained", *c) for c in trained] + [("report", "trained", *c) for c in trained]
            + [("report", "held_out", *c) for c in held_out])


def _evaluation_holdout(seeds):
    groups = _evaluation_groups()
    group_names = [("screen", "trained"), ("report", "trained"), ("report", "held_out")]
    edge = {}
    for tier, kind in group_names:
        members = [n for n, g in enumerate(groups) if (g[0], g[1]) == (tier, kind)]
        chosen = sorted(int(m) for m in _stream(f"evaluation:edge:{tier}:{kind}", seeds).permutation(members)[:EVAL_EDGE_PER_GROUP])
        for j, n in enumerate(chosen):
            edge[n] = ("inner", "outer")[j % 2]
    interior = [n for n in range(EVAL_COUNT) if n not in edge]
    points = _lhs("evaluation", len(interior), seeds)
    point_of = {n: points[j] for j, n in enumerate(interior)}
    holdout = {}
    for n, (tier, kind, pick, place, side) in enumerate(groups):
        layout_id = f"evaluation_{n:03d}"
        if n in edge:
            pick_xy, place_xy = _edge_xy(f"evaluation:{layout_id}", side, edge[n], seeds)
        else:
            pick_xy, place_xy = _interior_xy(point_of[n], side)
        placement = "near_pick" if n % 2 == 0 else "near_place"
        order = _others(pick, place)
        if n % 4 >= 2:
            order.reverse()  # which colour is the near one alternates too
        distractor_xy = draw_distractors(f"distractor:evaluation:{layout_id}", pick_xy, place_xy,
                                         placement, order, seeds)
        home, hover = _jitter(f"evaluation:{layout_id}", seeds)
        conditions = {"distractor": distractor_xy}
        if tier == "report":
            conditions = {"control": {}, **conditions}
        holdout[layout_id] = dict(
            tier=tier, kind=kind, pick=pick, place=place, side=side, edge=edge.get(n, "interior"),
            cube_xy_m=dict(sorted({pick: pick_xy, place: place_xy}.items())),
            distractors=order, near=placement, conditions=conditions,
            instruction=instruction(pick, place),
            home_pose_jitter_deg=home, hover_z_offset=hover)
    return holdout


# --- the plan ------------------------------------------------------------------------------------

def make_plan() -> dict:
    seeds = {}
    train_entries = []
    for index, cell in enumerate(CELLS):
        train_entries += _cell_entries(index, cell, "train", seeds)
    train_entries = _promote_riskiest(_shuffled(train_entries, "order:train", seeds))
    validation_entries = []
    for index, cell in enumerate(CELLS):
        validation_entries += _cell_entries(index, cell, "validation", seeds)
    validation_entries = _shuffled(validation_entries, "order:validation", seeds)

    episodes = []
    for split, entries, count in (("train", train_entries, TRAIN_COUNT), ("validation", validation_entries, VALIDATION_COUNT)):
        interior = sum(e["edge"] == "interior" for e in entries)
        points = _lhs(split, interior, seeds)
        used = 0
        for local, entry in enumerate(entries):
            point = None
            if entry["edge"] == "interior":
                point, used = points[used], used + 1
            episodes.append(_episode(len(episodes), local, entry, point, seeds))
        assert len(entries) == count

    holdout = _evaluation_holdout(seeds)
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; stop without redraw")
    layouts = [dict(layout_id=e["layout_id"], split=e["split"], cube_xy_m=e["cube_xy_m"]) for e in episodes]
    return dict(
        version=VERSION, master_seed=MASTER, rng="numpy.random.RandomState", rng_seeds=seeds,
        seed_derivation="SHA256(version:master:identity) first four bytes big endian",
        spec=SPEC, plan=PLAN, expert_version=EXPERT_VERSION,
        colours=list(COLOURS), held_out_pairs=[list(p) for p in HELD_OUT_PAIRS],
        trained_pairs=[list(p) for p in TRAINED_PAIRS], cells=[list(c) for c in CELLS],
        instructions=list(INSTRUCTIONS),
        regions=dict(x=list(REGION_X), abs_y=list(REGION_ABS_Y),
                     edge={k: [list(v[0]), list(v[1])] for k, v in EDGE_BANDS.items()},
                     distractor=dict(x=list(DISTRACTOR_X), y=list(DISTRACTOR_Y))),
        layouts=layouts, episodes=episodes,
        train_ids=list(range(TRAIN_COUNT)), validation_ids=list(range(TRAIN_COUNT, ADMIT_COUNT)),
        evaluation_holdout=holdout,
        evaluation_screen_ids=[k for k, v in holdout.items() if v["tier"] == "screen"],
        evaluation_report_ids=[k for k, v in holdout.items() if v["tier"] == "report"],
        min_clearance_m=float(MIN_CLEARANCE_M), near_max_m=float(NEAR_MAX_M),
        far_clearance_m=float(FAR_CLEARANCE_M), distractor_disturbance_m=float(DISTRACTOR_DISTURBANCE_M),
        home_pose_jitter_deg_max=float(HOME_JITTER_DEG_MAX), hover_z_jitter_m_max=float(HOVER_Z_JITTER_M_MAX),
        attempts_per_layout=ATTEMPTS_PER_LAYOUT, max_substitutions=MAX_SUBSTITUTIONS,
        expected_episode_frames=EXPECTED_FRAMES, fps=30, state_dim=9, action_dim=8,
    )


def validate_plan(plan):
    if digest(plan) != digest(make_plan()):
        raise ValueError("Noncanonical B3 plan; no count/seed/layout overrides")
