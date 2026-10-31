"""M0 layouts: 80 selection (S) and 576 comparison (C) fresh P2-style layouts. Pure; admission image.

Samplers are P2's own (`props_p2_data.plan`, imported read-only): start region, yaw caps, symmetric spacing, hand
clearance, shuffled distractor decks (here one full shuffled pass over the options before any repeat). Identities are `vla_inputs:m0:<tier>:<cell>:<i>`, a namespace no earlier plan used,
and `check_disjoint` refuses any seed or pose that coincides with a P2 or P2 re-test layout. The rollout seed of a layout
(`seed`) is shared by every arm. Design: docs/superpowers/specs/2026-10-30-vla-inputs-m0-evaluation-design.md.
"""
import json
from pathlib import Path

from experiments.props_p2_data import plan as p2
from experiments.xvla_props_p2_retest import plan as retest

VERSION = "vla_inputs_m0_v1"
NAMESPACE = "vla_inputs:m0"
DISTRACTOR_COUNTS = (1, 2)
SELECTION_PER_CELL = 2
COMPARISON_PER_CELL = 12
SELECTION_CELLS = tuple((o, b, s, n) for o, b, s in p2.TRAINED_CELLS for n in DISTRACTOR_COUNTS)
COMPARISON_CELLS = tuple((o, b, s, n) for o, b, s in p2.CELLS for n in DISTRACTOR_COUNTS)
P2_ROOT = Path(__file__).resolve().parents[2] / "outputs/props_p2_data/props_p2_20261013"


def cell_name(cell):
    obj, bin_colour, slot, count = cell
    return f"{obj}-{bin_colour}-{slot}-d{count}"


def layout_id(identity):
    return identity.replace(":", "_")


def forbidden_values():
    """Every seed and every pose (start or distractor) in the sealed P2 plan and the P2 re-test plan."""
    sealed = json.loads((P2_ROOT / "plan.json").read_text())
    rows = [*sealed["episodes"], *sealed["evaluation_holdout"].values(), *sealed["gate_attempts"],
            *retest.make_plan()["evaluation_holdout"].values()]
    seeds = set(sealed["rng_seeds"].values()) | set(retest.make_plan()["seeds"].values()) | {r["seed"] for r in rows}
    poses = set()
    for row in rows:
        poses.add(tuple(row["start"]))
        poses |= {tuple(d["pose"]) for d in row["distractors"]}
    return {"seeds": seeds, "poses": poses}


def check_disjoint(layouts, forbidden):
    """Raises ValueError if any layout seed, start pose or distractor pose is in `forbidden`."""
    for lid, layout in layouts.items():
        if layout["seed"] in forbidden["seeds"]:
            raise ValueError(f"{lid}: seed {layout['seed']} is a P2 or re-test seed")
        if tuple(layout["start"]) in forbidden["poses"]:
            raise ValueError(f"{lid}: start pose coincides with a P2 or re-test pose")
        if any(tuple(d["pose"]) in forbidden["poses"] for d in layout["distractors"]):
            raise ValueError(f"{lid}: a distractor pose coincides with a P2 or re-test pose")


def _deck(target, count, n, identity, seeds):
    """n distractor sets: every option once (shuffled) before any repeats, pass after pass."""
    options, cards, k = p2._options(target, count), [], 0
    while len(cards) < n:
        cards += p2._shuffled(options, f"{identity}:pass{k}", seeds)
        k += 1
    return cards[:n]


def _tier(tier, cells, per_cell, seeds):
    layouts = []
    for cell in cells:
        obj, bin_colour, slot, count = cell
        deck = _deck(obj, count, per_cell, f"{NAMESPACE}:{tier}:{cell_name(cell)}:deck", seeds)
        for i in range(per_cell):
            identity = f"{NAMESPACE}:{tier}:{cell_name(cell)}:{i}"
            seed = p2.seed(f"{identity}:expert")
            seeds[f"{identity}:expert"] = seed
            geometry = p2._draw_layout(obj, deck[i], identity, seeds)
            layouts.append(dict(
                layout_id=layout_id(identity), identity=identity, tier=tier, cell=list(cell), object=obj, bin=bin_colour,
                blue_slot=slot, distractor_count=count, instruction=p2.instruction(obj, bin_colour),
                held_out=(obj, bin_colour) in p2.HELD_OUT_PAIRS, home_pose_jitter_deg=p2._jitter(identity, seeds),
                seed=seed, **geometry))
    return layouts


def make_plan(forbidden=None):
    """The whole plan, byte-deterministic. Raises if any draw is forbidden (never redraws)."""
    seeds = {}
    selection = _tier("selection", SELECTION_CELLS, SELECTION_PER_CELL, seeds)
    comparison = _tier("comparison", COMPARISON_CELLS, COMPARISON_PER_CELL, seeds)
    layouts = {l["layout_id"]: l for l in selection + comparison}
    if len(layouts) != len(selection) + len(comparison) or len(set(seeds.values())) != len(seeds):
        raise ValueError("Identity or seed collision; stop without redraw")
    check_disjoint(layouts, forbidden if forbidden is not None else forbidden_values())
    smoke = [next(l for l in selection if l["distractor_count"] == n) for n in (2, 1)]
    return dict(version=VERSION, namespace=NAMESPACE, master=p2.MASTER, p2_version=p2.VERSION, rng_seeds=seeds,
                layouts=layouts, tiers=dict(smoke=[l["layout_id"] for l in smoke],
                                           selection=[l["layout_id"] for l in selection],
                                           comparison=[l["layout_id"] for l in comparison]))


def write(out):
    """Write plan.json and plan.sha256 under `out` (created; refuses an existing directory). Returns the digest."""
    out = Path(out)
    plan = make_plan()
    out.mkdir(parents=True)
    value = p2.digest(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (out / "plan.sha256").write_text(value + "\n")
    return value
