"""G0 layouts: 144 benchmark, 24 selection and 2 smoke fresh P2-style layouts, each with one level per axis. Pure; admission image.

Samplers are P2's own (`props_p2_data.plan`, imported read-only, as M0's plan does): start region, yaw caps, symmetric spacing, hand
clearance, shuffled distractor decks. Identities are `gen:g0:<tier>:<cell>:<i>`, a namespace no earlier plan used; `check_disjoint`
(M0's) refuses any seed or pose that coincides with a P2, P2 re-test, M0 or pilot layout. Benchmark: M0's 48 comparison cells x 3 = 144
layouts. Selection (for choosing a later arm's checkpoint, never for claims): 24 of the 48 cells (every other one) x 1, other
identities. Smoke: two layouts (two and one distractors), never scored.

Every layout is run once under nominal conditions and once under each of the five axes (`gen_scene.levels.assign` gives it one
level per axis: layout n of a tier gets level (n + offset) % levels, so every level is used equally often). The conditions,
including every drawn pose and texture, are stored here and hashed with the plan: nothing is drawn at run time.
Design: docs/superpowers/specs/2026-11-06-g0-generalization-benchmark-design.md (D3, D4, D5).
"""
import json
from pathlib import Path

from experiments.gen_scene import levels
from experiments.props_p2_data import plan as p2
from experiments.vla_inputs_eval import plan as m0
from experiments.vla_inputs_pilot_plan import plan as pilot

VERSION = "gen_g0_v1"
NAMESPACE = "gen:g0"
PER_CELL = {"benchmark": 3, "selection": 1}
CELLS = {"benchmark": m0.COMPARISON_CELLS, "selection": m0.COMPARISON_CELLS[::2]}
SMOKE_CELLS = pilot.SMOKE_CELLS
TIERS = ("smoke", "selection", "benchmark")

cell_name = m0.cell_name
layout_id = m0.layout_id
check_disjoint = m0.check_disjoint
check_unique = pilot.check_unique


def forbidden_values():
    """Every seed and pose of the sealed P2 plan, the P2 re-test, M0's plan and the pilot plan."""
    forbidden = pilot.forbidden_values()
    seeds, poses = set(forbidden["seeds"]), set(forbidden["poses"])
    earlier = pilot.make_plan()
    seeds |= set(earlier["rng_seeds"].values())
    for layout in earlier["layouts"].values():
        seeds.add(layout["seed"])
        poses.add(tuple(layout["start"]))
        poses |= {tuple(d["pose"]) for d in layout["distractors"]}
    return {"seeds": seeds, "poses": poses}


def _tier(tier, cells, per_cell, seeds):
    layouts = []
    for cell in cells:
        obj, bin_colour, slot, count = cell
        deck = m0._deck(obj, count, per_cell, f"{NAMESPACE}:{tier}:{cell_name(cell)}:deck", seeds)
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


def _with_conditions(layouts, manifest=None):
    """Attach `axes`: {axis: {level, condition, condition_sha256}} to each layout of one tier, in tier order."""
    assigned = levels.assign(layouts, manifest)
    out = []
    for layout in layouts:
        axes = {axis: dict(level=c.level, condition=c.to_dict(), condition_sha256=c.digest())
                for axis, c in assigned[layout["layout_id"]].items()}
        out.append(dict(layout, axes=axes))
    return out


def make_plan(forbidden=None, manifest=None):
    """The whole plan, byte-deterministic. Raises if any draw is forbidden or repeated (never redraws)."""
    seeds = {}
    tiers = {"smoke": _tier("smoke", SMOKE_CELLS, 1, seeds)}
    for tier in ("selection", "benchmark"):
        tiers[tier] = _tier(tier, CELLS[tier], PER_CELL[tier], seeds)
    tiers = {tier: _with_conditions(ls, manifest) for tier, ls in tiers.items()}
    layouts = {l["layout_id"]: l for ls in tiers.values() for l in ls}
    if len(layouts) != sum(len(ls) for ls in tiers.values()) or len(set(seeds.values())) != len(seeds):
        raise ValueError("Identity or seed collision; stop without redraw")
    check_unique(layouts)
    check_disjoint(layouts, forbidden if forbidden is not None else forbidden_values())
    return dict(version=VERSION, namespace=NAMESPACE, master=p2.MASTER, p2_version=p2.VERSION, levels_version=levels.VERSION,
                levels_master=levels.MASTER, rng_seeds=seeds, layouts=layouts,
                tiers={tier: [l["layout_id"] for l in ls] for tier, ls in tiers.items()})


def write(out):
    """Write plan.json and plan.sha256 under `out` (created; refuses an existing directory). Returns the digest."""
    out = Path(out)
    plan = make_plan()
    out.mkdir(parents=True)
    value = p2.digest(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (out / "plan.sha256").write_text(value + "\n")
    return value
