"""Pilot layouts: 192 comparison (48 cells x 4), 2 smoke, an EMPTY selection tier. Pure; admission image.

Same P2 samplers and the same layout schema as M0's plan (`vla_inputs_eval.plan`, imported read-only: nothing in the frozen
core is edited), under the namespace `vla_inputs:pilot`, which no earlier plan used. The pilot recipe has no selection
stage, so the selection tier is empty and the two smoke layouts are extra layouts (trained cells, two and one
distractors) rather than members of a tier; smoke trials only check that the rollout machinery runs and are never scored.
`check_disjoint` is M0's, applied to a forbidden set that adds M0's whole plan (selection and comparison tiers) to the P2
and P2 re-test seeds and poses. Design: docs/superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md.
"""
import json
from pathlib import Path

from experiments.props_p2_data import plan as p2
from experiments.vla_inputs_eval import plan as m0

VERSION = "vla_inputs_pilot_v1"
NAMESPACE = "vla_inputs:pilot"
COMPARISON_PER_CELL = 4
COMPARISON_CELLS = m0.COMPARISON_CELLS
SMOKE_CELLS = tuple(next(c for c in m0.SELECTION_CELLS if c[3] == n) for n in (2, 1))

cell_name = m0.cell_name
layout_id = m0.layout_id
check_disjoint = m0.check_disjoint


def forbidden_values():
    """M0's forbidden set (sealed P2 and P2 re-test) plus every seed and pose in M0's own plan."""
    forbidden = m0.forbidden_values()
    seeds, poses = set(forbidden["seeds"]), set(forbidden["poses"])
    earlier = m0.make_plan()
    seeds |= set(earlier["rng_seeds"].values())
    for layout in earlier["layouts"].values():
        seeds.add(layout["seed"])
        poses.add(tuple(layout["start"]))
        poses |= {tuple(d["pose"]) for d in layout["distractors"]}
    return {"seeds": seeds, "poses": poses}


def check_unique(layouts):
    """Raises ValueError on a repeated layout id, seed or start pose inside this plan."""
    ids = [l["layout_id"] for l in layouts.values()]
    if len(set(ids)) != len(ids) or any(k != l["layout_id"] for k, l in layouts.items()):
        raise ValueError("Layout id collision; stop without redraw")
    if len({l["seed"] for l in layouts.values()}) != len(layouts):
        raise ValueError("Seed collision; stop without redraw")
    if len({tuple(l["start"]) for l in layouts.values()}) != len(layouts):
        raise ValueError("Start pose collision; stop without redraw")


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


def make_plan(forbidden=None):
    """The whole plan, byte-deterministic. Raises if any draw is forbidden or repeated (never redraws)."""
    seeds = {}
    smoke = _tier("smoke", SMOKE_CELLS, 1, seeds)
    comparison = _tier("comparison", COMPARISON_CELLS, COMPARISON_PER_CELL, seeds)
    layouts = {l["layout_id"]: l for l in smoke + comparison}
    if len(layouts) != len(smoke) + len(comparison) or len(set(seeds.values())) != len(seeds):
        raise ValueError("Identity or seed collision; stop without redraw")
    check_unique(layouts)
    check_disjoint(layouts, forbidden if forbidden is not None else forbidden_values())
    return dict(version=VERSION, namespace=NAMESPACE, master=p2.MASTER, p2_version=p2.VERSION, rng_seeds=seeds,
                layouts=layouts, tiers=dict(smoke=[l["layout_id"] for l in smoke], selection=[],
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
