"""G0 trials, stages and sharding. Pure; admission image. Part of the frozen shared core.

A stage is one tier of one arm at one checkpoint. Its trials are, for every layout of the tier in plan order, one trial per
condition: `nominal` and the five axes, in that order (`ordinal` = 6 x layout position + axis position + 1, a gap where a trial was
dropped). A trial runs only if it passed the expert feasibility pass (D8); one whose layout is infeasible under NOMINAL conditions
is dropped with all its siblings, because the paired comparison needs the nominal trial. Each trial runs with its layout's own
rollout seed, the same under every condition and for every arm. Every stage folder holds one folder per trial, written by exactly
one worker.
"""
from pathlib import Path

from experiments.gen_scene import levels
from experiments.gen_scene.conditions import NOMINAL
from experiments.props_p2_data.plan import HELD_OUT_PAIRS

TIERS = ("smoke", "selection", "benchmark")
AXES = ("nominal", *levels.AXES)
TRIAL_FILES = ("cameras.mp4", "trace.npz", "episode.json")


def trial_key(layout_id, axis):
    return f"{layout_id}:{axis}"


def condition_of(layout, axis):
    """(level, condition dict) of a layout under `axis`; nominal for `nominal`."""
    if axis == "nominal":
        return "nominal", NOMINAL.to_dict()
    entry = layout["axes"][axis]
    return entry["level"], entry["condition"]


def trial_of(layout, axis, ordinal):
    """The trial record of a layout under one axis; the fields P2's classifier and the read-outs expect, plus the condition."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    level, condition = condition_of(layout, axis)
    obj, bin_colour = layout["object"], layout["bin"]
    max_ticks = levels.DISTURBANCE_TICKS if condition["disturbance"] != "none" else levels.NOMINAL_TICKS
    return dict(ordinal=ordinal, layout_id=layout["layout_id"], tier=layout["tier"], seed=layout["seed"], seed_index=0,
                object=obj, bin=bin_colour, blue_slot=layout["blue_slot"], cell=list(layout["cell"]),
                instruction=layout["instruction"], kind="held_out" if (obj, bin_colour) in HELD_OUT_PAIRS else "trained",
                distractor_count=layout["distractor_count"], distractors=layout["distractors"], start=layout["start"],
                home_pose_jitter_deg=layout["home_pose_jitter_deg"], axis=axis, level=level, condition=condition,
                max_ticks=max_ticks)


def ordinal_of(layout_position, axis):
    return layout_position * len(AXES) + AXES.index(axis) + 1


def stage_trials(plan, feasibility, tier):
    """Trials of `tier`: feasible ones only, in plan order, ordinals keeping their plan position."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    feasible = set(feasibility["feasible_trials"])
    trials = []
    for position, layout_id in enumerate(plan["tiers"][tier]):
        layout = plan["layouts"][layout_id]
        for axis in AXES:
            if trial_key(layout_id, axis) in feasible:
                trials.append(trial_of(layout, axis, ordinal_of(position, axis)))
    return trials


def shard_trials(trials, shard, shards):
    """A share of a stage. Trial (layout position p, axis a) goes to shard (p + a) % shards, so a layout's six trials go to six
    different workers and every worker gets every axis equally often (the disturbance trials, which run longer, spread evenly)."""
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is not valid")
    out = []
    for trial in trials:
        index = trial["ordinal"] - 1
        if (index // len(AXES) + index % len(AXES)) % shards == shard:
            out.append(trial)
    return out


def stage_dir(root, arm, tier, checkpoint):
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    return Path(root) / "arms" / arm / tier / f"step_{checkpoint:06d}"


def trial_folder(directory, trial):
    return Path(directory) / f"trial_{trial['ordinal']:04d}_{trial['layout_id']}_{trial['axis']}"


def trial_is_complete(folder):
    return all((Path(folder) / name).is_file() for name in TRIAL_FILES)
