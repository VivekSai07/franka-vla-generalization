"""M0 trials, stages and sharding. Pure; admission image. Part of the frozen shared core.

A stage is one tier of one arm at one checkpoint. Its trials are the plan's layouts of that tier that passed the expert
feasibility pass, in plan order, one rollout seed each (the layout's own, shared by every arm). Every stage folder holds one
folder per trial, written by exactly one worker.
"""
from pathlib import Path

from experiments.props_p2_data.plan import HELD_OUT_PAIRS

TIERS = ("smoke", "selection", "comparison")
TRIAL_FILES = ("cameras.mp4", "trace.npz", "episode.json")


def trial_of(layout, ordinal):
    """The trial record of a layout; the fields P2's classifier and read-outs expect."""
    obj, bin_colour = layout["object"], layout["bin"]
    return dict(ordinal=ordinal, layout_id=layout["layout_id"], tier=layout["tier"], seed=layout["seed"], seed_index=0,
                object=obj, bin=bin_colour, blue_slot=layout["blue_slot"], cell=list(layout["cell"]),
                instruction=layout["instruction"], kind="held_out" if (obj, bin_colour) in HELD_OUT_PAIRS else "trained",
                distractor_count=layout["distractor_count"], distractors=layout["distractors"], start=layout["start"],
                home_pose_jitter_deg=layout["home_pose_jitter_deg"])


def stage_trials(plan, feasibility, tier):
    """Trials of `tier`: feasible layouts only, ordinals = position in the plan's tier list (dropped ones leave a gap)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    feasible = set(feasibility["feasible"])
    return [trial_of(plan["layouts"][lid], i + 1) for i, lid in enumerate(plan["tiers"][tier]) if lid in feasible]


def shard_trials(trials, shard, shards):
    """Interleaved share of a stage, so every worker meets the easy and the hard layouts."""
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is not valid")
    return trials[shard::shards]


def stage_dir(root, arm, tier, checkpoint):
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    return Path(root) / "arms" / arm / tier / f"step_{checkpoint:06d}"


def trial_folder(directory, trial):
    return Path(directory) / f"trial_{trial['ordinal']:03d}_{trial['layout_id']}"


def trial_is_complete(folder):
    return all((Path(folder) / name).is_file() for name in TRIAL_FILES)
