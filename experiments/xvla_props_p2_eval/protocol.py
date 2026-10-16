"""Props P2 evaluation protocol: which trials, which seeds, which jobs, how they are sharded. Pure; no simulator.

Reads the reserved layouts from the sealed P2 plan (`PLAN_JSON`) and never writes under `PARENT`. Rules fixed
before any result: the P2 spec (docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md) and
docs/superpowers/plans/2026-10-16-props-p2-evaluation.md.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from experiments.props_p2_data.plan import HELD_OUT_PAIRS, TRAINED_PAIRS, instruction, seed as plan_seed
from experiments.xvla_single_task import contract as single_task

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/props_p2_data/props_p2_20261013'
PLAN_JSON = PARENT/'plan.json'
EVAL_ROOT = ROOT/'outputs/xvla_props_p2_eval'
# The six non-trivial P2 checkpoints (update 1 is machinery sanity only, never screened) and the one P1 checkpoint
# the cross-run uses.
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)
P1_CHECKPOINT = 126900
POLICIES = ('p2', 'p1')
MODES = ('smoke', 'screen', 'report', 'control', 'confirm')
REPORT_SEEDS = 2
SCREEN_TIE_TRIALS = 2
ALL_INSTRUCTIONS = tuple(sorted(instruction(obj, bin_) for obj, bin_ in (*TRAINED_PAIRS, *HELD_OUT_PAIRS)))


def _plan():
    return json.loads(PLAN_JSON.read_text())


def _trial(ordinal, layout_id, layout, trial_seed, seed_index):
    obj, bin_colour = layout['object'], layout['bin']
    return dict(ordinal=ordinal, layout_id=layout_id, tier=layout['tier'], seed=trial_seed, seed_index=seed_index,
                object=obj, bin=bin_colour, blue_slot=layout['blue_slot'], cell=list(layout['cell']),
                instruction=instruction(obj, bin_colour),
                kind='held_out' if (obj, bin_colour) in HELD_OUT_PAIRS else 'trained',
                distractor_count=layout['distractor_count'], distractors=layout['distractors'],
                start=layout['start'], home_pose_jitter_deg=layout['home_pose_jitter_deg'])


def _tier_trials(tier, seeds):
    plan = _plan()
    hold = plan['evaluation_holdout']
    trials = []
    for lid in plan[f'evaluation_{tier}_ids']:
        for k in range(seeds):
            identity = f'evaluation:{tier}:{lid}' + (f':{k}' if seeds > 1 else '')
            trials.append(_trial(len(trials)+1, lid, hold[lid], plan_seed(identity), k))
    return trials


def screen_trials():
    """20 trials: the screen layouts, one rollout seed each."""
    return _tier_trials('screen', 1)


def report_trials():
    """120 trials: 60 report layouts (20 trained x 1 distractor, 20 trained x 2, 20 held-out) x 2 seeds."""
    return _tier_trials('report', REPORT_SEEDS)


def control_trials():
    """20 trials: one per trained cell, no distractor."""
    return _tier_trials('control', 1)


def confirm_trials():
    """20 trials: the confirm layouts, one rollout seed each. Touched once, on the report's winner."""
    return _tier_trials('confirm', 1)


def smoke_trials():
    held_out = next(t for t in report_trials() if t['kind'] == 'held_out')
    two = next(t for t in screen_trials() if t['distractor_count'] == 2)
    return [dict(two, ordinal=1), dict(held_out, ordinal=2)]


def select_trials(mode):
    functions = dict(smoke=smoke_trials, screen=screen_trials, report=report_trials, control=control_trials,
                     confirm=confirm_trials)
    if mode not in functions:
        raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    return functions[mode]()


def jobs(mode, policy, candidates):
    """Every (policy, candidate, trial) of a stage, ordered by candidate then ordinal."""
    if policy not in POLICIES:
        raise ValueError(f'Unapproved policy; expected one of {POLICIES}')
    allowed = CANDIDATES if policy == 'p2' else (P1_CHECKPOINT,)
    if not candidates or any(c not in allowed for c in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError(f'Unapproved candidates {candidates} for {policy}; expected distinct values of {allowed}')
    trials = select_trials(mode)
    return [dict(policy=policy, candidate=c, trial=t) for c in sorted(candidates) for t in trials]


def shard_jobs(all_jobs, shard, shards):
    """Interleaved share of a stage so every worker sees every checkpoint's mix of easy and hard trials;
    ordered by candidate so a worker loads each policy once."""
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f'shard {shard} of {shards} is not valid')
    mine = all_jobs[shard::shards]
    return sorted(mine, key=lambda j: (j['candidate'], j['trial']['ordinal']))


@contextmanager
def eval_instruction(text):
    """Swap the approved instruction `xvla_single_task.contract.observation` compares against.

    Accepts all 12 ordered object/bin pairs, including the two held-out ones, which are allowed only in this
    package. The shared module hardcodes one `TASK` and is bound by several contracts, so it is rebound through
    this context and restored even when the body raises.
    """
    if text not in ALL_INSTRUCTIONS:
        raise ValueError(f'Unapproved instruction: {text!r}')
    original = single_task.TASK
    single_task.TASK = text
    try:
        yield
    finally:
        single_task.TASK = original


def output_dir(mode, policy, candidate):
    if mode not in MODES:
        raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    if policy not in POLICIES:
        raise ValueError(f'Unapproved policy; expected one of {POLICIES}')
    if candidate not in (CANDIDATES if policy == 'p2' else (P1_CHECKPOINT,)):
        raise ValueError(f'Unapproved evaluation candidate {candidate} for {policy}')
    return EVAL_ROOT/f'{mode}_{policy}_step_{candidate:06d}'


def trial_folder(directory, trial):
    return Path(directory)/f"trial_{trial['ordinal']:03d}_{trial['layout_id']}"
