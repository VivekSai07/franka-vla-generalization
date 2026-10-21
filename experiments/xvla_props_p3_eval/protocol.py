"""Props P3 evaluation protocol: which trials, which seeds, which jobs, how they are sharded. Pure; no simulator.

Reads the reserved layouts from the sealed P3 plan (`PLAN_JSON`) and never writes under `PARENT`. Rules fixed before
any rollout: the P3 spec (docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md, revised 2026-10-20) and
docs/superpowers/plans/2026-10-21-props-p3-training-eval.md.

288 reserved layouts, one rollout seed each: screen 24 (3 per trained cell), report 240 (80 trained, 80 per held-out
object), confirm 24. Every trial carries the one fixed prompt. There is no control tier and no cross-run policy: with
one object per episode and no distractors there is nothing to control for, and a P1 policy has no meaningful reading of
the prompt.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from experiments.props_p3_data.plan import PROMPT, seed as plan_seed
from experiments.xvla_single_task import contract as single_task

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/props_p3_data/props_p3_20261020'
PLAN_JSON = PARENT/'plan.json'
EVAL_ROOT = ROOT/'outputs/xvla_props_p3_eval'
# The six non-trivial checkpoints (update 1 is machinery sanity only, never screened).
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)
MODES = ('smoke', 'screen', 'report', 'confirm')
TIERS = ('screen', 'report', 'confirm')
SCREEN_TIE_TRIALS = 2
LAYOUTS = 288


def _plan():
    return json.loads(PLAN_JSON.read_text())


def _trial(ordinal, layout_id, layout):
    return dict(ordinal=ordinal, layout_id=layout_id, tier=layout['tier'], seed=plan_seed(f'rollout:{layout_id}'),
                seed_index=0, object=layout['object'], category=layout['category'], bin=layout['bin'],
                blue_slot=layout['blue_slot'], cell=list(layout['cell']), instruction=PROMPT,
                kind='held_out' if layout['held_out'] else 'trained',
                start=layout['start'], home_pose_jitter_deg=layout['home_pose_jitter_deg'])


def _tier_trials(tier):
    plan = _plan()
    hold = plan['evaluation_holdout']
    trials = []
    for lid in plan[f'evaluation_{tier}_ids']:
        trials.append(_trial(len(trials)+1, lid, hold[lid]))
    return trials


def screen_trials():
    """24 trials: the screen layouts (trained objects only), one rollout seed each."""
    return _tier_trials('screen')


def report_trials():
    """240 trials: 80 trained-object layouts and 160 held-out (80 bottle, 80 cereal), one seed each."""
    return _tier_trials('report')


def confirm_trials():
    """24 trials: the confirm layouts, one rollout seed each. Touched once, on the report's winner."""
    return _tier_trials('confirm')


def smoke_trials():
    held_out = next(t for t in report_trials() if t['kind'] == 'held_out')
    trained = screen_trials()[0]
    return [dict(trained, ordinal=1), dict(held_out, ordinal=2)]


def select_trials(mode):
    functions = dict(smoke=smoke_trials, screen=screen_trials, report=report_trials, confirm=confirm_trials)
    if mode not in functions:
        raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    return functions[mode]()


def jobs(mode, candidates):
    """Every (candidate, trial) of a stage, ordered by candidate then ordinal."""
    if not candidates or any(c not in CANDIDATES for c in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError(f'Unapproved candidates {candidates}; expected distinct values of {CANDIDATES}')
    trials = select_trials(mode)
    return [dict(candidate=c, trial=t) for c in sorted(candidates) for t in trials]


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

    The one fixed prompt is the only text ever accepted. The shared module hardcodes one `TASK` and is bound by
    several contracts, so it is rebound through this context and restored even when the body raises.
    """
    if text != PROMPT:
        raise ValueError(f'Unapproved instruction: {text!r}')
    original = single_task.TASK
    single_task.TASK = text
    try:
        yield
    finally:
        single_task.TASK = original


def output_dir(mode, candidate):
    if mode not in MODES:
        raise ValueError(f'Unapproved evaluation mode; expected one of {MODES}')
    if candidate not in CANDIDATES:
        raise ValueError(f'Unapproved evaluation candidate {candidate}')
    return EVAL_ROOT/f'{mode}_step_{candidate:06d}'


def trial_folder(directory, trial):
    return Path(directory)/f"trial_{trial['ordinal']:03d}_{trial['layout_id']}"
