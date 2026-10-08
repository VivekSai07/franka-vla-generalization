"""Props P1 evaluation protocol: which trials, which seeds, which instructions. Pure; no simulator.

Reads the reserved layouts from the sealed plan (`PLAN_JSON`) and never writes under `PARENT`.
Rules fixed before any result: the "P1" and "P1 training and evaluation" sections of
docs/superpowers/specs/2026-10-05-props-sorting-design.md.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from experiments.props_p1_data.plan import HELD_OUT_PAIRS, TRAINED_PAIRS, instruction, seed as plan_seed
from experiments.xvla_single_task import contract as single_task

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/props_p1_data/props_p1_20261006'
PLAN_JSON = PARENT/'plan.json'
EVAL_ROOT = ROOT/'outputs/xvla_props_p1_eval'
# The six non-trivial checkpoints (update 1 is machinery sanity only, never screened).
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)
MODES = ('smoke', 'screen', 'report', 'confirm')
REPORT_SEEDS = 2
ALL_INSTRUCTIONS = tuple(sorted(instruction(obj, bin_) for obj, bin_ in (*TRAINED_PAIRS, *HELD_OUT_PAIRS)))


def _plan():
    return json.loads(PLAN_JSON.read_text())


def _trial(ordinal, layout_id, layout, trial_seed, seed_index):
    obj, bin_colour = layout['object'], layout['bin']
    return dict(ordinal=ordinal, layout_id=layout_id, seed=trial_seed, seed_index=seed_index,
                object=obj, bin=bin_colour, blue_slot=layout['blue_slot'], cell=list(layout['cell']),
                instruction=instruction(obj, bin_colour),
                kind='held_out' if (obj, bin_colour) in HELD_OUT_PAIRS else 'trained')


def screen_trials():
    """20 trials: the screen layouts, one rollout seed each."""
    plan = _plan()
    hold = plan['evaluation_holdout']
    return [_trial(n, lid, hold[lid], plan_seed(f'evaluation:screen:{lid}'), 0)
            for n, lid in enumerate(plan['evaluation_screen_ids'], start=1)]


def report_trials():
    """80 trials: 40 report layouts x 2 rollout seeds (20 trained pairs + 20 held-out-cell draws)."""
    plan = _plan()
    hold = plan['evaluation_holdout']
    trials = []
    for lid in plan['evaluation_report_ids']:
        for k in range(REPORT_SEEDS):
            trials.append(_trial(len(trials)+1, lid, hold[lid], plan_seed(f'evaluation:report:{lid}:{k}'), k))
    return trials


def confirm_trials():
    """20 trials: the confirm layouts, one rollout seed each. Touched once, after the report."""
    plan = _plan()
    hold = plan['evaluation_holdout']
    return [_trial(n, lid, hold[lid], plan_seed(f'evaluation:confirm:{lid}'), 0)
            for n, lid in enumerate(plan['evaluation_confirm_ids'], start=1)]


@contextmanager
def eval_instruction(text):
    """Swap the approved instruction `xvla_single_task.contract.observation` compares against.

    Accepts all 12 ordered object/bin pairs, including the two held-out ones, which are allowed
    only in this package. The shared module hardcodes one `TASK` and is bound by several
    contracts, so it is rebound through this context and restored even when the body raises.
    """
    if text not in ALL_INSTRUCTIONS:
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
        raise ValueError(f'Unapproved evaluation candidate; expected one of {CANDIDATES}')
    return EVAL_ROOT/f'{mode}_step_{candidate:06d}'
