"""B3 evaluation protocol: which trials, which seeds, which instructions. Pure; no simulator.

Reads the reserved layouts from the sealed plan (`PLAN_JSON`) and never writes under `PARENT`.
Rules fixed before any result: spec section 4 of
docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md.
"""
from contextlib import contextmanager
from itertools import permutations
import json
from pathlib import Path

from experiments.b3_robosuite_data.plan import COLOURS, HELD_OUT_PAIRS, seed
from experiments.paired_demonstrations.plan import instruction
from experiments.xvla_single_task import contract as single_task

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/b3_robosuite_data/b3_robosuite_20261003'
BANK_NPZ = PARENT/'bank.npz'
BANK_JSON = PARENT/'bank.json'
PLAN_JSON = PARENT/'plan.json'
EVAL_ROOT = ROOT/'outputs/xvla_b3_robosuite_eval'
CANDIDATES = (62400, 93600, 124800, 156000)  # epochs 2-5; update 1 and 31,200 are not evaluated
MODES = ('smoke', 'screen', 'report')
REPORT_SEEDS = 2
CONDITIONS_REPORT = ('control', 'distractor')
ALL_INSTRUCTIONS = tuple(sorted(instruction(p, q) for p, q in permutations(COLOURS, 2)))


def _plan():
    return json.loads(PLAN_JSON.read_text())


def _trial(ordinal, layout_id, layout, condition, trial_seed, seed_index):
    pick, place = layout['pick'], layout['place']
    present = {pick, place} | (set(layout['conditions'][condition]) if condition == 'distractor' else set())
    return dict(ordinal=ordinal, layout_id=layout_id, condition=condition, seed=trial_seed,
                seed_index=seed_index, pick=pick, place=place, side=layout['side'],
                instruction=instruction(pick, place),
                kind='held_out' if (pick, place) in HELD_OUT_PAIRS else 'trained',
                edge=layout['edge'], colours=sorted(present), bank_key=f'{layout_id}:{condition}')


def screen_trials():
    """20 trials: the screen layouts, distractor condition, one rollout seed each."""
    plan = _plan()
    hold = plan['evaluation_holdout']
    return [_trial(n, lid, hold[lid], 'distractor', seed(f'evaluation:screen:{lid}'), 0)
            for n, lid in enumerate(plan['evaluation_screen_ids'], start=1)]


def report_trials():
    """160 trials: 40 report layouts x {control, distractor} x 2 seeds; both conditions share a seed."""
    plan = _plan()
    hold = plan['evaluation_holdout']
    trials = []
    for lid in plan['evaluation_report_ids']:
        for k in range(REPORT_SEEDS):
            for condition in CONDITIONS_REPORT:
                trials.append(_trial(len(trials)+1, lid, hold[lid], condition,
                                     seed(f'evaluation:report:{lid}:{k}'), k))
    return trials


@contextmanager
def eval_instruction(text):
    """Swap the approved instruction `xvla_single_task.contract.observation` compares against.

    Accepts the 12 ordered colour pairs, including the two held-out ones, which are allowed only
    in this package. The shared module hardcodes one `TASK` and is bound by several contracts, so
    it is rebound through this context and restored even when the body raises.
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
