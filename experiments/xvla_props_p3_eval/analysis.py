"""Props P3 evaluation read-outs, failure classifier, selection rules and the pre-registered bin-choice criteria. Pure numpy.

Fixed before any rollout (spec revised 2026-10-20, docs/superpowers/plans/2026-10-21-props-p3-training-eval.md) and
not changed after a result is seen. One rollout seed per layout, so the layout is the independent unit and intervals
are Wilson intervals over layouts.

The gate is *bin choice*, not overall success: motor failures (never lifted, hold, carry) dominate this robot and say
nothing about Food vs Drinks. Bin choice is defined on **stable placements**: trials in which the object came to rest
inside either bin (`placed_bin` is not None at the end), scored correct when that bin is the object's category's.

An object is *lifted* when its centre rises LIFT_M above its start and stays up SUSTAIN_TICKS.
"""
from math import sqrt

import numpy as np

LIFT_M = 0.02
SUSTAIN_TICKS = 15
CAUSES = ('success', 'never_lifted', 'wrong_bin', 'unstable_in_target', 'timeout_carrying', 'dropped_outside')
OBJECTS = ('can', 'bread', 'lemon', 'milk', 'bottle', 'cereal')
HELD_OUT = ('bottle', 'cereal')
WRONG_BIN_MAX = 0.05        # trained objects: wrong-bin share of stable placements, point estimate
TRAINED_FLOOR = 60          # stable placements needed among the 80 trained report layouts
HELD_OUT_FLOOR = 30         # stable placements needed per held-out object (of 80 layouts)
EVIDENCE_LOWER_BOUND = 0.5  # Wilson 95% lower bound of held-out bin-choice accuracy must exceed this, for both objects


def lift_tick(z_series, threshold_m=LIFT_M, sustain=SUSTAIN_TICKS):
    """First tick at which the object is `threshold_m` above where it started and stays so `sustain` ticks."""
    z = np.asarray(z_series, dtype=float)
    if len(z) == 0:
        return None
    up = z - z[0] > threshold_m
    for tick in range(0, len(up) - sustain + 1):
        if up[tick] and up[tick:tick + sustain].all():
            return tick
    return None


def classify(trace, bin_colour, success):
    """First cause of a trial, mutually exclusive, in the order of CAUSES.

    `trace`: `props_z` (the object's height per tick), `placed_bin` (the bin it rests in at the end, or None),
    `final_gripper_open`. The most specific condition (`placed_bin`) is checked before the gripper state (FAILURES #53).
    """
    if success:
        return 'success'
    if lift_tick(trace['props_z']) is None:
        return 'never_lifted'
    if trace['placed_bin'] is not None:
        return 'wrong_bin' if trace['placed_bin'] != bin_colour else 'unstable_in_target'
    if not trace['final_gripper_open']:
        return 'timeout_carrying'
    return 'dropped_outside'


# --- statistics ----------------------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval for k of n, clipped to [0, 1]; (None, None) for n = 0 (JSON cannot hold NaN)."""
    if n == 0:
        return (None, None)
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator))


def _rate(rows, key):
    count = sum(bool(key(r)) for r in rows)
    if not rows:
        return dict(n=0, count=0, rate=None, interval=None)
    return dict(n=len(rows), count=count, rate=count / len(rows), interval=list(wilson(count, len(rows))))


def bin_choice(rows):
    """Bin choice among stable placements: how many layouts came to rest in a bin, and in the right one."""
    stable = [r for r in rows if r['placed_bin'] is not None]
    correct = sum(r['placed_bin'] == r['bin'] for r in stable)
    n = len(stable)
    return dict(layouts=len(rows), stable=n, correct=correct, wrong=n - correct,
                wrong_rate=None if n == 0 else (n - correct) / n,
                accuracy=None if n == 0 else correct / n,
                accuracy_interval=list(wilson(correct, n)), wrong_interval=list(wilson(n - correct, n)))


def readouts(rows, excluded_layouts):
    """Every read-out fixed by the plan, after dropping layouts the expert cannot solve (counted, not replaced)."""
    excluded = set(excluded_layouts)
    kept = [r for r in rows if r['layout_id'] not in excluded]
    success = lambda r: r['task_success']

    def group(field, values):
        return {str(v): _rate([r for r in kept if r[field] == v], success) for v in values}

    def per_object(obj):
        mine = [r for r in kept if r['object'] == obj]
        return dict(success=_rate(mine, success), bin_choice=bin_choice(mine),
                    causes={cause: sum(1 for r in mine if r['cause'] == cause) for cause in CAUSES},
                    by_blue_slot={s: dict(success=_rate([r for r in mine if r['blue_slot'] == s], success),
                                          bin_choice=bin_choice([r for r in mine if r['blue_slot'] == s]))
                                  for s in ('left', 'right')})

    return dict(
        dropped_infeasible_trials=len(rows) - len(kept), excluded_layouts=sorted(excluded),
        overall=_rate(kept, success),
        by_kind=group('kind', ('trained', 'held_out')),
        # The physical side of the blue bin, not the colour: a position shortcut is invisible if grouped by colour.
        by_blue_slot=group('blue_slot', ('left', 'right')),
        by_object={obj: per_object(obj) for obj in OBJECTS if any(r['object'] == obj for r in kept)},
        bin_choice=dict(trained=bin_choice([r for r in kept if r['kind'] == 'trained']),
                        held_out=bin_choice([r for r in kept if r['kind'] == 'held_out'])),
        causes={cause: sum(1 for r in kept if r['cause'] == cause) for cause in CAUSES},
        # Must be 0: a success is a correct stable placement by construction. Reported so a mismatch is visible.
        success_without_a_correct_stable_placement=sum(1 for r in kept if r['task_success'] and r['placed_bin'] != r['bin']),
    )


def selection_count(rows, excluded_layouts):
    """Successes on trained-object layouts (held-out layouts never enter checkpoint selection)."""
    excluded = set(excluded_layouts)
    return sum(1 for r in rows if r['layout_id'] not in excluded and r['kind'] == 'trained' and r['task_success'])


# --- selection rules (fixed before any result) -----------------------------------------------------------------------

def screen_survivors(counts, tie=2):
    """Every candidate within `tie` successes of the best screen count goes to the report (a tie is a tie)."""
    best = max(counts.values())
    return sorted(c for c, v in counts.items() if best - v <= tie)


def report_winner(counts):
    """The highest report success count on trained-object layouts; ties go to the later checkpoint."""
    best = max(counts.values())
    return max(c for c, v in counts.items() if v == best)


# --- the pre-registered criteria -------------------------------------------------------------------------------------

def criteria(rows, excluded_layouts):
    """Evaluate the criteria on the winner's report tier.

    1. trained objects: wrong-bin share of stable placements at most WRONG_BIN_MAX (verdict `not_interpretable` below
       TRAINED_FLOOR stable placements: neither pass nor fail);
    2. held-out objects, exploratory: for each object, `evidence` if the stable placements reach HELD_OUT_FLOOR and the
       Wilson lower bound of bin-choice accuracy exceeds EVIDENCE_LOWER_BOUND, `not_interpretable` below the floor,
       otherwise `not_shown` (inconclusive, not a negative); overall `evidence` only if both objects show it.
    """
    excluded = set(excluded_layouts)
    kept = [r for r in rows if r['layout_id'] not in excluded]
    trained = bin_choice([r for r in kept if r['kind'] == 'trained'])
    if trained['stable'] < TRAINED_FLOOR:
        verdict = 'not_interpretable'
    else:
        verdict = 'pass' if trained['wrong_rate'] <= WRONG_BIN_MAX else 'fail'
    held = {}
    for obj in HELD_OUT:
        choice = bin_choice([r for r in kept if r['object'] == obj])
        if choice['stable'] < HELD_OUT_FLOOR:
            status = 'not_interpretable'
        elif choice['accuracy_interval'][0] > EVIDENCE_LOWER_BOUND:
            status = 'evidence'
        else:
            status = 'not_shown'
        held[obj] = dict(choice, status=status)
    statuses = {v['status'] for v in held.values()}
    overall = 'evidence' if statuses == {'evidence'} else 'not_interpretable' if 'not_interpretable' in statuses else 'not_shown'
    return {
        '1_trained_wrong_bin_at_most_5pct': dict(choice=trained, threshold=WRONG_BIN_MAX, floor=TRAINED_FLOOR,
                                                 verdict=verdict),
        '2_held_out_generalisation_exploratory': dict(per_object=held, floor=HELD_OUT_FLOOR,
                                                      lower_bound_above=EVIDENCE_LOWER_BOUND, verdict=overall),
        'layouts': len(kept),
    }
