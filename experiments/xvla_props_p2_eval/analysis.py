"""Props P2 evaluation read-outs, failure classifier, selection rules and the four success criteria. Pure numpy.

Fixed before any rollout (docs/superpowers/plans/2026-10-16-props-p2-evaluation.md) and not changed after a
result is seen. Intervals are layout-clustered: layouts, not trials, are the independent units.

An object is *lifted* when its centre rises LIFT_M above its start and stays up SUSTAIN_TICKS. Measured on the 540
sealed expert episodes: an untouched distractor rises at most 0.02 cm (the lemon, which rocks, 0.00), a carried
target 18.3-20.8 cm, so the 2 cm threshold is far from both and `wrong_object` is unambiguous.
"""
from collections import Counter, defaultdict
from math import comb

import numpy as np

LIFT_M = 0.02
SUSTAIN_TICKS = 15
CAUSES = ('success', 'never_lifted', 'wrong_object', 'wrong_bin', 'unstable_in_target', 'timeout_carrying',
          'dropped_outside')
CRITERIA_OVERALL, CRITERIA_WRONG_OBJECT, CRITERIA_KIND_GAP = 0.70, 0.05, 0.10


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


def first_lifted(props_z, names):
    """Name of the object lifted earliest, or None. `props_z` (ticks, objects), columns in `names` order."""
    z = np.asarray(props_z, dtype=float)
    if z.ndim != 2 or z.shape[1] != len(names):
        raise ValueError('props_z must be (ticks, len(names))')
    ticks = [lift_tick(z[:, i]) for i in range(len(names))]
    lifted = [(t, i) for i, t in enumerate(ticks) if t is not None]
    return names[min(lifted)[1]] if lifted else None


def classify(trace, target, bin_colour, success):
    """First cause of a trial, mutually exclusive, in the order of CAUSES.

    `trace`: `props_z` (ticks, objects), `names`, `placed_bin` (the bin the target rests in at the end, or None),
    `final_gripper_open`. P1's chain is kept once the target was the first object lifted; the most specific
    condition (`placed_bin`) is checked before the gripper state (FAILURES #53).
    """
    if success:
        return 'success'
    first = first_lifted(trace['props_z'], trace['names'])
    if first is None:
        return 'never_lifted'
    if first != target:
        return 'wrong_object'
    if trace['placed_bin'] is not None:
        return 'wrong_bin' if trace['placed_bin'] != bin_colour else 'unstable_in_target'
    if not trace['final_gripper_open']:
        return 'timeout_carrying'
    return 'dropped_outside'


# --- statistics ----------------------------------------------------------------------------------------------------

def clustered_interval(rows, key, n=10_000, seed=0):
    """95% percentile bootstrap over layouts (each with all its trials); `key` is a field or a callable."""
    value = key if callable(key) else (lambda row: row[key])
    groups = defaultdict(list)
    for row in rows:
        groups[row['layout_id']].append(float(value(row)))
    if not groups:
        return (float('nan'), float('nan'))
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(n, len(groups)))
    rates = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def _rate(rows, key):
    if not rows:
        return dict(n=0, layouts=0, count=0, rate=None, interval=None)
    count = sum(bool(key(r)) for r in rows)
    return dict(n=len(rows), layouts=len({r['layout_id'] for r in rows}), count=count, rate=count/len(rows),
                interval=list(clustered_interval(rows, lambda r: bool(key(r)))))


def mcnemar_exact(only_a, only_b):
    """Two-sided exact p for discordant pairs (a succeeded and b failed: `only_a`; the reverse: `only_b`)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def readouts(rows, excluded_layouts):
    """Every read-out fixed by the plan, after dropping layouts the expert cannot solve."""
    excluded = set(excluded_layouts)
    kept = [r for r in rows if r['layout_id'] not in excluded]
    success = lambda r: r['task_success']
    wrong_object = lambda r: r['wrong_object']
    wrong_bin = lambda r: r['cause'] == 'wrong_bin'
    by = lambda field, values: {str(v): _rate([r for r in kept if r[field] == v], success) for v in values}
    return dict(
        dropped_infeasible_trials=len(rows) - len(kept), excluded_layouts=sorted(excluded),
        overall=_rate(kept, success),
        by_kind=by('kind', ('trained', 'held_out')),
        by_distractor_count=by('distractor_count', (0, 1, 2)),
        # The physical side the blue bin is on, not the instructed colour: a position shortcut is invisible if
        # grouped by target colour.
        by_blue_slot=by('blue_slot', ('left', 'right')),
        wrong_object=dict(overall=_rate(kept, wrong_object),
                          by_kind={k: _rate([r for r in kept if r['kind'] == k], wrong_object)
                                   for k in ('trained', 'held_out')},
                          by_distractor_count={str(n): _rate([r for r in kept if r['distractor_count'] == n],
                                                             wrong_object) for n in (1, 2)}),
        wrong_bin=dict(overall=_rate(kept, wrong_bin)),
        causes={cause: sum(1 for r in kept if r['cause'] == cause) for cause in CAUSES},
    )


def success_count(rows, excluded_layouts):
    excluded = set(excluded_layouts)
    return sum(1 for r in rows if r['layout_id'] not in excluded and r['task_success'])


# --- selection rules (fixed before any result) -----------------------------------------------------------------------

def screen_survivors(counts, tie=2):
    """Every candidate within `tie` successes of the best screen count goes to the report (a tie is a tie)."""
    best = max(counts.values())
    return sorted(c for c, v in counts.items() if best - v <= tie)


def report_winner(counts):
    """The highest report success count; ties go to the later checkpoint."""
    best = max(counts.values())
    return max(c for c, v in counts.items() if v == best)


# --- the four pre-registered criteria --------------------------------------------------------------------------------

def criteria(p2_rows, p1_rows, excluded_layouts):
    """Evaluate the four criteria on the winner's report tier against P1's policy on the same trials."""
    excluded = set(excluded_layouts)
    p2 = [r for r in p2_rows if r['layout_id'] not in excluded]
    p1 = [r for r in p1_rows if r['layout_id'] not in excluded]
    p2_by = {(r['ordinal'], r['layout_id']): r for r in p2}
    p1_by = {(r['ordinal'], r['layout_id']): r for r in p1}
    shared = sorted(set(p2_by) & set(p1_by))
    if not shared:
        raise ValueError('No trial shared by the two policies')
    a = [p2_by[k] for k in shared]
    b = [p1_by[k] for k in shared]
    overall = _rate(a, lambda r: r['task_success'])
    other = _rate(b, lambda r: r['task_success'])
    wrong = _rate(a, lambda r: r['wrong_object'])
    trained = _rate([r for r in a if r['kind'] == 'trained'], lambda r: r['task_success'])
    held = _rate([r for r in a if r['kind'] == 'held_out'], lambda r: r['task_success'])
    gap = None if trained['rate'] is None or held['rate'] is None else abs(trained['rate'] - held['rate'])
    only_a = sum(1 for x, y in zip(a, b) if x['task_success'] and not y['task_success'])
    only_b = sum(1 for x, y in zip(a, b) if y['task_success'] and not x['task_success'])
    separated = overall['interval'][0] > other['interval'][1]
    result = {
        '1_overall_at_least_70pct': dict(value=overall['rate'], threshold=CRITERIA_OVERALL, passed=overall['rate'] >= CRITERIA_OVERALL),
        '2_wrong_object_at_most_5pct': dict(value=wrong['rate'], threshold=CRITERIA_WRONG_OBJECT, passed=wrong['rate'] <= CRITERIA_WRONG_OBJECT),
        '3_kind_gap_within_10_points': dict(value=gap, signed=None if gap is None else trained['rate'] - held['rate'],
                                           threshold=CRITERIA_KIND_GAP, passed=gap is not None and gap <= CRITERIA_KIND_GAP),
        '4_beats_p1_non_overlapping': dict(p2=overall, p1=other, p1_minus_p2_discordant=[only_b, only_a],
                                           mcnemar_exact_p=mcnemar_exact(only_a, only_b),
                                           passed=bool(overall['rate'] > other['rate'] and separated)),
        'trials': len(shared),
    }
    result['all_passed'] = all(v['passed'] for k, v in result.items() if k[0].isdigit())
    return result
