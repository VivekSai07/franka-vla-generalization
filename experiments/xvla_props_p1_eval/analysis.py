"""Props P1 evaluation read-outs. Pure numpy; no simulator.

The read-outs and their interpretation are fixed by the "P1 training and evaluation" section of
docs/superpowers/specs/2026-10-05-props-sorting-design.md and are not changed after any result is
seen. Intervals are layout-clustered: layouts, not trials, are the independent units.

Unlike B3 (pick/place cubes, a stacking success rule, control/distractor conditions), props P1 has
one object and two bins: `classify` and the read-outs below replace B3's cube-specific causes and
the distractor contrast with `wrong_bin` (placed stably in the *other* bin — the clean signature of
a colour that was not read) and the arrangement split by `blue_slot` (Review Focus 5: this must be
the physical side the blue bin is on, not the target colour, or a position shortcut is invisible).
"""
from collections import Counter, defaultdict

import numpy as np

LIFT_M = 0.03        # the prop counts as lifted when it is this far above where it started
SUSTAIN_TICKS = 15   # ... and stays there; a bump is not a lift
CAUSES = ('success', 'wrong_bin', 'never_lifted', 'timeout_carrying', 'dropped_outside')


def lifted_and_sustained(z_series, threshold_m=LIFT_M, sustain=SUSTAIN_TICKS):
    """Whether the prop rose `threshold_m` above its start and stayed up `sustain` ticks."""
    z = np.asarray(z_series, dtype=float)
    if len(z) == 0:
        return False
    up = z - z[0] > threshold_m
    for tick in range(0, len(up) - sustain + 1):
        if up[tick] and up[tick:tick + sustain].all():
            return True
    return False


def classify(trace, bin_colour, success):
    """First cause of a trial: success, wrong_bin, never_lifted, timeout_carrying, dropped_outside.

    `trace`: `prop_z` (ticks,), `placed_bin` (str or None — whichever bin it rests in at the end,
    if any), `final_gripper_open` (bool).
    """
    if success:
        return 'success'
    if not lifted_and_sustained(trace['prop_z']):
        return 'never_lifted'
    if trace['placed_bin'] is not None and trace['placed_bin'] != bin_colour:
        return 'wrong_bin'
    if not trace['final_gripper_open']:
        return 'timeout_carrying'
    return 'dropped_outside'


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


def readouts(rows, excluded_layouts):
    """Every read-out fixed by the spec, after dropping layouts the expert cannot solve."""
    excluded = set(excluded_layouts)
    kept = [r for r in rows if r['layout_id'] not in excluded]
    success = lambda r: r['task_success']
    wrong = lambda r: r['cause'] == 'wrong_bin'
    by = lambda field, values: {v: _rate([r for r in kept if r[field] == v], success) for v in values}
    return dict(
        dropped_infeasible_trials=len(rows) - len(kept), excluded_layouts=sorted(excluded),
        overall=_rate(kept, success),
        by_kind=by('kind', ('trained', 'held_out')),
        # Review Focus 5: split by the physical arrangement (which side the blue bin is on), not by
        # the instructed colour — a position shortcut is invisible if grouped by target colour.
        by_blue_slot=by('blue_slot', ('left', 'right')),
        wrong_bin=dict(overall=_rate(kept, wrong),
                       by_kind={k: _rate([r for r in kept if r['kind'] == k], wrong)
                                for k in ('trained', 'held_out')}),
        causes={cause: dict(Counter(r['cause'] for r in kept))[cause] if any(r['cause'] == cause for r in kept) else 0
                for cause in CAUSES},
    )
