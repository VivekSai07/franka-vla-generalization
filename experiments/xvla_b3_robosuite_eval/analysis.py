"""B3 evaluation read-outs. Pure numpy; no simulator.

The read-outs and their interpretation are fixed by spec section 4 and are not changed after any
result is seen. Intervals are layout-clustered: layouts, not trials, are the independent units.
"""
from collections import Counter, defaultdict

import numpy as np

LIFT_M = 0.03        # a cube counts as lifted when it is this far above where it started
SUSTAIN_TICKS = 15   # ... and stays there; a bump is not a lift
CAUSES = ('success', 'wrong_cube', 'never_lifted', 'lifted_not_carried', 'released_off_target')
GRIPPER_OPEN = 255


def first_lifted(trace, threshold_m=LIFT_M, sustain=SUSTAIN_TICKS):
    """Colour of the first cube that rises `threshold_m` above its start and stays up `sustain` ticks."""
    cube_xyz = np.asarray(trace['cube_xyz'])
    colours = [str(c) for c in np.asarray(trace['colours'])]
    best_tick, best_colour = None, None
    for i, colour in enumerate(colours):
        up = cube_xyz[:, i, 2] - cube_xyz[0, i, 2] > threshold_m
        for tick in range(0, len(up) - sustain + 1):
            if up[tick] and up[tick:tick + sustain].all():
                if best_tick is None or tick < best_tick:
                    best_tick, best_colour = tick, colour
                break
    return best_colour


def _pick_lift(trace, pick):
    colours = [str(c) for c in np.asarray(trace['colours'])]
    z = np.asarray(trace['cube_xyz'])[:, colours.index(pick), 2]
    return z - z[0]


def classify(trace, pick, success):
    """First cause of a trial: success, wrong_cube, never_lifted, released_off_target, lifted_not_carried."""
    if success:
        return 'success'
    lifted = first_lifted(trace)
    if lifted is not None and lifted != pick:
        return 'wrong_cube'
    lift = _pick_lift(trace, pick)
    if lift.size == 0 or lift.max() < LIFT_M:
        return 'never_lifted'
    first = int(np.argmax(lift > LIFT_M))
    command = np.asarray(trace['applied_action'])[:, 7]
    if np.any(command[first + 1:] == GRIPPER_OPEN):
        return 'released_off_target'
    return 'lifted_not_carried'


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


def _contrast(rows, n=10_000, seed=0):
    """distractor minus control, per layout, over layouts that have both conditions."""
    by_layout = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_layout[row['layout_id']][row['condition']].append(float(row['task_success']))
    diffs = {lid: float(np.mean(c['distractor']) - np.mean(c['control']))
             for lid, c in by_layout.items() if c.get('control') and c.get('distractor')}
    unpaired = sorted(set(by_layout) - set(diffs))
    if not diffs:
        return dict(layouts=0, mean_difference=None, interval=None, excluded_unpaired_layouts=unpaired)
    values = np.array(list(diffs.values()))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return dict(layouts=len(diffs), mean_difference=float(values.mean()),
                interval=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                excluded_unpaired_layouts=unpaired)


def readouts(rows, excluded_layouts):
    """Every read-out fixed by spec section 4, after dropping layouts the expert cannot solve."""
    excluded = set(excluded_layouts)
    kept = [r for r in rows if r['layout_id'] not in excluded]
    success = lambda r: r['task_success']
    wrong = lambda r: r['cause'] == 'wrong_cube'
    by = lambda field, values: {v: _rate([r for r in kept if r[field] == v], success) for v in values}
    interior = [r for r in kept if r['edge'] == 'interior']
    edge = [r for r in kept if r['edge'] != 'interior']
    return dict(
        dropped_infeasible_trials=len(rows) - len(kept), excluded_layouts=sorted(excluded),
        overall=_rate(kept, success),
        by_kind=by('kind', ('trained', 'held_out')),
        by_condition=by('condition', ('control', 'distractor')),
        distractor_minus_control=_contrast(kept),
        by_edge={**by('edge', ('interior', 'inner', 'outer')), 'edge_any': _rate(edge, success),
                 'interior_only': _rate(interior, success)},
        wrong_cube=dict(overall=_rate(kept, wrong),
                        by_kind={k: _rate([r for r in kept if r['kind'] == k], wrong)
                                 for k in ('trained', 'held_out')}),
        causes={cause: dict(Counter(r['cause'] for r in kept))[cause] if any(r['cause'] == cause for r in kept) else 0
                for cause in CAUSES},
    )
