"""G0 statistics, fixed before any rollout. Pure numpy; admission image.

Every comparison is paired by layout: the same layout under nominal conditions and under one axis's level, one trial each. Per arm,
axis and level: exact two-sided McNemar on the discordant pairs, and the paired DROP (nominal minus perturbed success) with a 95%
percentile bootstrap over layouts stratified by object (10,000 resamples, seed 0). An axis is HELD by an arm iff the upper bound
of the drop's interval is at most 10 points (pre-registered, spec D13; later G1's target). Holm over the five axes is reported
(a measurement, not a family-wise claim). Selection (for a later arm's checkpoint, never for G0 claims): most successes over every
trial of the selection tier; a tie goes to the later checkpoint (administrative, not evidence).

`mcnemar`, `holm`, `tied`, `select` and the bootstrap are forked from `experiments/vla_inputs_eval/stats.py` (frozen, unchanged).
"""
from collections import defaultdict
from math import comb

import numpy as np

HELD_MAX_DROP = 0.10
BOOTSTRAP_N, BOOTSTRAP_SEED = 10_000, 0


def mcnemar(b, c):
    """Exact two-sided p for discordant pair counts `b` and `c`."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)


def _by_object(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["object"]].append(int(bool(row["nominal"])) - int(bool(row["perturbed"])))
    return {k: np.asarray(v, dtype=float) for k, v in sorted(groups.items())}


def paired_drop(rows, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """{"estimate", "lo", "hi"}: mean over layouts of (nominal - perturbed), 95% percentile interval, resampling layouts within
    each object so a resample never moves a layout between objects. A positive drop is a loss."""
    groups = _by_object(rows)
    if not groups:
        return {"estimate": None, "lo": None, "hi": None}
    total = sum(len(d) for d in groups.values())
    rng = np.random.default_rng(seed)
    draws = {obj: rng.integers(0, len(d), size=(n, len(d))) for obj, d in groups.items()}
    means = sum(d[draws[o]].sum(axis=1) for o, d in groups.items()) / total
    return {"estimate": float(sum(d.sum() for d in groups.values()) / total),
            "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def held(drop):
    """The pre-registered rule: held iff the drop's 95% interval upper bound is at most 10 points. None for an empty cell."""
    if drop["hi"] is None:
        return None
    return bool(drop["hi"] <= HELD_MAX_DROP)


def pair_counts(rows):
    """(both, nominal_only, perturbed_only, neither) over paired rows."""
    both = sum(1 for r in rows if r["nominal"] and r["perturbed"])
    c = sum(1 for r in rows if r["nominal"] and not r["perturbed"])
    b = sum(1 for r in rows if r["perturbed"] and not r["nominal"])
    return both, c, b, len(rows) - both - c - b


def holm(pvalues):
    """Holm step-down adjusted p-values for a dict {name: p}."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, adjusted = len(order), 0.0, {}
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def tied(successes_by_checkpoint):
    """Every checkpoint with the best success count, ascending."""
    if not successes_by_checkpoint:
        raise ValueError("no candidates")
    best = max(successes_by_checkpoint.values())
    return sorted(c for c, s in successes_by_checkpoint.items() if s == best)


def select(successes_by_checkpoint):
    """Most successes; a tie goes to the later checkpoint."""
    return tied(successes_by_checkpoint)[-1]
