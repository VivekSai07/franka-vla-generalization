"""M0 statistics, fixed before any rollout. Pure numpy; admission image.

Primary test per arm against A0: exact two-sided McNemar on the paired comparison tier; effect = mean paired difference
with a 95% percentile bootstrap over layouts stratified by object; Holm over the fixed A1-A4 family (unrun arms enter as
p = 1). Selection: most successes on the selection tier, a tie goes to the later checkpoint (an administrative rule, not
evidence that the later checkpoint is better; `tied` records every tie).
"""
from collections import defaultdict
from math import comb

import numpy as np

FAMILY = ("A1", "A2", "A3", "A4")


def mcnemar(b, c):
    """Exact two-sided p for discordant pair counts `b` (new only) and `c` (control only)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)


def _by_object(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["object"]].append(int(bool(row["new"])) - int(bool(row["control"])))
    return {k: np.asarray(v, dtype=float) for k, v in sorted(groups.items())}


def _indices(groups, n, rng):
    """For each object, `n` bootstrap index draws of its own size: resampling never moves a layout between objects."""
    return {obj: rng.integers(0, len(d), size=(n, len(d))) for obj, d in groups.items()}


def bootstrap_sample(rows, rng):
    """One stratified resample of `rows` (same count of layouts for every object)."""
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object"]].append(row)
    draws = _indices({o: np.zeros(len(r)) for o, r in sorted(by_object.items())}, 1, rng)
    return [by_object[o][i] for o in sorted(by_object) for i in draws[o][0]]


def paired_effect(rows, n=10_000, seed=0):
    """{"estimate", "lo", "hi"}: mean of (new - control) over layouts, 95% percentile interval, stratified by object."""
    groups = _by_object(rows)
    total = sum(len(d) for d in groups.values())
    rng = np.random.default_rng(seed)
    draws = _indices(groups, n, rng)
    means = sum(d[draws[o]].sum(axis=1) for o, d in groups.items()) / total
    return {"estimate": float(sum(d.sum() for d in groups.values()) / total),
            "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def holm(pvalues):
    """Holm step-down adjusted p-values for a dict {name: p}."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, adjusted = len(order), 0.0, {}
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def family_holm(pvalues):
    """Holm over the fixed family A1-A4; arms without a result enter as p = 1."""
    if set(pvalues) - set(FAMILY):
        raise ValueError(f"arms outside the fixed family {FAMILY}: {sorted(set(pvalues) - set(FAMILY))}")
    return holm({arm: pvalues.get(arm, 1.0) for arm in FAMILY})


def tied(successes_by_checkpoint):
    """Every checkpoint with the best success count, ascending."""
    if not successes_by_checkpoint:
        raise ValueError("no candidates")
    best = max(successes_by_checkpoint.values())
    return sorted(c for c, s in successes_by_checkpoint.items() if s == best)


def select(successes_by_checkpoint):
    """Most successes; a tie goes to the later checkpoint."""
    return tied(successes_by_checkpoint)[-1]
