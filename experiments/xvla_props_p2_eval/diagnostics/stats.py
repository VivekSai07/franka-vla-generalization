"""Layout-level statistics for the P2 failure diagnostics. Pure numpy (the admission image has no scipy)."""
from itertools import combinations
from math import comb

import numpy as np


def concentration(failures_by_layout):
    """Sum of squared per-layout failure counts: large when failures pile into few layouts."""
    return int(sum(int(v) ** 2 for v in failures_by_layout.values()))


def concentration_p(failed, n=10_000, seed=0):
    """One-sided permutation p for failure concentration.

    `failed` maps layout -> list of per-trial failure flags. The failure labels are permuted over all trials with each
    layout keeping its trial count, so the layout is the unit being tested, not the trial.
    """
    layouts = sorted(failed)
    sizes = [len(failed[k]) for k in layouts]
    flat = np.array([bool(x) for k in layouts for x in failed[k]])
    total = int(flat.sum())
    if total == 0:
        return 1.0
    edges = np.cumsum([0] + sizes)
    observed = sum(int(flat[a:b].sum()) ** 2 for a, b in zip(edges[:-1], edges[1:]))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        shuffled = rng.permutation(flat)
        if sum(int(shuffled[a:b].sum()) ** 2 for a, b in zip(edges[:-1], edges[1:])) >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


EXACT_LIMIT = 200_000


def _average_ranks(values):
    order = values.argsort(kind='mergesort')
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def mann_whitney_p(pos, neg, n=20_000, seed=0):
    """Two-sided permutation p for the Mann-Whitney U statistic (ties get average ranks).

    Exact (every split of the pooled ranks enumerated, no seed) when C(N, k) <= EXACT_LIMIT, else Monte Carlo with `n`
    permutations from `seed`.
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('Both groups need at least one value')
    ranks = _average_ranks(np.concatenate([pos, neg]))
    k, total = len(pos), len(pos) + len(neg)
    expected = k * (total + 1) / 2
    observed = abs(ranks[:k].sum() - expected)
    if comb(total, k) <= EXACT_LIMIT:
        hits = count = 0
        for subset in combinations(range(total), k):
            count += 1
            if abs(ranks[list(subset)].sum() - expected) >= observed - 1e-9:
                hits += 1
        return hits / count
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        if abs(ranks[rng.permutation(total)[:k]].sum() - expected) >= observed - 1e-9:
            hits += 1
    return (hits + 1) / (n + 1)
