"""Paired analysis of the P2 re-test: the pooled unseen-pair penalty and its pre-registered reading. Pure; numpy only.

A *pair* is one fresh layout run under both bin instructions (arms `trained` and `held`); `d = trained - held` is in
{-1, 0, +1}. Layouts, i.e. pairs, are the independent unit. Rules: docs/superpowers/specs/2026-10-19-props-p2-retest-design.md.
"""
import collections

import numpy as np

ARMS = ("trained", "held")
SMALL_BELOW = 0.10         # pre-registered: a penalty is "small" when the upper bound is below 10 points
INTERVAL = (2.5, 97.5)


def _split(layout_id):
    pair_id, _, arm = layout_id.rpartition("_")
    if arm not in ARMS:
        raise ValueError(f"{layout_id!r}: last field is {arm!r}, expected one of {ARMS}")
    return pair_id, arm


def pairs_from_rows(rows, excluded=()):
    """([pair dicts sorted by pair id], [dropped pair ids]). A pair with an excluded arm is dropped whole; a pair with a
    missing or duplicated arm among the rest is an error, never a smaller sample."""
    excluded_pairs = {_split(layout_id)[0] for layout_id in excluded}
    grouped = collections.defaultdict(dict)
    for r in rows:
        pair_id, arm = _split(r["layout_id"])
        if pair_id in excluded_pairs:
            continue
        if arm in grouped[pair_id]:
            raise ValueError(f"{pair_id}: arm {arm} appears twice")
        grouped[pair_id][arm] = r
    pairs = []
    for pair_id in sorted(grouped):
        arms = grouped[pair_id]
        if set(arms) != set(ARMS):
            raise ValueError(f"{pair_id}: has arms {sorted(arms)}, needs {list(ARMS)}")
        t, h = arms["trained"], arms["held"]
        pairs.append(dict(pair_id=pair_id, object=t["object"], trained=bool(t["task_success"]), held=bool(h["task_success"]),
                          d=int(bool(t["task_success"])) - int(bool(h["task_success"])), blue_slot=t["blue_slot"],
                          distractor_count=t["distractor_count"], causes=dict(trained=t["cause"], held=h["cause"])))
    return pairs, sorted(excluded_pairs & {_split(r["layout_id"])[0] for r in rows})


def penalty(pairs, n=10_000, seed=0):
    """Pooled mean of d over all pairs with a 95% percentile bootstrap over pairs, resampled within each object."""
    by_object = collections.defaultdict(list)
    for p in pairs:
        by_object[p["object"]].append(p["d"])
    total = sum(len(v) for v in by_object.values())
    rng = np.random.default_rng(seed)
    sums = np.zeros(n)
    for obj in sorted(by_object):
        d = np.asarray(by_object[obj], dtype=float)
        sums += d[rng.integers(0, len(d), size=(n, len(d)))].sum(axis=1)
    lo, hi = np.percentile(sums / total, INTERVAL)
    return dict(estimate=float(sum(sum(v) for v in by_object.values()) / total), lo=float(lo), hi=float(hi), n_pairs=total,
                per_object={obj: dict(n=len(v), estimate=float(np.mean(v))) for obj, v in sorted(by_object.items())})


def reading(lo, hi):
    """The pre-registered reading of the interval: the penalty exists if it excludes 0 from below, is small if its upper
    bound is below 10 points. Both bounds are strict."""
    exists, small = lo > 0, hi < SMALL_BELOW
    return {(True, True): "exists_and_small", (True, False): "exists", (False, True): "small",
            (False, False): "inconclusive"}[(exists, small)]


def pair_table(pairs):
    def cell(ps):
        return dict(both=sum(p["trained"] and p["held"] for p in ps), only_trained=sum(p["trained"] and not p["held"] for p in ps),
                    only_held=sum(p["held"] and not p["trained"] for p in ps), neither=sum(not p["trained"] and not p["held"] for p in ps))
    out = {obj: cell([p for p in pairs if p["object"] == obj]) for obj in sorted({p["object"] for p in pairs})}
    out["all"] = cell(pairs)
    return out


def penalty_by(pairs):
    def group(key):
        groups = collections.defaultdict(list)
        for p in pairs:
            groups[str(p[key])].append(p["d"])
        return {k: dict(n=len(v), estimate=float(np.mean(v))) for k, v in sorted(groups.items())}
    return dict(blue_slot=group("blue_slot"), distractor_count=group("distractor_count"))


def causes(pairs):
    return {arm: dict(collections.Counter(p["causes"][arm] for p in pairs)) for arm in ARMS}
