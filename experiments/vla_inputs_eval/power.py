"""Planning power of the primary test (exact two-sided McNemar, one rollout per layout), recomputed at the feasible N.

D is the discordance rate (layouts where exactly one arm succeeds), delta the true improvement in success rate. Exact
enumeration: M ~ Bin(N, D) discordant layouts; given M, the new-only count B ~ Bin(M, (D + delta) / (2 D)); the test rejects when
B > M / 2 and the exact two-sided p < alpha. Assumptions that matter: layouts independent, one fixed (D, delta) for every
layout (no per-object heterogeneity), the test is applied once (alpha .0125 is the conservative first Holm threshold of
four, not exact joint Holm power), and nothing here knows any policy outcome. Pure numpy; admission image.
"""
import argparse
import json
from math import lgamma

import numpy as np

DISCORDANCES = (0.19, 0.30, 0.40)
DELTAS = (0.06, 0.08, 0.10, 0.15)
ALPHAS = (0.05, 0.0125)


def _log_choose(n, k, table):
    return table[n] - table[k] - table[n - k]


def _pmf(n, p, table):
    """Binomial(n, p) pmf over 0..n, by logs."""
    k = np.arange(n + 1)
    if p <= 0.0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1.0:
        out = np.zeros(n + 1)
        out[n] = 1.0
        return out
    return np.exp(_log_choose(n, k, table) + k * np.log(p) + (n - k) * np.log1p(-p))


def exact_power(n, discordance, delta, alpha):
    """Power of the exact McNemar test at `n` layouts."""
    q = (discordance + delta) / (2 * discordance)
    if not 0.0 <= q <= 1.0:
        raise ValueError("delta must satisfy |delta| <= discordance")
    table = np.array([lgamma(i + 1) for i in range(n + 2)])
    weights = _pmf(n, discordance, table)
    total = 0.0
    for m in range(1, n + 1):
        half = np.cumsum(_pmf(m, 0.5, table))
        b = np.arange(m // 2 + 1, m + 1)  # B > M / 2
        p_value = np.minimum(1.0, 2 * half[m - b])
        reject = b[p_value < alpha]
        if len(reject):
            total += weights[m] * _pmf(m, q, table)[reject].sum()
    return float(total)


def monte_carlo_power(n, discordance, delta, alpha, reps=20000, seed=0):
    """(power, standard error) by simulating layouts; a cross-check of `exact_power`."""
    rng = np.random.default_rng(seed)
    p01, p10 = (discordance + delta) / 2, (discordance - delta) / 2
    u = rng.random((reps, n))
    wins, losses = (u < p01).sum(1), ((u >= p01) & (u < p01 + p10)).sum(1)
    table = np.array([lgamma(i + 1) for i in range(n + 2)])
    cache = {}

    def rejects(b, m):
        if (b, m) not in cache:
            cache[(b, m)] = b > m - b and min(1.0, 2 * np.cumsum(_pmf(m, 0.5, table))[m - b]) < alpha
        return cache[(b, m)]

    hits = np.array([rejects(int(b), int(b + c)) if b + c else False for b, c in zip(wins, losses)])
    power = float(hits.mean())
    return power, float(np.sqrt(power * (1 - power) / reps))


def table(n, alphas=ALPHAS, discordances=DISCORDANCES, deltas=DELTAS):
    return {f"alpha={a}": {f"delta={d}": {f"D={x}": round(exact_power(n, x, d, a), 3) for x in discordances} for d in deltas}
            for a in alphas}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, required=True, help="the feasible comparison-tier size")
    parser.add_argument("--mc-reps", type=int, default=20000)
    args = parser.parse_args(argv)
    cross = {}
    for a in ALPHAS:
        exact = exact_power(args.layouts, 0.30, 0.10, a)
        mc, se = monte_carlo_power(args.layouts, 0.30, 0.10, a, reps=args.mc_reps)
        cross[f"alpha={a}"] = dict(exact=round(exact, 4), monte_carlo=round(mc, 4), mc_standard_error=round(se, 4), reps=args.mc_reps)
    print(json.dumps(dict(layouts=args.layouts, exact=table(args.layouts), monte_carlo_check_D0_30_delta_0_10=cross,
                          assumptions="one fixed (D, delta) for all layouts; independent layouts; one test; alpha .0125 is the first "
                                      "of four Holm thresholds, not joint power; no policy outcome used"), indent=1))


if __name__ == "__main__":
    main()
