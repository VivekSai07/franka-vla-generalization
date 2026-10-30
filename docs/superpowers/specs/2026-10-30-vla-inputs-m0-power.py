"""M0 power: paired McNemar (exact, two-sided, alpha 0.05), one rollout per layout, arm against control.

Run in the admission image (numpy only):
  docker run --rm -i --network none --entrypoint python <admission digest> - < this file

D is the discordance rate (fraction of layouts where exactly one arm succeeds); delta the true improvement.
D = 0.19 is what two checkpoints of the same P2 run show on the 120 identical P2 report trials (15 + 8 of 120);
arms with different inputs or models are expected to disagree more, so 0.30 and 0.40 are the planning values.
"""
import math

import numpy as np

REPS, LAYOUTS = 4000, (60, 120, 180, 240, 300, 384, 480)
rng = np.random.default_rng(0)
cache = {}


def significant(b, m):
    if (b, m) not in cache:
        k = min(b, m - b)
        cache[(b, m)] = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) < 0.05
    return cache[(b, m)]


def power(n, p01, p10):
    u = rng.random((REPS, n))
    wins, losses = (u < p01).sum(1), ((u >= p01) & (u < p01 + p10)).sum(1)
    return float(np.mean([b > c and significant(int(b), int(b + c)) for b, c in zip(wins, losses)]))


for discordance in (0.19, 0.30, 0.40):
    for delta in (0.06, 0.08, 0.10, 0.15):
        p01, p10 = (discordance + delta) / 2, (discordance - delta) / 2
        row = ", ".join(f"{n}: {power(n, p01, p10):.2f}" for n in LAYOUTS)
        print(f"D={discordance:.2f} delta={delta:.2f}  {row}")
