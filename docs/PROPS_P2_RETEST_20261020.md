# Props P2 fresh-layout re-test: no large unseen-pair penalty on the same objects — 2026-10-20

Checkpoint 101,520 (from the archive bundle) on 240 fresh paired layouts of milk and bread, each run under its trained and its
held-out bin instruction. Design and the pre-registered reading (fixed before any rollout):
[spec](superpowers/specs/2026-10-19-props-p2-retest-design.md), [plan](superpowers/plans/2026-10-19-props-p2-retest.md).
Package `experiments/xvla_props_p2_retest/`; raw results and `analysis.json` (gitignored) in `outputs/xvla_props_p2_retest/`.

## Result

**Pooled unseen-pair penalty (trained minus held-out success, same layouts): 2.5 points, 95% interval [−4.2, +9.2]**
(238 pairs, layout-clustered bootstrap stratified by object, 10,000 resamples, seed 0). By the pre-registered reading, the upper
bound is below 10 points and the lower bound is below 0, so: **the penalty is small, and its existence is not shown.**
That is not "no penalty": a true penalty of up to about 9 points is compatible with these data.

| | trained | held-out | penalty [95%] |
|---|---|---|---|
| milk (`milk→orange` / `milk→blue`), 118 pairs | 68/118 = 57.6% | 67/118 = 56.8% | 0.8 [−11.0, 12.7] |
| bread (`bread→blue` / `bread→orange`), 120 pairs | 97/120 = 80.8% | 92/120 = 76.7% | 4.2 [−2.5, 11.7] |
| **all, 238 pairs** | 165/238 = 69.3% | 159/238 = 66.8% | **2.5 [−4.2, 9.2]** |

The per-object rows are descriptive only (pre-registered: the milk-versus-bread contrast cannot be resolved at this size;
power for an 8-point difference was about 15%). Pairs: both succeed 128, only trained 37, only held-out 31, neither 42; the
68 discordant pairs split almost evenly. By blue-bin side the penalty is 2.5 points either way (118 left, 120 right); by distractor
count 1.7 (one) and 3.4 (two). Failure causes: trained arm 165 success, 32 never lifted, 24 timed out carrying, 7 wrong object,
7 dropped outside, 3 unstable; held-out arm 159 success, 34 never lifted, 26 timed out carrying, 8 dropped outside, 7 wrong
object, 3 unstable, **1 wrong bin**.

## What it means for the P2 open question

P2's criterion 3 (trained-versus-held-out gap ≤ 10 points) failed with a 30-point gap (trained 85.0%, held-out 55.0%). That
comparison mixed objects: the trained cells include easy objects (can 28/28) and the held-out cells are milk and bread. On the
same two objects and the same layouts the gap is 2.5 points (upper bound 9.2). **The data are consistent with most of the 30
points being which objects the cells contain, not that the pairs were unseen.** The pre-registered criterion is unchanged and
stays failed; it is not reinterpreted.

The earlier cell rates quoted in the diagnostics and the spec (milk 7/14 trained and 7/20 held-out; bread 12/14 and 15/20; 10
layouts per cell) were small-sample noise: on 118 and 120 fresh layouts milk is 0.58 and 0.57. Milk is still the weak object
(57% against bread's 79% overall) in both bins.

## Execution

- 480 entries drawn with P2's own samplers under a `retest:` identity namespace (tests: no pose, expert seed, home jitter or
  layout id coincides with any P2 episode or reserved layout; both arms share all geometry). Plan sha256 `35d34397…fa8d`.
- **Expert feasibility 476 of 480.** The expert fails (`grasp`, 182 frames) on both arms of `retest_milk_013` and `retest_milk_113`,
  with starts at radius 0.286 m and 0.290 m, at the inner edge of the start region where P0 recorded its own reach failures. Per
  the spec those two pairs are dropped as pairs: 238 pairs analysed, all 480 trials were still run.
- Checkpoint weights verified through the bundle route (`model.safetensors` sha256 `59d02d0a…932e` equals its marker; seal
  fingerprint `b1d642b3…f4d898` recomputed over the bundle). Rollout loop, scoring and failure classifier are the P2 evaluator's, unchanged.
- 8 workers pinned to 5 cores each on one GPU, 480 of 480 trials complete, no worker failure, median 68 s per trial (min 34,
  max 145). OSMesa rendering, as the checkpoint was trained.
- **Deviation:** the evaluator's `load_feasibility` hard-codes "120 reserved layouts", which cannot hold for this plan. It was not
  edited; `retest.load_feasibility` checks the summary against the re-test plan's own layouts (tests, with mutation checks).

## What this does not show

- Whether any residual penalty (up to about 9 points) is "unseen pair" or bin colour or side; balance cancels side and colour on
  average, it does not isolate them.
- Anything about the other four objects, the other eight cells, or other checkpoints (126,900 was not run).
- A difference between milk and bread: the per-object intervals overlap and cannot resolve one.
- The P2 diagnostics' account of why milk fails (at the hold, not the approach) is untouched by this result.

## Process notes

The design came from a power analysis done before any rollout. Using P2's four cell rates and the measured within-layout
correlation of two seeds (ICC 0.56), the milk-versus-bread contrast first proposed was shown to be unresolvable at any feasible
size, so the primary contrast was changed (with the user) to the pooled paired penalty, and one seed on 120 layouts per object
replaced two seeds on 60 (standard error 3.4 against 4.3 points for the same 480 rollouts). A first version of that simulation
mis-calibrated the marginal rates and was discarded before use.
