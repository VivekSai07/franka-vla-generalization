# Props P1 X-VLA evaluation on the reserved layouts — 2026-10-09

A dated report; do not edit it to reflect later findings. Measured, not sealed: the evaluation
package (`experiments/xvla_props_p1_eval/`) has no contract. Robosuite image
(`sha256:54d861aa…ccf27e`, robosuite 1.5.2 / MuJoCo 3.4.0), GPU for rollouts. Training:
[docs/PROPS_P1_TRAINING_20261009.md](PROPS_P1_TRAINING_20261009.md).

## Protocol deviation from the original plan (user-directed, stated here plainly)

The plan ([docs/superpowers/plans/2026-10-07-props-p1-training-eval.md](superpowers/plans/2026-10-07-props-p1-training-eval.md))
specified: screen all 6 non-trivial checkpoints, carry only those within 2 successes of the best to
the report, confirm the report's winner once. **Screen ran exactly as planned.** The tie rule
selected a single survivor (88,830, 18/20); no other checkpoint was within 2 successes of it.

The user then asked to run the report tier on all 6 checkpoints, not just the survivor, given the
20-trial screen counts are noisy. This is flagged because it reproduces the shape of a known trap:
B3's own evaluation ([docs/B3_XVLA_EVALUATION_20261005.md](B3_XVLA_EVALUATION_20261005.md)) notes
that letting the report tier both select and measure the winner was overly optimistic. **Before any
of the 5 extra report runs started or produced a trial**, the confirm-selection rule was fixed and
written down: *confirm runs on whichever of the 6 checkpoints has the highest report-tier overall
success rate, ties broken toward the later checkpoint.* Confirm itself is still touched exactly
once, on a rule fixed before the data existed — only the **candidate pool** widened from "tie-rule
survivors" (1) to "all 6". Numbers below are reported as actually run; this deviation is the reason
all 6 have report numbers instead of just one.

## Feasibility

All 80 reserved layouts (20 screen + 40 report + 20 confirm) solved by the scripted expert before
any policy rollout: **80/80, 0 excluded.** `outputs/xvla_props_p1_eval/feasibility/feasibility.json`.

## Screen (20 layouts, one seed each, all 6 checkpoints)

| checkpoint (epoch) | success | causes |
|---|---|---|
| 50,760 (2.0) | 7/20 (35%) | timeout_carrying 9, success 7, never_lifted 4 |
| 76,140 (3.0) | 3/20 (15%) | never_lifted 7, timeout_carrying 10, success 3 |
| **88,830 (3.5)** | **18/20 (90%)** | success 18, never_lifted 2 |
| 101,520 (4.0) | 14/20 (70%) | success 14, never_lifted 3, dropped_outside 1, timeout_carrying 2 |
| 114,210 (4.5) | 15/20 (75%) | success 15, timeout_carrying 3, dropped_outside 1, never_lifted 1 |
| 126,900 (5.0) | 14/20 (70%) | success 14, never_lifted 2, timeout_carrying 3, unstable_in_target 1 |

Tie rule (within 2 of the best, 18): only 88,830 qualifies — a clear single winner, not a tie.

## Report (40 layouts x 2 seeds = 80 trials, all 6 checkpoints)

| checkpoint | overall | 95% CI (layout-clustered) | trained | held-out | blue_slot L | blue_slot R | wrong_bin |
|---|---|---|---|---|---|---|---|
| 50,760 | 51.2% (41/80) | [37.5, 66.3] | 42.5% | 60.0% | 40.0% | 62.5% | 0/80 |
| 76,140 | 43.8% (35/80) | [30.0, 57.5] | 42.5% | 45.0% | 42.5% | 45.0% | 0/80 |
| 88,830 | 75.0% (60/80) | [63.8, 85.0] | 70.0% | 80.0% | 70.0% | 80.0% | 0/80 |
| 101,520 | 62.5% (50/80) | [47.5, 76.3] | 62.5% | 62.5% | 70.0% | 55.0% | 0/80 |
| 114,210 | 70.0% (56/80) | [56.3, 82.5] | 65.0% | 75.0% | 75.0% | 65.0% | 0/80 |
| **126,900** | **75.0% (60/80)** | [62.5, 86.3] | 72.5% | 77.5% | 85.0% | 65.0% | 0/80 |

`dropped_infeasible_trials: 0` for all six. **Zero `wrong_bin` outcomes across all 480 report
trials and all six checkpoints** — every failure is a motor/timing problem (never lifted, timed out
carrying, dropped outside, or correct-bin-but-unstable), never a colour confusion. Held-out pairs
perform at or above trained pairs on every checkpoint (no held-out collapse at any point in
training).

**Confirm-rule outcome:** 88,830 and 126,900 are **exactly tied at 75.0% (60/80)**. Per the rule
fixed before this data existed, the tie breaks toward the later checkpoint: **confirm runs on
126,900.**

One caveat on that pick, independent of the procedural tie-break: 126,900's arrangement split is
wider (blue-left 85% vs blue-right 65%, a 20-point gap) than 88,830's (70% vs 80%, a 10-point gap).
The spec's own interpretation guide treats "within a few points" as the colour-reading signature; a
20-point gap is on the edge of what that guide would call clean. Recorded as a caveat on the
confirm winner, not a reason to override the pre-committed rule.

## Confirm (20 trained-cell layouts, touched once, checkpoint 126,900)

**20/20 = 100%.** [1.0, 1.0] (degenerate interval at n=20 layouts, all succeeding). Both bin
arrangements perfect (left 10/10, right 10/10). Zero wrong-bin, zero of any failure cause.

**This is the headline number: 100% (20/20) on the confirm tier, checkpoint 126,900.** Read
alongside the report tier's 75% (60/80) on the same checkpoint — the confirm layouts are a
different, previously-untouched set of 20 trained-cell draws, smaller and with a degenerate (not
informative) interval; the report's 75% with its tighter-but-still-wide CI [62.5, 86.3] is the more
statistically meaningful estimate of this checkpoint's trained-cell success rate. Confirm's role
here is a clean, protocol-correct, one-shot check on fresh layouts — it is not a second, independent
80-trial estimate.

## A classifier bug, found by inspection, fixed before these numbers were final

While reviewing screen videos, one `screen_015` trial (checkpoint 126,900, bottle→orange) was
labelled `dropped_outside` despite the bottle resting correctly in the target bin (inside the
margin, below the wall height, released). `analysis.classify()` checked `final_gripper_open` before
`placed_bin`, so a correct-bin rest that never sustained the 30-consecutive-tick success streak
(finger contact cleared only ~22 ticks before the 900-tick cap) fell through to the catch-all case.
Fixed (`5a3f21a`): check `placed_bin` first, add `unstable_in_target` as its own cause. A regression
test was written and confirmed to fail against the pre-fix code before fixing. All 120 screen trials
were scanned for the same pattern: **exactly one was affected**, matching what was spotted by eye.
Reclassified from the already-saved `trace.npz` files — no rollouts were re-run, and `task_success`
itself (the real-time pass/fail from `success_streak`) was never wrong, only the descriptive label.

## The 900-tick cap: evidence gathered, not changed

Reviewing videos also surfaced trials that appeared to carry the object above the bin before timing
out. Every `timeout_carrying` trial's final state was checked (`xy_margin_m`, `lowest_z_m`): the
picture is mixed — some are plausibly near-misses (margin -0.06 to -0.09, still airborne ~0.10 m),
several are clearly far off-target (margin -0.23 to -0.39) or at table height (~0.001 m, consistent
with an early drop, not "almost there"). `MAX_TICKS = 900` is a fixed global constraint from the
plan, agreed before any rollout; screen already ran all 6 checkpoints under it. The evidence is not
clean enough to justify moving a fixed threshold mid-protocol, so it was left unchanged. **Recorded
here as a limitation and a candidate follow-up**: a separately labelled, explicitly comparative run
with a larger cap, never a silent retroactive change to this evaluation.

## Reading (spec interpretation, fixed in advance)

Held-out within 10 points of trained (report: 77.5% vs 72.5%, a 5-point gap, well within the 10-point
reading), wrong-bin rate 0% (under the 5% reading), both arrangements within a few points on 88,830
(10-point gap) but wider on 126,900 (20-point gap) — reads as *the policy follows the colour word*
on the weight of the held-out and wrong-bin evidence, with the arrangement gap on the confirmed
checkpoint flagged as the one piece of evidence that is not fully clean.

## Decision

Props P1 closed: training complete and verified, evaluation complete across screen/report/confirm
with one documented protocol deviation and one fixed classifier bug. Headline: **100% (20/20) on
the confirm tier, checkpoint 126,900**; **75% (60/80) report-tier estimate on the same checkpoint**
is the more informative number for trained-cell success going forward.
