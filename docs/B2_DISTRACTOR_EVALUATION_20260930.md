# B2 distractor evaluation — 2026-09-30

Evaluation of the X-VLA model trained on the sealed B2 distractor dataset
(`b2_distractor_20260928`, seal `b6cfff3f…0c12340`; training run
`xvla_franka_stacking_b2_distractors_v1_20260928`, 50,000 updates). Covers the headline numbers,
a full diagnosis of every failed rollout, and what it suggests for future work. Reproduction:
[`experiments/xvla_b2_distractors_eval/analysis/`](../experiments/xvla_b2_distractors_eval/analysis/README.md).
Per-failure table: [B2_DISTRACTOR_FAILURES_20260930.csv](B2_DISTRACTOR_FAILURES_20260930.csv).

## 1. Headline

Report protocol: 40 held-out layouts × 2 conditions × 2 seeds = 160 trials per checkpoint. Layouts
are disjoint from the 20 screening layouts. **Layout is the independent unit** (n = 40), so the
cluster intervals below are the honest ones.

| checkpoint | overall | 95% CI (Wilson) | 95% CI (layout-cluster) | control | distractor |
|---|---|---|---|---|---|
| 33,400 | **132/160 = 82.5%** | [75.9, 87.6] | [73.1, 90.6] | 67/80 = 83.8% | 65/80 = 81.2% |
| 50,000 | **147/160 = 91.9%** | [86.6, 95.2] | [83.8, 98.1] | 74/80 = 92.5% | 73/80 = 91.2% |

**1. Distractors had no detectable effect.** Within-layout paired contrast (control − distractor
success, per layout): +2.5 pts at 33,400 (95% CI [−7.5, +12.5], permutation p = 0.82; 5 layouts
better on control, 4 worse, 31 tied) and +1.25 pts at 50,000 (CI [−5.0, +8.8], p = 1.0; 2 / 2 / 36).
The design spec names a null result as a valid finding. The interval rules out a drop larger than
about 9–12 points; it does not rule out a small one.

**2. More training helped, and the screening tie-break chose the wrong checkpoint.** Screening
(20 layouts, distractor condition) scored 8 / 16 / 17 / 16 / 17 of 20 for 16.7k / 25k / 33.4k /
41.7k / 50k. 33,400 and 50,000 **tied at 17/20**; the chain script takes the first maximum in array
order, so it reported 33,400. The report reversed the tie: paired on the same trial ordinal (same
layout, condition and rollout seed), 127 trials succeed at both, 5 only at 33,400, **20 only at
50,000**, 8 at neither (McNemar exact p = 0.0041; 10 vs 3 for control, 10 vs 2 for distractor). An
unqueued 50,000 report would have published 82.5% instead of 91.9%. See FAILURES #37.

**3. Control arm vs B1.** 50,000's control arm (92.5%) is in the range of B1's 94.4% (151/160).
Different layouts, different checkpoint selection and a different LR phase (the cosine is stretched
over 50,000 updates), so this is a sanity check and not a delta.

**4. Statistical caveat on headlining 50,000.** Both checkpoints were reported after the tie, and
91.9% is the larger. The paired difference is strong enough that selection is an unlikely
explanation, but a fresh holdout is the clean confirmation.

## 2. What "timeout" means here

All 41 failures are labelled `outcome: timeout`. That is the only non-success label the protocol
has: a trial ends on a 30-tick stable stack or at the 1,200-tick (40 s) cap. **It is not a
cause.** Only **2 of 41** are genuine time-budget near-misses (§4, M3). Successes finish in a median
375 ticks (max 762 at 33,400; one slow outlier at 1,161 at 50,000), so the budget is not generally
binding.

## 3. Failure overview

41 failures in 320 trials (12.8%): 28 at 33,400, 13 at 50,000.

**Failures are layout-specific, not random.** At 33,400, 12 of 40 layouts had a failure and the top
5 hold 57% of them. At 50,000, **5 layouts hold all 13** and the top 3 hold 77%. Concentration
versus a shuffled-label null: permutation p = 0.0002 at both checkpoints. Four layouts fail at both
checkpoints (`040`, `048`, `054`, `059`) and account for 11 of 28 and 12 of 13 failures. Within a
layout the outcome is usually the same across the four trials (29/40 layouts at 33,400, 36/40 at
50,000).

**Reachability is ruled out.** The reserved layouts were never expert-verified (FAILURES #14), so
the scripted expert was run on them: **23/23 succeeded**, including all 13 layouts the policy
failed on, the distractor variants of the four persistent layouts, and six layouts the policy
always solved. No failing layout is infeasible.

**No distractor was ever touched.** Maximum displacement of any active distractor across all 160
distractor trials at both checkpoints: **0.064 mm** (the measure sees motion: the pick cube moves
0.51 m in a success, and the unused parked cubes, which keep their residual velocity per FAILURES
#32, travel tens of metres). The base cube moved more than 1 cm in only one success. So B2 as
built measured *visual* distraction only; nothing physically interfered.

**Action clipping does not discriminate** (some clipped tick in 46% of failing vs 47% of passing
trials at 33,400; 38% vs 27% at 50,000). Joint-limit margin also does not: joint 4 sits within
~0.01 rad of a limit in the typical trial, successes included.

## 4. Mechanisms (first cause, mutually exclusive, all 41)

Classified from the trace and terminal geometry; thresholds are in
`analysis/12_mechanisms.py`. Counts are small, so read them as proportions, not precise rates.

| mechanism | 33,400 | 50,000 | total | layouts |
|---|---|---|---|---|
| **M4a** release offset ≥ 2 cm → cube set down off-centre, slid or toppled | 11 | 4 | **15** | 037, 059, 040, 027, 054, 029 |
| **M4b** release looked fine (< 2 cm) but the cube slid / ended ≥ 2 cm off | 4 | 2 | **6** | 052, 059, 033, 040 |
| **M2** grasped but never lifted | 6 | 1 | **7** | 048, 024, 025, 056 |
| **M3** too slow: hesitated, ran out of budget | 5 | 2 | **7** | 058, 048, 025, 052, 056 |
| **M1** acquisition failure: never secured the cube | 0 | 4 | **4** | 054 |
| **M5** wrong cube picked first | 2 | 0 | **2** | 054 |

### M4: release precision is the dominant failure (21 of 41, 51%)

Success collapses exactly where the physics says it should. Outcome versus lateral offset of the
carried cube at the last release (303 trials with a carry-and-release; layouts pooled):

| offset at release | n | success |
|---|---|---|
| < 1.0 cm | 186 | 98.4% |
| 1.0–1.5 cm | 68 | 98.5% |
| 1.5–2.0 cm | 32 | 84.4% |
| 2.0–2.5 cm | 9 | **22.2%** |
| 2.5–3.0 cm | 6 | **0%** |
| > 3.0 cm | 2 | **0%** |

The cube edge is 4.2 cm, so its centre of mass leaves the support at **2.1 cm**. The 2.0 cm success
tolerance is essentially the physical tipping limit, not an arbitrary threshold; these are real
toppled stacks and not scoring artefacts. Confirmed on video (`ck50000` ord 157: stacked at
t≈320, gripper opens at 2.6 cm offset, the cube slides off and rests beside the base).

The policy inherits the expert's precision but with a fatter tail:

| | mean release offset | p99 | share > 1.5 cm | share > 2.0 cm |
|---|---|---|---|---|
| expert demonstrations (n = 200) | 0.75 cm | 1.91 | 7.0% | **0.5%** |
| policy 33,400 | 1.03 cm | 2.80 | 20.1% | **8.7%** |
| policy 50,000 | 0.92 cm | 2.44 | 12.3% | **2.6%** |

Training sharpened it, which explains 9 of the 15 fewer failures at 50,000 (M4 fell 15 → 6). Among the 23
"set down beside the target" failures, none was released ≥ 5 cm away and the mean release height was
4.5 cm (resting height is 4.2 cm), so nothing was dropped from above. The cube was released at a
mean 2.1 cm offset and ended at 4.5 cm: opening the gripper dragged it about 2.4 cm sideways.

### M1 and M2: acquisition failures (11 of 41, 27%)

These never lifted the cube. 8 of the 11 sit in two layouts with distinct signatures (cube centre
in the **gripper's own frame** at first closure; successes cluster at x = 0.19 / −0.10 cm,
y = −0.2 / +0.2, z = 10.5, sd ≤ 0.84 cm, at 33,400 / 50,000); the other three are single trials on
`024`, `025` and `056`.

- **`evaluation_054`** (pick at (0.242, 0.203), near the robot base, reach 0.316 m; failed 7 of 8
  trials): the cube is off-centre in hand-x by −1.7 to −3.3 cm in the failing trials (layout mean
  −2.7 cm, −3.2 sd across layouts). At 33,400 the behaviours scatter (1 success, 1 cube left 10 cm
  from the target, 2 wrong-cube picks, see M5); at **50,000 all four trials are
  identical**: fingers close on air, joints freeze, cube untouched. The layout got worse with more
  training (3 → 4 failures).
- **`evaluation_048`** (pick at (0.339, 0.425), reach 0.544 m, the **largest** reach of the 40):
  cube centre **1.8–2.6 cm farther along the approach axis** than in successes (+3.9 sd), so the
  fingertips only reach the cube's top face. One trial made **16 grasp attempts** and moved the
  cube 3.3 cm. Hypothesis (untested): descent falls short at the workspace edge.

The two layouts sit at opposite ends of the reach range, so there is no simple reach explanation.
They are not clean isolated quirks either: 7 of the 10 nearest neighbours had no failures, but the
nearest neighbour of `054` (`027`, 1.9 cm away) failed 2/8 and `040` (5.6 cm) failed 4/8.

**Empty closes are a strong failure signature.** A gripper close with no bilateral contact within
40 ticks occurs in 14/28 and 10/13 failing trials versus 4/132 and 1/147 passing ones; failures
average 4.4–4.9 close events versus 1.2. The policy retries, but from the same bad pose.

**Coverage (exploratory).** Joint 4-D (pick + place) distance to the nearest same-cell training
layout does not separate failing from passing layouts (p ≥ 0.17 in every variant), matching
FAILURES #23. The
2-D *pick*-position distance does for the acquisition layouts specifically: 2.5 cm vs 1.5 cm, n = 5
vs 35, p = 0.014. That is one of roughly thirty tests run, so treat it as a lead.

### M3: hesitation and near-misses (7 of 41)

Five trials at 33,400 *held* the cube but waited 838–916 ticks before lifting (median for
successes is 48–57). Those four layouts (`025`, `052`, `056`, `058`) have no such failure at
50,000: **extra optimisation removed this mode.** The other two M3 trials are at 50,000, both on
`048`. Two M3 trials are real time-out near-misses: `evaluation_056` ord 146 (one of the five
above; stable-stack streak 23/30 when the cap hit) and `evaluation_048` ord 115 (18/30), both
stacked correctly and released at tick ≈ 1,180.

### M5: wrong cube picked first (2 of 41; 4 of 320 trials lifted the instructed base cube)

Both M5 trials are `evaluation_054` at 33,400, distractor condition, instruction "green onto red":

- ord 139: closed on air near green, then moved the **red** cube onto green (a clean reversed stack,
  xy 1.7 cm, z −4.2 cm) and held it to the cap.
- ord 140: lifted the **red** cube first (tick ≈ 459) and carried it 55 cm away; the green cube was
  only lifted at tick ≈ 900.

The instructed base cube was lifted in 4 of 320 trials, never in a success, and **all four were in
the distractor condition** (binomial two-sided p = 0.125, so a lead, not a finding). The other two
(`ck33400` ord 71 on `037`, `ck50000` ord 159 on `059`) are different: the policy had already put
the correct cube down near the target, then re-grasped the base cube and carried it off, destroying
a near-stack. The policy does try to recover, but the recovery targets the wrong cube. 0 of 160
distractor trials ever picked up or moved a distractor, so whatever the distractors do, it is not
physical.

## 5. Arrangement effect (unexplained, exploratory)

Failures concentrate in the *familiar* arrangement (red in region A, green in region B). By layout:
33,400 — 11/22 familiar layouts had a failure vs 1/18 swapped (mean failures per layout 1.18 vs
0.11, permutation p = 0.0034); 50,000 — 4/22 vs 1/18 (0.55 vs 0.06, p = 0.13). Pooled over both
checkpoints, C1 14/88 and C3 24/88 versus C2 1/80 and C4 2/64. Pick colour (p = 0.35 / 0.44), pick
region and place region are not significant, so it is **not** explained by which side the cube
starts on or by the positional-shortcut story. Arrangement was a background factor in B2, not a
reported one. With roughly thirty uncorrected tests this does not survive a Bonferroni correction,
so **replicate before believing it**. It is the largest effect in the data that we cannot explain.

## 6. Checks that did not hold up (recorded so they are not repeated)

- **Finger-to-cube distance in the world frame.** I first read ~4.7 cm as "closed on air 5.5 cm
  above the cube". It is > 3.5 cm in **all 320 trials** and averages 4.75 cm for the successes: a
  frame offset, not a miss distance. Redone in the gripper's own frame (§4).
- **Joint-limit margin** (joint 4 near its limit in every trial) does not discriminate.
- **An early reading from the 20-trial screens** was that every failure got past grasping and the
  problem was transport or placement. It was wrong: 11 of 41 never lifted the cube (M1 + M2). It
  came from counting bilateral contact, which the failing grasps also satisfy.
- **The first version of the mechanism classifier** labelled `ck33400` ord 140 an acquisition
  failure. It fired the wrong-cube rule only when the instructed cube never moved, and missed a trial
  where the base cube was lifted first. Found by reading the trace of an odd case, fixed in
  `analysis/12_mechanisms.py` and verified with `analysis/15_wrong_cube.py`.
- **`stages.first_bilateral_contact_tick`** and related fields are satisfied by failed grasps; use
  lift and release events instead.

Lesson for the skills: validate every diagnostic on the **success** population before using it on
failures. A metric that is identical for both is noise.

## 7. Limitations

- 41 failures across 13 layouts; mechanism counts are small and the classifier is heuristic.
- About thirty permutation tests were run, uncorrected. The firm findings are the ones backed by
  mechanism or physics (release cliff, expert feasibility, non-contact of distractors, layout
  concentration); the coverage, arrangement and 50,000-reach results are leads.
- Two checkpoints evaluated on identical layouts and seeds: paired, not independent.
- Distractors were ≥ 15 cm clear of the targets and never touched; B2 cannot speak to physical
  interference or crowding.

## 8. Directions to consider (hypotheses, nothing decided)

Ordered by how much of the failure mass each addresses.

1. **Tighten release precision (addresses 21/41).** The expert's admission gate allows 2.0 cm
   (0.5% of demos exceed it) and its release offset has a p90 of 1.4 cm, so the demonstrations
   carry placement variance right at the tipping cliff. Options: tighten the admission gate
   (rejections cost little: 9 in 200), make the expert's final descent slower or lower, or weight
   the place phase in training. Test: re-collect with a ≤ 1.2 cm gate and compare the release-offset
   tail.
2. **Targeted data for thin pick coverage (addresses 12/41).** Acquisition failures sit where
   same-cell pick coverage is thinner. Test: add a small batch of demonstrations near the
   `054` / `048` pick regions on *new* layouts, re-evaluate on the same holdout. One variable at a
   time.
3. **Recovery behaviour.** Empty close is a clean failure signal. A retry that *repositions*, not
   one that repeats, is what is missing (the 16-attempt trial retried from the same pose).
4. **Evaluation protocol.** On a screening tie, report every tied candidate; or screen on more than
   20 layouts. Log release offset, empty-close count and a failure-mechanism tag per trial, so this
   diagnosis does not need a forensic pass. Add a secondary "stacked then slid" outcome.
5. **A harder distractor test (B2b).** Put distractors inside the carry path or at reduced clearance
   so physical interference is possible; B2's null applies to non-interacting distractors only.
6. **Understand before acting:** why `054` degraded with training, and why familiar is worse. Both
   are cheap to probe on the existing data before spending GPU time.
