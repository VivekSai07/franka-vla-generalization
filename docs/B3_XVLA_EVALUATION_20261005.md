# B3 X-VLA evaluation on the robosuite scene — 2026-10-05

A dated report; do not edit it to reflect later findings. Measured, not sealed: the evaluation
package (`experiments/xvla_b3_robosuite_eval/`) has no contract, and everything here ran on
robosuite 1.5.2 / MuJoCo 3.4.0, which is **not physics-comparable** with any sealed MuJoCo 3.13.0
result. Protocol, interpretation rules and the three exploratory predictions were fixed before any
report trial was read (spec section 4; package README; commits `d4f4837`, `49df81d`).

## What was run

| step | result |
|---|---|
| Training (`xvla_b3_robosuite` v2, 560 episodes, sealed `e14af864…`) | 156,000 updates = 5.0 epochs, 132,298 s, final-6k loss mean 0.0005; six checkpoints verify. v1 died at update 227 (FAILURES #42) and could not resume (#43); v2 is the resumable rerun |
| Expert feasibility, 100 banked evaluation scenes | 100/100 solved, 0 layouts excluded |
| Smoke, step 156,000, 2 trials | both finish; trial 1 reproduced to the tick (475) in the later screen |
| Screen: 20 layouts × distractor × 1 seed, four candidates | **13 / 12 / 13 / 13 of 20** (steps 62,400 / 93,600 / 124,800 / 156,000). A tie by the pre-fixed rule: all four carried |
| Report: 40 layouts × {control, distractor} × 2 seeds, four candidates | 160 trials each, all `complete`, 640 trials, no retries; wall-clock 3 h 40 / 3 h 23 / 2 h 50 / 3 h 45 |

Deviation from the plan: the four screens and the four reports ran as parallel containers, not one
after another (the GPU was 1% busy and rollouts are CPU-bound). Nothing in a trial depends on the
others; each reseeds itself.

## Result (spec section 4 read-outs)

Intervals resample **layouts** (the independent unit), 10,000 draws.

| candidate (epoch) | overall | trained pairs | held-out pairs | control | distractor | distractor − control | edge | interior |
|---|---|---|---|---|---|---|---|---|
| 62,400 (2) | **110/160 = 68.8%** [56.9, 80.0] | 55/80 = 68.8% | 55/80 = 68.8% | 70.0% | 67.5% | −2.5 [−15.0, +10.0] | 17/40 = 42.5% | 93/120 = 77.5% |
| 93,600 (3) | **116/160 = 72.5%** [61.3, 83.1] | 58/80 = 72.5% | 58/80 = 72.5% | 78.8% | 66.3% | −12.5 [−26.3, +1.3] | 25/40 = 62.5% | 91/120 = 75.8% |
| 124,800 (4) | **141/160 = 88.1%** [80.0, 95.0] | 71/80 = 88.8% | 70/80 = 87.5% | 95.0% | 81.3% | **−13.8 [−25.0, −3.8]** | 33/40 = 82.5% | 108/120 = 90.0% |
| 156,000 (5) | **105/160 = 65.6%** [53.7, 76.9] | 56/80 = 70.0% | 49/80 = 61.3% | 71.3% | 60.0% | −11.3 [−27.5, +5.0] | 21/40 = 52.5% | 84/120 = 70.0% |

Edge layouts are only 10 (6 inner-corner, 4 outer-corner); inner corners are the weak band
(29% / 50% / 71% / 38%), outer corners are not (63% / 81% / 100% / 75%).

**Language (spec reading, fixed in advance).** Held-out minus trained: 0.0, 0.0, −1.2, −8.7 points;
wrong-cube rate 2/160, 0/160, 0/160, 1/160. All four are within 10 points with a wrong-cube rate
under 5%, which the spec reads as *the policy follows colour words it has not seen combined*.
Layout-clustered intervals on the gap are wide ([−23.8, +23.8], [−21.3, +21.3], [−17.5, +13.7],
[−31.2, +15.0]; 20 layouts per kind), so a gap of 15–30 points is not excluded for any candidate. The
two held-out pairs (`green→blue`, `yellow→red`) score 70 / 68 / 80 / 53% and 68 / 78 / 95 / 70%.

**Distractors.** The difference is negative for all four candidates and excludes zero only for
124,800. **Space.** Edge is below interior for all four and excludes zero only for 62,400
(−35.0 [−60.8, −8.3]); with 10 edge layouts the sign is the finding, not the size.

**Checkpoints (paired on the same trials).** 124,800 beats the other three:
+19.4 [+7.5, +31.2] over 62,400, +15.6 [+3.8, +28.1] over 93,600, +22.5 [+12.5, +33.1] over 156,000.
The other three are indistinguishable (93,600 − 62,400 +3.8 [−11.9, +19.4]; 156,000 − 62,400 −3.1
[−18.8, +12.5]; 156,000 − 93,600 −6.9 [−17.5, +4.4]). **The final checkpoint is the worst of the four
and the curve is non-monotonic**, while training loss fell monotonically to 0.0005: offline metrics did
not track capability here either. Two cautions: 124,800 is the maximum of four estimates on the same
160 trials, so its 88.1% is optimistic as a point estimate and has no held-out confirmation; and the
screen (20 trials, one seed) did not see any of this (13 / 12 / 13 / 13).

## Failures: what the video and traces show

168 failed trials. The pre-fixed `cause` (first-cause taxonomy of spec section 4) is kept as the
reported cause. It names failures by what the gripper was commanded, not by what the cube did, so a
finer mode (`failure_modes.py`, frozen before any report trial was read) is given beside it.

| pre-fixed cause | 62,400 | 93,600 | 124,800 | 156,000 | all |
|---|---|---|---|---|---|
| success | 110 | 116 | 141 | 105 | 472 |
| wrong_cube | 2 | 0 | 0 | 1 | 3 |
| never_lifted | 8 | 2 | 2 | 2 | 14 |
| lifted_not_carried | 13 | 6 | 7 | 37 | 63 |
| released_off_target | 27 | 36 | 10 | 15 | 88 |

| finer mode | 62,400 | 93,600 | 124,800 | 156,000 | all |
|---|---|---|---|---|---|
| near_miss_toppled | 17 | 34 | 9 | 9 | **69** |
| stall_above_target | 18 | 5 | 6 | 36 | **65** |
| grasp_failed | 8 | 0 | 2 | 6 | 16 |
| never_started | 2 | 2 | 0 | 0 | 4 |
| other | 1 | 2 | 0 | 1 | 4 |
| wrong_cube_lifted | 2 | 0 | 0 | 1 | 3 |
| stacked_on_wrong_cube | 1 | 0 | 0 | 1 | 2 |
| cube_lost | 0 | 1 | 0 | 1 | 2 |
| carry_aborted_near_pick | 1 | 0 | 1 | 0 | 2 |
| released_away_from_target | 0 | 0 | 1 | 0 | 1 |

What the modes are, each confirmed on video (at least one trial per mode, all 3 `wrong_cube`, all 4
`other`, 2 of 2 `stacked_on_wrong_cube`, 2 of 2 `cube_lost`):

- **near_miss_toppled (69).** The cube is carried over the target and released at the right height
  (median 5.0 cm above the stacking height, the same as successful releases), then slides off.
  It slides a median 2.5 cm in the 30 ticks after release; a successful release moves 0.06 cm. The
  signed offset at release is biased outward in y (side B −1.5 cm, side A +0.7 cm, versus −0.1 cm for
  successes) and −0.5 cm in x. 46 of 69 release more than 2 cm off centre; **23 release
  within 2 cm and still topple** (0.7–2.0 cm), so the offset does not explain all of them. Cube yaw is
  not recorded in the trace and several videos show a visibly rotated cube; this is not tested.
- **stall_above_target (65).** The cube is carried to within 1–3 cm of the target in xy and then held
  closed 12–14 cm over the stacking height until the cap (successes hover a median 68–104 ticks and then
  descend). 54 of the 65 are on side B. At 156,000 it is 36 of 55 failures and all of them are on side B.
  In 63 `lifted_not_carried` trials the cube *was* carried: the label means only "the gripper never reopened".
- **Identity and target confusion (3 + 2, plus 2).** `wrong_cube_lifted` picks a neighbouring cube (or the
  place cube) and stacks it on the other one; `stacked_on_wrong_cube` leaves the instructed cube on a
  neighbour; `carry_aborted_near_pick` holds it beside the pick for the whole episode.
  **The five `wrong_cube_lifted` and `stacked_on_wrong_cube` trials are all in the distractor condition on
  `near_pick` layouts (5 of 160 such trials; 0 of 160 `near_place`; 0 of 320 control), and so are the two
  `carry_aborted_near_pick`.**
- **grasp_failed (16) / never_started (4).** Repeated closes without lifting, or an arm that never leaves
  home (the same layout, `evaluation_046` distractor, fails at all four candidates: the arm never starts at
  two, no grasp at one, and at the fourth one seed picks the wrong cube and the other fails to grasp).
- **cube_lost (2)**: a toppled cube falls off the table edge. **released_away_from_target (1)**: the
  cube is stacked on a neighbour, then topples.
- **other (4).** The instructed cube is held closed for the rest of the episode without reaching the
  target: near the base at 62,400 (#54) and low beside the target at 93,600 (#54, #56) and 156,000 (#142;
  `evaluation_033` red→green, `evaluation_055` green→blue, both distractor).

**The classifier was validated on the successes first**: all 472 successes have `first_lifted == pick`.
Three failures are late successes rather than failures of the stack: at 62,400 (#18, #73) and 93,600
(#96) the stack stands but was completed 3, 5 and 26 ticks before the 1,200-tick cap, short of the
30-tick window. They stay failures under the fixed protocol.

### Side, crowding, and the pre-registered predictions

Side A has the pick at y < 0 and the place at y > 0; side B is the reverse. In the 20 screen layouts
(and the 20 report trained layouts) side A is always `near_pick` and side B always `near_place`
(distractors 8–12 cm from the pick, respectively the place), so the screen could not separate them.
The report can: the control condition keeps the side and removes the distractors, and the held-out
layouts cross the two.

| success (of 40 per cell) | side A control | side A distractor | side B control | side B distractor |
|---|---|---|---|---|
| 62,400 | 31 | 26 | 25 | 28 |
| 93,600 | 37 | 33 | 26 | 20 |
| 124,800 | 40 | 38 | 36 | 27 |
| 156,000 | 38 | 36 | **19** | **12** |

1. **Crowding of the place target causes the side-B place failures: inconclusive by the rule fixed in
   advance.** On the report trained side-B layouts (10 layouts, 20 trials per cell), control minus
   distractor is +0, −5, **+35**, **+20** points: two candidates at or above 15, two within 5, neither
   reaching 3 of 4. A first count of the second group gave 1 because 0.60 − 0.65 evaluated to
   −5.000000000000004 in floating point; the true value is exactly −5, and the verdict is unchanged.
   The descriptive picture is that **side is the larger effect**: at 156,000 stalls occur in 16 of 40
   side-B control trials (no distractor anywhere on the table) and 0 of 40 side-A control trials, and
   side B is worse than side A in control for all four candidates (62.5 vs 77.5, 65.0 vs 92.5, 90.0 vs
   100, 47.5 vs 95.0%). Crowding
   adds on top for 124,800 (side B 90% → 67.5%) and 93,600 and 156,000, not for 62,400.
2. **Failures belong to the layout, not the seed: supported.** Both seeds of a layout × condition cell
   agree in 90.0 / 90.0 / 96.2 / 93.8% of cells (rule: at least 80%; independent coins would give 55–79%).
   The 160 trials are about 80 independent cells, and the layout-clustered intervals are the honest ones.
3. **A failing release has an offset above 2 cm, a successful release 2 cm or less: half supported.**
   471 of 472 successful carried releases are at 2.0 cm or less (one at 2.4 cm); but only 54 of 79
   failed releases within 5 cm of the target (68%) are above 2 cm.

Not pre-registered, so exploratory, but within-layout and consistent across candidates: on the same
20 `near_pick` layouts, failures of the pick phase (`grasp_failed`, `never_started`,
`carry_aborted_near_pick`, `wrong_cube_lifted`, `stacked_on_wrong_cube`) are **2 of 160 trials without
distractors and 20 of 160 with them** (9 / 2 / 3 / 6 per candidate), while on the 20 `near_place`
layouts they are 4 of 160 and 1 of 160. A distractor 8–12 cm from the pick is what raises pick-phase
failures; placement-phase failures follow the side.

Layout level: of 80 layout × condition cells, 27 succeed twice at every candidate, 3 fail twice at all
four (`evaluation_033` distractor, `evaluation_046` distractor, `evaluation_057` control), and 43 fail
twice at one or more. Cell-level success correlates only weakly between candidates (0.07–0.51).

## Corrections to what was said before

- The screen review (2026-10-05, earlier today) read the side-B place failures as probably caused by a
  distractor near the target ("crowding is the likely cause"), while flagging that side and crowding were
  confounded. The report does not support it as the main cause: the stall persists without distractors and side
  is the larger effect (crowding adds on top for three candidates). See FAILURES #44.
- The screen's label for trial 093600 #10 was corrected from "stall over the target" to "released away
  from the target" by its trace (the cube was held 11–13 cm *beside* the target).
- The pre-fixed `released_off_target` holds six different mechanisms and `lifted_not_carried` is 90% one
  of them. See FAILURES #45.

## Limitations

- Two seeds per layout, with 90–96% agreement: about 80 independent cells per candidate, 20 layouts per
  pair kind, 10 edge layouts. No per-layout claims, per the spec.
- The failure modes were fitted to the 29 screen failures and are validated by video on the report; the
  side, crowding, near-pick, offset and slide analyses are post hoc apart from the three predictions.
- The `near_pick` / `near_place` and side comparisons are between layouts, not randomised assignments;
  the only within-layout contrast is control against distractor.
- Robosuite scene only; physics not comparable with the earlier MuJoCo results, so none of the figures
  here may be set beside B1 or B2 numbers. The sim2sim figure (the B2 policy, 3/20 on a robosuite
  scene) used another protocol.
- No pass mark was set; the first-cause taxonomy and the finer modes describe, they do not rank.

## Not established

Why side B stalls and side A does not, and why placement worsens late in training at 156,000. Not
tested: a difference between the expert's demonstrations on the two sides, joint-limit margins on the
side-B reach, cube yaw at release (not logged), the effect of the white table. Training-side
explanations (loss, schedule) are not supported by anything here.

## Reproduce

Runs: `outputs/xvla_b3_robosuite_eval/{smoke,screen,report}_step_<N>/` (read-only inputs, nothing
committed). Review artefacts and scripts: `outputs/xvla_b3_robosuite_eval/screen_review/` and
`.../report_review/` (`analysis.txt` … `analysis4.txt`, contact sheets). The modes come from
`python -m experiments.xvla_b3_robosuite_eval.failure_modes --dir <run dir> --out <file>`.
