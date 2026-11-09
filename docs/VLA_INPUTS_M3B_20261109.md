# VLA inputs M3b: the box from pixels instead of the oracle, on A3's frozen policy — 2026-11-09

Written 2026-11-09 from the recorded outputs (`outputs/vla_inputs_eval/m0_20261030/` arms A5-A9 and `m3b_analysis.json`, `m3b_detector_error.json`,
`outputs/vla_inputs_box_detector/run_v1/`, the identity-gate scratch folders) and the chain logs. Dated record: do not edit; later findings go into the live documents.
[Spec](superpowers/specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md) · [plan](superpowers/plans/2026-11-05-vla-inputs-m3b-predicted-box.md) ·
packages `experiments/vla_inputs_arm_xvla_force_box_noisy/` (A6-A9), `experiments/vla_inputs_box_detector/`, `experiments/vla_inputs_arm_xvla_force_box_pred/` (A5),
`experiments/vla_inputs_m3b_analysis/`. **No policy was trained.**

## Question and rule

M3's box is an oracle (simulator state). M3b asks how much of A3's gain over A2 survives a box predicted from the camera images (Q1), what box error A3 tolerates (Q2),
whether A3 follows the box or the language (Q3), and where a predicted box fails (Q4). One scale: A2 (no box, 523/573 at 88,830) is the floor, A3 (oracle box, 571/573 at
101,520) the ceiling, a gap of 48 layouts. **R = (S - 523) / 48.** Decision rule D10, fixed before any run: **R >= 0.75 closes M3b** (M4a reconsidered at lower priority);
0.25 <= R < 0.75 partial (M3b-2 considered); R < 0.25, or A5 not above A2, points to M4a.

## Result: R = 0.81, the rule says M3b closes (the interval straddles the 0.75 threshold)

Comparison tier, 573 feasible layouts of 576 (the same 3 dropped for every arm), one trial each, paired. Core `c4f6c303f2ae`, checkpoint 101,520 for A3 and A5-A9.

| | successes | note |
|---|---|---|
| A2 (floor, 88,830) | 523 / 573 = 91.3% | |
| **A5 (box from pixels)** | **562 / 573 = 98.1%** | arm `301d7d723561` |
| A3 (oracle box, ceiling) | 571 / 573 = 99.65% | |

- **R = 0.8125, 95% interval [0.647, 0.952]** (layouts resampled stratified by object, the three arms together, 10,000 resamples, seed 0). With A2 at 101,520 as the
  floor (517, gap 54): R = 0.833 [0.694, 0.957]. **The point estimate is in the "closes" band; the interval reaches below 0.75, so the band is not established by the
  interval.** The rule picks the branch by the point estimate and the claim is made with the interval.
- **A5 against A2 (the one pre-registered test): +6.8 points [4.4, 9.3], 46 layouts only A5, 7 only A2, exact McNemar p = 4.0e-8**; A5 is above A2.
- A5 against A3 (descriptive): -1.6 points [-2.8, -0.4], 2 only A5, 11 only A3, p = 0.022. The oracle still beats the predictor on 11 layouts.
- A5's 11 failures: never lifted 7, dropped outside 2, timeout while carrying 2, **wrong object 0**. By object: bottle 96/96, bread 95/95, can 95/96, cereal 94/96, lemon 89/94
  (5 of the 11), milk 93/96. Trained pairs 467/477, held-out 95/96. One distractor 285/287, two 277/286.

## The detector (A5's input)

A class-conditional two-camera centre-point network (a ResNet-18-shaped backbone per camera, 23,784,084 parameters), from scratch, trained on the 480 training episodes of the
sealed P2 dataset (every 8th frame, 30,780 cached frames; S0 boxes as labels; absence learned from exact frame translations), 60 epochs. The checkpoint is epoch 59 by validation IoU
(0.9746 mean over the two cameras). **Gate (thresholds committed before extraction, never moved), all met with large margins:**

| per camera | front | gripper | threshold |
|---|---|---|---|
| decoded validation frames: mean IoU | 0.965 | 0.984 | 0.85 |
| decoded: share of frames below IoU 0.5 | 0.88% | 0.03% | 3% |
| raw replayed renders: mean IoU | 0.965 | 0.984 | 0.85 |
| raw renders: share below IoU 0.5 | 0.85% | 0.03% | 3% |
| synthetic out-of-view presence F1 (smaller class) | 0.986 | 0.989 | 0.90 |

The raw-render numbers equal the decoded ones, so the video-codec shift (mean absolute difference about 1.9 grey levels on the front crop) costs nothing measurable. **Caveat: the checkpoint was chosen on
the same 60 validation episodes the gate measures (as the spec says), so the gate is a go/no-go for GPU time, not an estimate of rollout performance.** Offline IoU did not predict R in advance
and is not used as capability here.

## Corrupted-box arms A6-A9 (Q2, Q3), each against A3's 571

| arm | condition | successes | A3-only / arm-only | exact McNemar p (Holm over the four) | prediction | |
|---|---|---|---|---|---|---|
| A6 | jitter, sigma 6 px | 542 (94.6%) | 31 / 2 | 1.3e-7 (2.6e-7) | 560-571 | below the range |
| A7 | jitter, sigma 18 px | 345 (60.2%) | 226 / 0 | 1.9e-68 (5.6e-68) | 500-565 | far below |
| A8 | gripper-camera box missing half the time | 545 (95.1%) | 28 / 2 | 8.7e-7 (8.7e-7) | 480-571 | inside |
| A9 | a distractor's box drawn instead of the target's | **0 (0.0%)** | 571 / 0 | 2.6e-172 (1.0e-171) | below 458 | as predicted |

Exploratory and unadjusted, against A2: A6 +48 / -29 (p = 0.04), A8 +48 / -26 (p = 0.014), A7 +28 / -206 (p = 1e-34).

- **Where the loss is.** Almost all of it is "never lifted": A6 25, A7 206, A8 24 of their 31, 228, 28 failures. Jitter and dropout leave the grasp-and-carry mostly intact; they break the approach.
  This is consistent with M3's reading that the box mainly says where the object is, not which one.
- **Q2, what A3 tolerates.** 6 px of noise already lowers the front-camera IoU with the true box to 0.585 (the front box is small, median 27 px) and leaves 31 failures (-5.1 points against A3); 18 px (front IoU 0.254)
  leaves 228 (-39 points) and puts A7 178 layouts *below* A2: a badly wrong box is worse than none. A missing gripper box half the time leaves 28 failures (-4.5 points), although A3 never saw a missing box;
  the front box seems to carry most of the information. Held-out pairs fall further under A7 (49/96 against 296/477 trained). IoU is not a clean dose axis: A5 ends 9 layouts below A3 at front IoU 0.93 (0.967 on its
  successes), A6 29 below at 0.585.
- **Q3, box or language.** A9 scores 0 of 573: the box wins over the instruction (the pre-registered reading, below 40%, is "the box is followed"). **But the dominant outcome is not the decoy being
  picked:** 446 trials never lift anything (78%), 126 lift a distractor (22%, wrong object), 1 carries the target and times out. Why the policy stalls with a conflicting box is untested; "conflicting cues freeze it" is
  a hypothesis, not a finding, and the box-names-the-target versus box-marks-the-place question stays open (A9 shows only that removing the correct box position from the images, and showing another object's, is fatal).

## Q4: where the predicted box fails

From A5's per-tick log matched to M0's traces by the oracle box (all 573 trials, 206,964 ticks; see "Log defect" below). Predicted against live oracle box, per tick:

| | front camera | gripper camera |
|---|---|---|
| mean IoU, all ticks | 0.931 | 0.931 |
| share of ticks below IoU 0.5 | 3.9% | 2.2% |
| share of ticks with no predicted box (oracle present) | 3.9% | 2.0% |
| false alarms (box predicted, oracle absent) | 0.0% | 0.02% |
| mean change between consecutive replans, predicted / the oracle's own | 4.96 px / 4.78 px | 7.46 px / 7.49 px |
| successful trials (562): mean IoU / share with no box | 0.967 / 0.3% | 0.943 / 0.9% |
| failed trials (11): mean IoU / share with no box | 0.218 / 76.9% | 0.700 / 54.9% |

- **The failures are detector misses in the front camera, not imprecise boxes:** the front box is absent for 77% of ticks in the failed trials against 0.3% in the successful ones, and the front IoU is about 0 in 7 of the 11
  (ordinals 67, 215, 229, 262, 263, 542, 575: 4 lemon, the smallest object, 2 cereal, 1 can; lemon is 5 of all 11 failures). It is not the fixed front crop: the oracle box lies inside the crop at every tick of all 573 trials. Why the detector misses
  these layouts is not diagnosed here.
- The predictor adds no measurable frame-to-frame jitter (its replan-to-replan change equals the oracle's own motion), consistent with A5 doing much better than A6 at a similar nominal error.
- The gripper box is missing for about 2% of ticks (the oracle itself has the target out of the gripper frame 1.5% of the time).

## The prediction against the result (recorded 2026-11-05, before any run)

| | predicted | measured |
|---|---|---|
| A5 R | 0.6 to 0.95 (S 552-569); "below 0.25 would surprise me" | **0.81 (S 562)**, inside |
| A5 never lifted / wrong object / timeout+unstable / dropped outside | 3-20 / 0-12 / 0-8 / 0-3 | 7 / 0 / 2 / 2, all inside |
| A6 | 560-571, almost no loss | 542, **below** (31 lost) |
| A7 | 500-565, a measurable drop | 345, **far below** |
| A8 | 480-571, the widest range | 545, inside |
| A9 | below 458, wrong-object the main cause | 0; direction right, **mechanism wrong** (22% wrong object, 78% never lifted) |

The headline and the A5 failure classes were predicted. The corruption arms were more fragile to jitter than predicted (the prediction treated 6 px as small, against a front box that is only 27 px wide) and A9's cause was misjudged:
same pattern as M2 and M3 (FAILURES #69), reasoning from one failure category.

## Run record

- **Build:** all packages tests first, every test confirmed against deliberate mutations; nothing frozen was edited (core `c4f6c303f2ae`, A2/A3 closures unchanged). A6-A9 and A5 share A3's frozen adapter, M2's force state and the same checkpoint;
  the only difference is the drawn box. **Live identity gate** (a zero-corruption adapter re-runs A3's trials 1 and 291, every trace array bit for bit): attempt 1 failed because my gate did not run under the worker's deterministic-algorithms runtime
  (FAILURES #70); attempt 2 passed on both trials, all 12 arrays. With a detector returning the oracle, A5 equals A3 byte for byte, and the leakage test fails against an adapter mutated to draw the oracle.
- **Night A** (A6, A7, A8, A9; 20:57 to 06:13, 573 of 573 each, no failed worker) and **the chain** (detector gate, A5 prepare, smoke 2 of 2, comparison 573 of 573; 14:19 to 16:55): no stage was retried.
- Detector: extraction about 7 minutes, training 60 epochs, gate in about 2 minutes.

## Log defect (disclosed)

A5's per-tick log was written as `worker_<pid>.jsonl` with trial id `p<pid>-r<resets>`; in a container every worker is pid 1, so the six workers appended to one file under colliding ids, and the analysis refused it. The headline
analysis does not use the log (it was re-run without it). The Q4 table above is a recovery: each trace tick takes the first unused log line, in file order, whose logged oracle box equals the trace's `target_box` byte for byte
(`analyse.detector_error(..., interleaved=True)`, tested on interleaved synthetic logs and against mutations). All 573 traces were assigned completely; 598 lines were left over, consistent with the two smoke trials' ticks written to the same
file (not verified line by line). A mis-assignment is possible only between identical oracle pairs, which the scene makes improbable; treat Q4 as exploratory.

## Caveats

- **The forces stay privileged.** A3's state carries M2's exact simulator wrench and finger forces: A5 is "a policy that needs a force sensor and a camera", not a deployable system.
- The detector is a closed-set detector for six known objects selected by the instruction, in one scene; nothing here says anything about open-vocabulary grounding.
- **A3 never saw noise, so A5's shortfall mixes "poor predicted information" with "an input distribution it was never trained on".** A6-A9 separate it only partly; a retrain on noisy or predicted boxes (M3b-2) was not run.
- One training seed, one rollout per layout; intervals describe layout noise only. A3's ceiling is 571 of 573, so near the top the tier cannot rank further.
- The R interval [0.65, 0.95] straddles the 0.75 threshold: the data support "most of the gain survives" but not a sharp location inside the closes band.
- The gate and the checkpoint choice share the 60 validation episodes.
- The A6-A9 and A5 comparisons are conditional on these checkpoints and recipes; the Holm family (A6-A9 against A3) is exploratory, and M3b adds no arm to the programme's family.

## Programme consequence

D10 branch: **R >= 0.75, M3b closes; M4a (learn the box inside the policy) is reconsidered at lower priority.** M3b-2 (retrain on noisy boxes) was not needed by the rule. The user has since set a different direction for the next steps; M3b-2, M4a and
any M3b download bundle were not started. To reclaim disk, at the user's instruction on 2026-11-09, the detector's frame cache and the per-trial videos (`cameras.mp4`) of arms A5-A9 were deleted; the arms' `prepared.json`, stage `result.json`, per-trial `episode.json`, `telemetry.json` and `trace.npz`, `m3b_analysis.json`, `m3b_detector_error.json`, the detector weights and gate record, this report, the code and the docs were kept (see BENCHMARK_STATUS.md). The Q4 table can still be re-run (it needs only the kept traces and A5's log).
