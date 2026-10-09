# X-VLA B3 evaluation (robosuite scene)

Published artifacts: [B3 dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) and
[X-VLA root step 124,800](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800); alternate [156,000](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800/tree/a71a75c988c3915777b8ccaf80482e112881f4d0/checkpoints/step_156000).
[HF artifact details](../../docs/HUGGING_FACE_ARTIFACTS.md#b3-robosuite-cubes) record publication
and the tokenizer/evaluator portability gaps. Downloading two checkpoints alone does
not satisfy this evaluator's original complete-training-run checks.

Measures a B3-trained X-VLA checkpoint in the robosuite scene on the 60 reserved layouts sealed with
the B3 dataset, and produces the read-outs spec section 4 fixed in advance. Plan:
[docs/superpowers/plans/2026-10-03-b3-xvla-evaluation.md](../../docs/superpowers/plans/2026-10-03-b3-xvla-evaluation.md).
Design: [docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md](../../docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md).

**Not hash-bound.** No contract lists this package. It imports the B3 data package, the robosuite
scene and the training contract read-only; the dataset and the training output are mounted
read-only and **nothing is written inside the dataset directory** (its seal covers every file in
it). All outputs go to `outputs/xvla_b3_robosuite_eval/`. Results from this scene run on robosuite
1.5.2 / MuJoCo 3.4.0 and are not physics-comparable with any sealed MuJoCo 3.13.0 result.

## Protocol (fixed before any result)

- Rollout: 1,200-tick cap; success = 30 consecutive ticks of `stack_metrics(...)["released_stack"]`
  with the unchanged window; actions bounded by the actuator ranges, gripper thresholded at 127,
  20 physics substeps per tick (the sim2sim loop, copied into `rollout.py`).
- Candidates: update 62,400 / 93,600 / 124,800 / 156,000 (epochs 2-5).
- **Screen:** the 20 screen layouts, distractor condition, one rollout seed each, every candidate:
  80 trials. Any candidate within 2 successes of the best goes to the report (a tie is a tie).
- **Report:** each carried candidate on the 40 report layouts x {control, distractor} x 2 seeds =
  160 trials; both conditions of a layout share its two seeds, so the contrast is paired.
- Seeds: `b3_robosuite_data.plan.seed("evaluation:screen:<layout>")` and
  `seed("evaluation:report:<layout>:<k>")`, identical across candidates.
- Instructions: the 12 ordered colour pairs; the two held-out ones are allowed only here.
- Read-outs: overall; held-out against trained pairs; distractor against none (per-layout pairs);
  edge against interior; wrong-cube rate; first cause of each failure. Intervals resample layouts.
  Interpretation is fixed by spec section 4. No per-layout claims (2 seeds per layout).
- Offline metrics never select a candidate.

## Files and images

| file | content | image |
|---|---|---|
| `protocol.py` | trials, seeds, 12-instruction approval, output directories | admission or robosuite |
| `analysis.py` | first-lift detection, failure causes, read-outs, layout-clustered intervals | admission or robosuite |
| `scenes.py` | restore a banked scene with hash checks | robosuite |
| `feasibility.py` | the scripted expert on all 100 banked scenes | robosuite, CPU |
| `rollout.py` | one policy rollout with per-cube tracking | robosuite, GPU |
| `failure_modes.py` | finer failure mode of a failed trial, read from `trace.npz` (`--dir RUN --out FILE`); beside `cause`, never instead | admission or robosuite |
| `screen_labels.json` | the 29 screen failures with the mode each was reviewed as | - |
| `evaluate.py` | `--mode smoke|screen|report --checkpoint N` | robosuite, GPU |

Tests: `test_protocol` and `test_analysis` in the admission image; `test_scenes`, `test_rollout`
and `test_evaluate` in the robosuite image. Run them by module name with the parent mounted
read-only (see the plan's "Running things").

## Order

1. Feasibility, once (CPU, about 75 minutes): `python -m experiments.xvla_b3_robosuite_eval.feasibility --out /vla/outputs/xvla_b3_robosuite_eval/feasibility`.
   A layout the expert cannot solve is excluded from every read-out and reported; more than 5
   such layouts means stop and ask.
2. After training has finished and verified: `--mode smoke --checkpoint 156000` (2 trials), then
   `--mode screen` for each candidate one after another (one GPU), then `--mode report` for the
   carried candidate(s). `evaluate.py` refuses without the feasibility file, without a complete
   training run whose checkpoints verify, and if the sealed parent no longer reproduces its
   fingerprint. An existing output directory is never reused, except by an explicit `--resume`
   of a `running` or `failed` run (finished trials kept, half-written ones quarantined).

An `rc=0` is not evidence that a rollout ran: check the wall-clock time and that `result.json`
lists every planned trial (FAILURES #36).

## Screen review and the exploratory predictions (written before any report trial)

The four screens (13 / 12 / 13 / 13 of 20) are a tie, so all four candidates go to the report.
Reviewing every failure on video and trace (`outputs/xvla_b3_robosuite_eval/screen_review/`)
showed that the pre-fixed `released_off_target` label holds five different failures and that
`lifted_not_carried` means a cube carried to the target and held 13-17 cm too high. The pre-fixed
`cause` stays the reported cause; a finer mode is added beside it, never in its place.

In the 20 screen layouts, side A is always `near_pick` and side B always `near_place` (parity of
the layout index in `b3_robosuite_data/plan.py`), so crowding and side cannot be separated there.
The report separates them: the control condition keeps the side and drops the distractors.

`failure_modes.py` fixes the modes before any report trial is read: `cube_lost`,
`wrong_cube_lifted`, `never_started` (gripper never closes), `grasp_failed`, `stacked_on_wrong_cube`,
`stall_above_target` (held closed 7 cm or more over the stacking height, near the target),
`near_miss_toppled` (carried release within 5 cm of the target that then fails),
`released_away_from_target`, `carry_aborted_near_pick`, `other`. On the screen it gives 8 / 6 / 5 /
3 / 2 / 2 / 1 / 1 / 1 for stall, near miss, released away, never started, grasp failed, carry
aborted, wrong cube, wrong-cube stack and cube lost. It was written to reproduce the reviewed
labels, so agreement there is a fit, not a validation; the report is its first held-out use. One
label was corrected by the trace: 093600 #10 holds the cube 11-13 cm beside the target for about
600 ticks and releases it there, which is `released_away_from_target`, not a stall over the target.

Exploratory, not gates, not part of spec section 4. Recorded now so no figure can move them:

1. Crowding of the place target causes the side-B place-phase failures (stall above the target,
   mid-air release). Supported if, on report trained side-B layouts, control success exceeds
   distractor success by 15 points or more for at least 3 of the 4 candidates; refuted if the two
   are within 5 points for at least 3 of the 4; otherwise inconclusive.
2. Failures are properties of the layout rather than of the seed: at least 80% of layout x
   condition cells have both seeds agreeing (both succeed or both fail), per candidate.
3. Near-miss placements are decided by the 2 cm xy criterion: a release that fails and then topples
   has a signed offset above 2 cm, a successful release 2 cm or less.

### Outcome (2026-10-05; the text above was fixed before any report trial was read)

1. Inconclusive by the stated rule: side-B control minus distractor is +0, −5, +35, +20 points (two at
   or above 15, two within 5). Side is the larger effect: at 156,000, stalls occur in 16 of 40 side-B
   control trials and 0 of 40 side-A ones.
2. Supported: seed agreement 90.0 / 90.0 / 96.2 / 93.8%.
3. Half supported: 471 of 472 successful carried releases are at 2.0 cm or less, but only 54 of 79 failed
   releases within 5 cm of the target (68%) are above 2 cm.

All four candidates went to the report. Results and analysis: `docs/B3_XVLA_EVALUATION_20261005.md`.
`--resume` was written and tested but not needed.
