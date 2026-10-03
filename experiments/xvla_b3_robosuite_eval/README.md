# X-VLA B3 evaluation (robosuite scene)

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
   fingerprint. An existing output directory is never reused.

An `rc=0` is not evidence that a rollout ran: check the wall-clock time and that `result.json`
lists every planned trial (FAILURES #36).
