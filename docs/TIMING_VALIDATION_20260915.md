# Timing correction and repeated small validation

Date: 2026-09-15. Branch: `feat/reusable-validation`.
Authorization: review prior rollout failures, correct timing and repeat the
small validation. No bulk collection, uploads or checkpoint resume.

## Baseline evidence

The earlier completed run is documented in [VALIDATION_20260915.md](VALIDATION_20260915.md).
It achieved zero of two held-out stacking successes after 100 updates on four
demonstrations. Sampled video frames at ticks 0, 99, 199, 299, 399 and 599 show
workspace motion without completing a stack. These samples do not establish
the exact failure cause or prove absence of all transient contact.

Generated contact sheets and a source/asset ZIP are saved with the old outputs.
Every archived collector/source asset was checked against the old dataset's
manifest hashes before modification. The old data and weights are untouched;
they must be used with that archived timing/runtime, not the new code.

The old stepping loop overshoots each requested 1/30-second interval to 17
physics steps at 2 ms each. A minimal simulation reproduced:

| Quantity | Expected | Observed |
|---|---:|---:|
| Time after 300 control ticks | 10.0 s | 10.2 s |
| Per-tick interval | 33.333 ms | 34 ms |
| Effective control frequency | 30 Hz | About 29.41 Hz |

A moving-body regression also demonstrated that the returned geometry lagged
integrated qpos by one physics step. These are confirmed timing/synchronization
defects. They are **not established causes** of the policy's zero success rate.

## Corrected contract

- Version: `fixed_substeps_synchronized_v2`.
- Physics timestep: 1/600 s; control period: 1/30 s; exactly 20 substeps.
- Reject nonintegral physics/control ratios instead of silently rounding up.
- Preserve 0.2 seconds of settling using 120 new physics substeps.
- Refresh geometry/contact/bias before observation and after each control tick.
  Preserve the complete MuJoCo integration state, including warm-start values,
  around the forward pass. The changed substep size and refreshed bias may
  change dynamics; new data/replay validation is therefore mandatory.
- Copy both rendered arrays explicitly. Record synchronized body/camera poses
  with qpos/timestamps for independent replay checks.
- Observe, compute commands, apply both arm and gripper commands, record, step.
- Check released stacking after integration using current geometry: XY <2 cm,
  height difference strictly 3.5–4.5 cm, open command, no finger contact, sustained
  for one second. Collection, replay and rollout use the same criterion.
- Model image inputs and measured joint state now describe the same control
  boundary. Policy inputs still exclude ground-truth object poses.
- Replay requires the exact contract, source/assets/verifier hashes, current
  runtime versions, LeRobot FPS and persisted timestamps to agree. Old PASS
  records cannot authorize new-timing training.

The legacy deployment client was updated to use the same timing and success
criterion. Its network/server path is not exercised by this local rollout test;
that is a separate deployment test.

## Rollout diagnostics

Retain the same 100-update, batch-2, four-train/two-validation integration test
and initialize from the same pinned SmolVLA base weights. Do not resume the
previous 100-update checkpoint. Keep 50 predicted actions, 10 execution steps
between replanning and a 600-tick rollout limit (now exactly 20 seconds).

Save raw/applied commands and timestamps; separately count joint clipping per
dimension, any-joint clipping, gripper clipping and gripper thresholding. Record
bilateral target contact, lift relative to initial height, finger-midpoint
distance and final placement error. These are diagnostics, not privileged
policy observations. If clipping occurs, never count success as unassisted.

Because corrected dynamics can change which attempts succeed, matching seeds
and the condition schedule do not guarantee identical accepted initial scenes.
Do not interpret a score change as a controlled timing-only performance gain.

## Execution checklist

- [x] Reproduce 200 ms drift over 300 ticks and stale endpoint geometry.
- [x] Preserve old source/assets and inspect sampled old rollout frames.
- [x] Add failing tests; implement timing and separate clipping diagnostics.
- [x] Expanded 15-test suite passes; static review findings addressed.
- [x] Collect six successes in a new directory (maximum 30 attempts).
- [x] Verify replay, clocks, geometry and the unchanged alignment criterion.
- [x] Run fresh 100-update training, save/reload and two instrumented rollouts.
- [x] Record results and remaining limitations; leave bulk collection on hold.

New dataset: `datasets/franka_benchmark_validation_20260915_timing_v2`.
New output: `outputs/validation/smolvla_20260915_timing_v2`.
No result is claimed until the corresponding command and persisted artifacts
confirm it.

### Active execution handoff

The six-episode collection completed with exit 0 in detached container
`smolvla-validation-timing-v2`, ID
`6589e2e6f16e6288d887d1cbe0d3be6ad5d4e5938d38f25965046d63d07dad2a`.
It saved six successes from eight attempts, 3,168 frames (528 per episode).
The manifest is complete with four training and two validation episodes.
Both replay modes passed for all six episodes. Reference full-state drift was
zero; the maximum float32-action policy-state drift was 5.5266e-5, below the
predeclared 1e-3 tolerance. No replay tolerance was relaxed. The float32-action
drift is larger than the old run, not an improvement claim; contact dynamics
can amplify small command-rounding differences. All endpoints remained released
stacks for 180–181 consecutive ticks.

Recorded timestamps agree with the 30 Hz clock (maximum persisted float32
timestamp error 8.901e-7 seconds). Exactly 20 physics substeps were checked per
tick. Reference body/camera positions also matched the synchronized trace.

The unchanged alignment verifier returned exit 0:

```text
mean |action_t - state_t|     (current, should be LARGER): 0.023667
mean |action_t - state_t+1|   (next, should be SMALLER):   0.017389
PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.
```

Fresh training/rollout container: `smolvla-learning-timing-v2`, ID
`4c8f2584377f6cb103fc45daae5bbd2cf446b7ab003d5e8154e6212ac5476688`.
It completed normally with exit 0 and no OOM. No containers remain running.

The learning test uses `validation/smolvla_smoke.py` with the new root and
fresh timing-v2 output directory, the same offline image and pinned base
assets. It does not resume the old learned checkpoint. Report rollout success and
separate clipping/contact/lift findings even if task success is still zero.

Whitespace check uses `cr-at-eol` because the existing deployment client uses
CRLF line endings; no unrelated whole-file newline conversion was made.

## Final learning and failure-analysis results

The fresh base initialization ran all 100 optimizer updates at batch 2 on the
four training episodes. Gradients and losses were finite and the tracked action
projection changed by up to 0.00302049. First and last sampled training losses
were 0.600249 and 0.142173; different sampled batches do not establish validation
improvement. Training-only elapsed time was 12.19 seconds, excluding model load,
save/reload and rollouts. Peak PyTorch allocated CUDA memory through training
was 2,102,137,856 bytes; this is not total GPU usage or a production estimate.
Saved/reloaded policy and processors produced exactly identical seeded physical
action predictions (maximum difference zero).

| Diagnostic | Red-on-blue rollout (episode 4) | Green-on-red rollout (episode 5) |
|---|---:|---:|
| Control ticks | 600 | 600 |
| Simulated elapsed time | 20 seconds | 20 seconds |
| Wall time | 106.12 seconds | 101.70 seconds |
| Released-stack success | No | No |
| Any arm-joint clipping | 0 ticks | 0 ticks |
| Gripper clipping | 204 ticks | 207 ticks |
| Raw gripper range | -141.87 to 419.19 | -183.76 to 381.96 |
| Bilateral target contact | 0 sampled ticks | 0 sampled ticks |
| Maximum target lift | About 6e-13 m (numerical zero) | About 6e-13 m (numerical zero) |
| Closest finger-body midpoint to target centre | 0.3722 m | 0.2312 m |
| Final target/base XY separation | 0.28064 m | 0.29108 m |

Finger-body midpoint distance is a diagnostic proxy, not calibrated fingertip
clearance. Contact/lift measurements are sampled at control boundaries and do
not rule out every transient substep contact. The inspected contact sheets at
six equally distributed ticks per video show workspace motion without grasping
or stacking, consistent with the recorded outcomes. This is sampled visual
inspection, not a frame-by-frame manual review.

Gripper outputs overshoot the expert's 0/255 command values. Mean/std
denormalization of a continuous model is not inherently bounded. Importantly,
clipping into [0,255] followed by a threshold at 127 yields the same binary
decision as thresholding the raw value. Therefore the clip counts must not be
described as a demonstrated cause of task failure. Joint-limit clipping did
not explain these two failures either. Incorrect approach/object selection,
limited training coverage and insufficient optimization remain hypotheses;
the present integration sample does not separate them.

### Comparison and next decision

Both old and corrected runs achieved 0/2 task successes. The timing defect is
fixed and the synchronized data are replayable; policy task competence is still
unproven. The corrected control loop produces 528 rather than 520 frames per
episode. Even with the same accepted attempt seeds here, changed dynamics,
image timing and training samples prevent causal performance attribution to
timing alone.

Recommended next diagnostic is an explicitly bounded **training-scene overfit
and replay/policy comparison**, with the same action interface, before making
claims about held-out generalization or scaling data. This follow-up has not
been run or authorized by completion of the current milestone. The eventual
300-demo benchmark still needs balanced successful quotas and a frozen split/
evaluation specification; pi-family adapters remain unvalidated.

Authoritative artifacts:

- Dataset: `datasets/franka_benchmark_validation_20260915_timing_v2`.
- Replay evidence: that dataset's `replay/replay_result.json`.
- Learning metrics: `outputs/validation/smolvla_20260915_timing_v2/result.json`.
- Videos: `rollout_4.mp4` and `rollout_5.mp4` in that output directory.
- Raw/applied commands and diagnostic traces: `rollout_4_trace.npz` and
  `rollout_5_trace.npz`, alongside sampled visual contact sheets.

All prior artifacts are preserved. No bulk collection, upload, commit, push or
pre-fix checkpoint resume was performed. Work remains on the uncommitted
`feat/reusable-validation` branch for review.