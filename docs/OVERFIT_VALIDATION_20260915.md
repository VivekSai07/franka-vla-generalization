# Single-training-scene overfit diagnostic

Date: 2026-09-15. User authorized this bounded diagnostic; no bulk collection.

## Question

Can SmolVLA fit commands from one known successful demonstration and reproduce
its task from the exact saved training scene? This separates fitting difficulty
from the earlier evaluation on unseen colour pairs. A success here establishes
neither generalization nor sufficiency of a larger dataset.

## Fixed protocol

- Existing timing-v2 dataset: `datasets/franka_benchmark_validation_20260915_timing_v2`.
- Select only episode 0 (red on green, two cubes), already passed expert-action
  replay. Other episodes, including validation, are not used for optimization,
  normalization, probe evaluation or rollout selection.
- Preserve the canonical dataset manifest/splits. The run separately records
  its actual selected training and evaluation episodes, both `[0]`.
- Initialize anew from the pinned `lerobot/smolvla_base` snapshot, not any
  previous fine-tuned checkpoint. Same verified image and timing contract.
- Maximum 2,000 optimizer updates, batch 2, AdamW learning rate 1e-4, BF16
  autocast, seed 20260915. Expert and configured projections train; VLM frozen.
- Sample only 50-action chunks contained within the single training episode.
  Compute statistics solely on its physical state/action rows.
- Probe nine fixed training frames spanning the valid chunk-start range at
  updates 0, 100, 500, 1,000 and 2,000. Seed model noise identically for each
  probe without changing the training RNG stream. Report physical arm MAE,
  first-ten-action MAE, gripper MAE and binary gripper agreement. Probes are
  training reconstruction metrics, not independent validation scores.
- Save/reload policy and processors; require seeded prediction agreement.
- One rollout on the exact training initialization after the capped training,
  up to 600 ticks (20 simulator seconds), execution horizon 10, identical
  controller and released-stable-stack criteria. Save video and raw diagnostic
  traces. Do not silently extend training if reproduction fails.

Fresh output: `outputs/validation/smolvla_overfit_20260915`.
Command entrypoint: `validation/smolvla_smoke.py --overfit` with the root/output
above, in the existing Docker image with networking disabled.

## Interpretation rules

1. Poor training reconstruction: investigate optimization, input/target mapping
   and model fitting before attributing failure to generalization.
2. Good reconstruction but failed closed-loop training scene: investigate
   distribution shift, action execution, feedback and recovery. Offline MAE
   alone is insufficient evidence of usable control.
3. Successful training-scene reproduction: a positive learnability diagnostic,
   not permission to advertise benchmark success or collect 300 episodes.

## Execution status

Sixteen unit tests passed, including the new single-episode selection test.
The source/data/runtime-bound replay gate was rechecked and passed before launch.
A read-only review found no concrete new isolation/probe correctness defect;
probe RNG restoration remains inspection-based, not a dedicated unit test.

Completed container: `smolvla-overfit-2000`, ID
`8ef6b43ad66cebe0d03c94979e3908ead35ad5fc6f1a96c6cdf931d8d11fd922`.
Exit code 0, no OOM. Previous datasets, checkpoints and timing reports remain
unchanged. The budget was not extended. No held-out scene was evaluated.

## Measured outcome: improved reconstruction, failed task reproduction

The selected episode contains 528 frames and 479 valid 50-action chunk starts.
Exactly 2,000 updates at batch 2 sampled 4,000 chunks (about 8.35 exposures per
valid start on average, not thousands of independent demonstrations). Training
including fixed probes took 238.23 seconds. The tracked action projection
changed by up to 0.00989551. Loss/gradient checks remained finite throughout.
Reloading the saved policy and both processors reproduced identical seeded
physical predictions, maximum difference zero.

### Fixed training-observation reconstruction

All measurements below use the same nine frame indices and seeded sampling
noise, across 50-action chunks. They use expert observations, not the states
visited by the learned policy. They are neither test-set nor closed-loop scores.

| Update | Arm MAE (rad) | First 10 actions arm MAE (rad) | Gripper MAE (0–255 units) | Binary gripper agreement |
|---:|---:|---:|---:|---:|
| 0 | 0.214860 | 0.173661 | 126.124 | 55.33% |
| 100 | 0.106771 | 0.104343 | 54.240 | 96.44% |
| 500 | 0.072774 | 0.063223 | 44.777 | 95.56% |
| 1,000 | 0.059276 | 0.049194 | 25.914 | 99.33% |
| 2,000 | 0.044300 | 0.039062 | 18.823 | 99.33% |

The 79.38% decrease in mean arm error demonstrates actual improvement on these
fixed training examples. Residual 0.0443 rad is about 2.54 degrees averaged
across joints/timesteps; it does not translate directly to a fixed end-effector
error and may still matter greatly for a precision grasp. Per-probe arm MAEs
at 2,000 updates range approximately 0.0283–0.0730 rad. Nine probes do not prove
that every recorded frame has been fit, and binary gripper agreement over
chunks does not establish correct gripper timing in closed loop.

### Exact training-scene rollout

Instruction: pick up the red cube and place it on top of the green cube.
Restored the exact saved integration state of episode 0; no expert trajectory
or simulator object poses were supplied as policy inputs.

| Metric | Result |
|---|---:|
| Task successes | **0/1** |
| Rollout duration | 600 control ticks / 20 simulated seconds |
| Wall time | 104.69 seconds |
| Arm-joint clipping | 0 ticks, all seven dimensions |
| Gripper clipping | 121 ticks |
| Bilateral target contact | 0 sampled control boundaries |
| Maximum target rise | 0.00822 m |
| Closest finger-body midpoint to target centre | 0.06685 m |
| Target net displacement | 0.09539 m |
| Final target/base XY separation | 0.5660 m |
| Final target/base Z separation | Approximately zero |

Sampled frames at ticks 0, 99, 199, 299, 399 and 599 show the robot approaching
the red cube, followed by cube displacement rather than a completed grasp and
stack. This is consistent with a failed contact interaction; the samples do
not pinpoint contact type or rule out transient between-tick contacts. The
reported finger-body midpoint distance is not a fingertip-clearance measure.

Gripper clipping precedes binary thresholding and does not change the final
open/close decision relative to thresholding the raw value. Do not call that
clipping an established cause of the failure. Unlike the previous two rollouts,
this test uses a different, trained scene; distance metrics are not directly
comparable measures of improvement across the experiments.

## Conclusion and next diagnostic

The training pipeline can improve reconstruction of this demonstration, but
**successful closed-loop memorization was not established within 2,000 updates**.
The result does not show that larger datasets alone will fix the failure, nor
does it establish that the architecture cannot learn the task.

Recommended next diagnostic, not executed here: compare policy versus expert
commands around the approach/grasp transition, then evaluate this same saved
checkpoint and training scene with a shorter replanning horizon (for example,
one action rather than ten), keeping controller, timing and scoring fixed.
This can help distinguish imperfect fitting from feedback/action-chunk effects
before another training-budget change. Use multiple fixed rollout seeds if
estimating reliability; the current single rollout is not a success-rate study.

Do not begin 300-episode collection on the basis of this result. Keep the
benchmark quotas and independent generalization evaluation as later milestones.

## Artifacts and handoff

- Machine-readable result: `outputs/validation/smolvla_overfit_20260915/result.json`.
- Local checkpoint/processors: `pretrained_model/` under that output.
- Training-scene video: `rollout_0.mp4`.
- Raw/applied actions and diagnostic trace: `rollout_0_trace.npz`.
- Sampled visual summary: `rollout_0_contact.png`.
- Train-only statistics: `training_stats.json`.

The canonical dataset and split were not modified. All generated artifacts are
Git-ignored. Code/report changes remain uncommitted on `feat/reusable-validation`.
No bulk collection, upload, push or pre-fix checkpoint resume occurred.