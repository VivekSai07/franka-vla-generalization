# Balanced Franka pilot: collection and evaluation contract

Date: 2026-09-15. Status: proposed exact specification for review.

The user approved moving toward structured pilot collection rather than further
diagnostic training loops. This specification defines the remaining preparation
work. Review the exact contract before implementing it and launching collection.
Successful recording is not a promise of successful policy learning.

## 1. Goal and evidence boundary

Produce one immutable demonstration dataset usable by SmolVLA, pi0-FAST, pi0
and pi05, with model-specific preprocessing and a fixed evaluation protocol.
The existing timing-v2 recording/replay pipeline passed; small learned policies
have not yet stacked successfully, including on a training scene after 2,000
updates. That is a risk to report, not a reason to run diagnostics indefinitely
or a basis for guaranteeing that more data fixes learning.

The first deliverable is collection preparation and its small regression batch.
The subsequent pilot is **300 training demonstrations plus 120 validation
demonstrations**, not a 300-total dataset with validation taken out afterward.
No training, model upload, Hub upload or pre-fix checkpoint resume is included.

## 2. Alternatives considered

1. **Quota-based pilot (selected):** exact successful-task counts, fixed splits,
   immutable seeds and shared evaluation. Additional preparation is justified
   by cross-model comparability and avoiding expensive recollection.
2. **Unrestricted random 300:** quicker to launch, but accepted successes can
   underrepresent hard instructions/layouts. Rejected for the benchmark pilot.
3. **More one-scene training experiments first:** may inform model optimization,
   but do not establish the required benchmark data coverage. Deferred.

## 3. Frozen physical data contract

- Timing version: `fixed_substeps_synchronized_v2`; physics 600 Hz, control
  30 Hz, exactly 20 substeps. Preserve 0.2 seconds of settling.
- Observations: synchronized front and wrist RGB images, 512 x 512, copied at
  the same boundary as the measured state. Preserve camera names/order/config.
- State: 9 physical float32 values, seven arm joint angles in radians and two
  finger positions in metres.
- Action: 8 physical float32 values, seven absolute commanded joint targets
  in radians and the gripper command (0 closed, 255 open). Never substitute
  next-achieved joint positions for the commanded target.
- Keep the existing robot, scene, controller gains, 15 cm cube clearance,
  workspace ranges and Gaussian target jitter. No domain expansion in this pilot.
- Common success rule: XY separation < 0.02 m, Z difference strictly between
  0.035 and 0.045 m, open gripper command and no finger contact with the top
  cube, sustained for 30 consecutive control ticks. Use fresh post-step geometry.
- Numerical invalidity, out-of-range expert commands, unreleased stacks and
  malformed frames must fail validation, not be silently clipped into successes.
- Model inputs never include object poses or expert future states. Replay/
  evaluator-only state and ground truth live in separately identified sidecars.

## 4. Successful episode quotas

Enumerate colours in the existing order red, green, blue, yellow; enumerate all
ordered unequal pairs in that order and assign pair IDs 0 through 11.

| Split | Per pair | Total |
|---|---:|---:|
| Training | 25 | 300 |
| Offline expert validation | 10 | 120 |

For pair ID p, allocate the extra example to distractor-count condition p mod 3:

- Training: counts [8,8,8] plus one extra in that condition.
- Validation: counts [3,3,3] plus one extra in that condition.

Thus overall training has exactly 100 examples each with zero, one and two
distractors; validation has 40 each. Every instruction occurs in every scene
size. For one-distractor slots, alternate the two eligible colours in a seeded
order, keeping their accepted counts within one per pair and split.

Failures do not consume a successful slot and never change that slot's pair,
distractor count or distractor identity. Retry with a new deterministic layout
seed. Stop the job if any slot reaches 10 failed attempts; do not quietly
replace a hard condition. Report counts and reasons even if the run stops.

This prevents instruction/count selection bias, but accepted layouts are still
conditioned on expert success. Report that limitation explicitly; the evaluation
scene bank must not be filtered by expert or policy success.

## 5. Seed, split and version integrity

Generate a plan manifest before simulation. Each slot has immutable ID, split,
pair, cube list and bounded retry seeds. Use separate deterministic seed streams
for training, validation, tuning scenes and final scenes; detect seed collisions.
Split identity is assigned before collection, not by random frames afterward.

Store full initial simulator state plus sampled object configurations, seeds,
attempt reason, timing metadata, source/asset/runtime hashes and image/controller
configuration. Detect exact duplicate initialized scenes across splits. Different
seeds alone are not proof of geometric independence. Report spatial occupancy,
relative distances and accepted-versus-attempted distributions rather than
claiming guaranteed continuous-space coverage from quota counts.

Each completed dataset has a content manifest and version. Future additional
training episodes must preserve the first 300 episode IDs as a nested subset.
Do not edit old metadata to make it look compatible with changed source or stats.
All historical smoke datasets and learned checkpoints remain separate.

## 6. Interruptions and completion

Keep implementation minimal: fresh explicit roots, incomplete status at creation,
attempt ledger, and complete status only after LeRobot finalization and validation.
Do not implement or advertise resume support in this milestone. If interrupted,
preserve partial artifacts and stop for a recovery decision; no overwrite and no
automatic full rerun. A saved episode count alone is not proof of finalization.

The validator must support arbitrary episode counts and splits, replacing the
current smoke-only assumptions of six episodes and four/two split. Require:

1. Exact plan quotas and disjoint complete split assignments.
2. Correct schema, finite values, image/video availability and complete decoding.
3. Monotonic matching timestamps, no episode-crossing chunks and correct substeps.
4. Every saved action matching the same-tick high-precision command within
   float32 rounding tolerance; gripper encoding exact.
5. Reference and float32-action replay for every demonstration, correct initial
   state/provenance, declared drift tolerances and released-stack outcome.
6. Original arm action-alignment gate PASS with actual episode-count reporting.
7. Source/runtime/content-bound validation results, invalidating stale passes.

Keep a separate recording-complete and validation-passed status. A failure does
not become a pass by loosening tolerances or silently dropping episodes. Training
must refuse a dataset lacking final validation.

## 7. Independent evaluation scenes

Freeze two separate scene banks before training:

- Tuning: 120 scenes, ten per pair; 3/3/3 plus one rotated extra per count.
- Final: 240 scenes, twenty per pair; 6/6/6 plus extras in two rotated count
  conditions per pair. Overall 80 final scenes per distractor count.

These are initial simulator states, not expert demonstrations or training rows.
Use the same feasibility checks as initialization (valid geometry, clearance,
workspace bounds), but do not reject a feasible scene because a policy/expert
fails. Seal the final bank's content hash before tuning and do not optimize on
its outcomes. No final evaluation is run during collection preparation.

Use the same controller, images, state, instructions, bounds, fresh scoring and
40-second simulation timeout (1,200 ticks) for the eventual controlled benchmark.
This deliberately differs from the earlier 20-second diagnostic timeout; label
old results accordingly. Record any clipping, failures and invalid actions.
Preserve binary task success and intervention-free success as separate metrics.

Use a controlled execution horizon of ten actions initially, record prediction
chunk size and sampling seed, and report wall time/latency alongside simulator
time. Simulated pauses awaiting inference are not real-time deployment success.
Model-optimized horizons belong to a separate deployment comparison.

## 8. Model reuse and training boundary

Keep raw physical recordings immutable. Resize images, map keys, pad dimensions,
normalize, encode FAST tokens or construct relative actions in versioned adapters.
Fit statistics and any learned tokenizer only on selected training episodes;
do not use dataset-global statistics across the validation split. Derived relative
actions require statistics in the same relative space and documented gripper rules.

The pi-family software/adapter smoke checks remain explicit prerequisites to
claiming tested compatibility; the current SmolVLA environment is not proof.
Pin checkpoint/data revisions and document pretraining, trainable modules,
precision, effective batch size, exposure and compute budgets for comparisons.

FP32-trainable checkpoint reload must use the verified precision-preserving
loader. The installed SmolVLA temporal padding-key mismatch and inclusion of
dummy action dimensions in the loss must be handled/documented before a new
training recipe is approved. They do not require changing the physical dataset.
Never resume pre-fix checkpoints or select a smoke dataset by a hardcoded name.

## 9. Preparation acceptance and launch decision

Implement tests for exact quota totals, failed-slot retries, seed/split isolation,
duplicate detection, validator count generalization and incomplete/stale result
rejection. Preserve timing tests and existing regressions.

Run a fresh **12-success preparation batch**, one per pair, rotating distractor
conditions; assign nine training and three validation slots before collection.
This is a wiring check, not the full per-pair/split coverage test. Unit tests must
exercise the full 420-slot plan; the small physical batch must pass replay,
timing, serialization, finalization and alignment without new policy training.
Do not count these preparation episodes toward the pilot.

Present plan counts, checks and estimated resource use after the preparation
gate. Then launch the reviewed 300+120 pilot in a detached, monitored container
with an explicit output path and no duplicate job, only under confirmed launch
authorization. Keep training and upload separate. Stop/report on failure.

Approximate recording cost at the earlier measured ~88 seconds per success:
7.3 hours for 300 training, 2.9 hours for 120 validation, about 10.3 hours total
before retry overhead and final replay/decoding validation. This is an estimate
from a small sample, not a completion guarantee. Check disk space before launch.

## 10. Confidence statement

The recording foundation is validated, and structured coverage is a rational
next investment. We can verify data integrity and fair evaluation controls.
We cannot guarantee policy success: the actual answer requires training and
independent closed-loop measurement. Neither repeated tiny-data failures nor
confidence in collection substitutes for those measurements.