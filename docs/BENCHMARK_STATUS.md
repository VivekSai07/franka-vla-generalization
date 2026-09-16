# Franka benchmark: start here

## Current status and authorization — 2026-09-16

**CURRENT AUTHORIZATION / LAUNCH:** user deferred Hub upload and explicitly
approved local **20,000-update SmolVLA pilot training**, batch 8 x accumulation 4,
FP32 trainable/BF16 autocast, 1e-4 peak LR, 1,000-step warmup/cosine to 1e-5.
54 tests and the two-update GPU/save-reload/data-binding preflight passed.
Fresh-base detached container `smolvla-pilot-train-20000` launched, output
`outputs/train/smolvla_pilot_20260916`. See
[training protocol and live handoff](SMOLVLA_PILOT_TRAINING_20260916.md).
No upload, further collection or final-scene evaluation. Launch is not completion;
do not modify training/provenance-bound source or start a duplicate job.

### Historical upload-first hold — superseded by training authorization above

**LATEST AUTHORIZATION / BLOCKER:** user authorized a safely named private
Hugging Face dataset upload, followed by SmolVLA protocol preparation once the
upload completes. Repository `VivekSai07/franka-panda-cube-stacking-pilot-420`
was created privately, but LFS upload failed with proxy HTTP 407. Fresh remote
inspection found only the initial Git attributes file, no dataset commit.
See [upload status and continuation](DATASET_UPLOAD_20260916.md). Local data is
unchanged. Training execution remains unauthorized; resolve upload first.

**PILOT COMPLETE — FINAL VALIDATION PASSED.** Independently inspected the
persisted runner, validator, replay result and container state on September 16.
Container `smolvla-pilot-420-overnight` exited 0 at 08:01:57 UTC, no OOM.
The dataset contains **420 episodes / 221,760 frames**, with **300 training /
120 validation** episodes, collected in 469 attempts. All 840 replay results
(420 episodes in each of two modes) passed. Alignment errors: current
0.021869, next 0.016087. Independent scene banks: 120 tuning / 240 final.
See [completion evidence](OVERNIGHT_PILOT_20260915.md#verified-completion--2026-09-16).

No training or further collection is authorized; upload is authorized above. Preserve
the dataset, scene banks and provenance-bound sources. Next proposed milestone:
prepare a fresh SmolVLA pilot training protocol with explicit manifest train IDs,
train-only statistics, held-out validation and an untouched final scene bank.
Never resume pre-fix checkpoints or blindly use the default Compose trainer.
Data-integrity PASS is not evidence of learned-policy stacking success.

## Historical authorization and progress — superseded by status above

**CURRENT LIVE JOB:** [overnight pilot](OVERNIGHT_PILOT_20260915.md) launched
after corrected preparation and full validation PASS. Container
`smolvla-pilot-420-overnight` is collecting **300 train + 120 validation** into
`datasets/franka_pilot_20260915`; runner state is
`outputs/pilot_overnight_20260915/run.json`. Automatic final validation follows
recording; no training/uploads. Do not modify hashed sources, run a duplicate,
or treat startup as completion. The historical holds/pending notes below are
superseded only by this conditional, validated launch authorization.

**Overnight launch approval:** the user requested proceeding with full dataset
collection tonight. Launch the reviewed **300 training + 120 validation** pilot
only after the running corrected 12-slot preparation completes and its full
validator passes. No waiver of integrity gates, no duplicate collection, no
training or uploads. At the latest check preparation had saved 8/12 episodes;
its manifest was still incomplete, so no pilot launch has occurred.

**Current task:** user approved joint-limit-aware expert correction and a fresh
12-slot preparation run. See [JOINT_LIMIT_PREPARATION_20260915.md](JOINT_LIMIT_PREPARATION_20260915.md).
The bounded command governor passed 48 tests and the formerly failing seed's
first approach; full recording and final validation remain pending. No pilot
or training authorization is implied.

Active container `smolvla-pilot-preparation-limits-12`; root
`datasets/franka_pilot_preparation_limits_20260915`; fresh banks under
`outputs/benchmark_preparation_limits_20260915`. Await completion, then run
the full pilot validator with `preparation_scenes`. Do not use the previous
incomplete three-episode root or claim validation before the final gate.

**Current preparation:** approved design implementation is underway; see
[PILOT_PREPARATION_20260915.md](PILOT_PREPARATION_20260915.md). Forty-four tests
pass, 420-slot and 12-slot plans and independent scene banks are generated.
`smolvla-pilot-preparation-12` **stopped with exit 1 after 3/12 episodes** in
`datasets/franka_pilot_preparation_20260915`. Joint 4 target -3.09553671 rad
violated lower actuator bound -3.0718 on slot preparation/train/0003,
seed 4281541513, tick 28. Deterministically reproduced without recording data.
The manifest is incomplete and rejected; no final validation PASS. No jobs run.
Both independent scene banks passed reload/provenance/geometry checks, but
cross-validation with a full demo set awaits the expert correction and fresh run.
Next decision: joint-limit-aware expert correction; do not weaken/skip bounds.
No 420-demonstration pilot launch, training or upload in this step. Historical
diagnostic sections below are not instructions to start another job.

**Current direction:** the user approved proceeding toward structured pilot
collection, not further automatic fitting diagnostics. The exact
[balanced-pilot specification](superpowers/specs/2026-09-15-balanced-pilot-design.md)
is recorded for review: 300 training + 120 validation demonstrations, successful
condition quotas, fixed evaluation banks and fail-closed validation. Implementation
and the 12-episode preparation gate follow review; no pilot launched yet.
Policy success is not guaranteed by dataset structure. Historical diagnostic
authorization/results below are completed, not instructions to start new jobs.

**Active authorization:** [precision-only overfit repeat](PRECISION_OVERFIT_20260915.md).
Same bounded 2,000-update single-episode test, FP32 trainable/optimizer storage
with the existing BF16 training autocast. No loss/data/timing change, no resume
or bulk collection. **Completed, exit 0:** all 33 expert normalization tensors
changed on step 1 and at the end; FP32 Adam moments verified; exact saved/reloaded
weights and predictions. Fixed-probe arm MAE 0.047295 rad vs prior 0.044300;
training-scene task result still 0/1. No arm clipping or bilateral grasp.
Evidence: `outputs/validation/smolvla_overfit_fp32_20260915`. Twenty tests pass.
Use `validation.precision.load_fp32_checkpoint` for its weights; generic older
loaders may round FP32 weights into BF16 parameter storage. No active jobs;
no additional training/loss changes automatically authorized. Preserve prior checkpoints.

Latest completed step: [fitting audit](FITTING_AUDIT_20260915.md), no optimizer
updates/rollouts/collection. Saved statistics and checked target/processor
roundtrips agree. VLM frozen as intended; all expert/projection tensors have
nonzero gradients in an audit batch. **33 BF16 expert normalization tensors
remained unchanged over the old 2,000-update run**; BF16 update rounding is an
actionable precision concern, not proven task-failure causality. The loss includes
24 padded action dimensions; temporal mask naming also mismatches but prior
training excluded padded chunks. Evidence:
`outputs/validation/smolvla_fitting_audit_20260915_final`.
Recommended next change is a controlled FP32-trainable-storage/BF16-autocast
test with the same bounded protocol; no new training is automatically authorized.

Current step: [paired replanning/grasp diagnostic](REPLANNING_DIAGNOSTIC_20260915.md),
evaluation only. Same overfit checkpoint/training scene, horizons 10 versus 1;
no additional training or collection. **Completed: both horizons failed (0/1
each), with 60 versus 600 measured chunk predictions.** Initial state/action
matched exactly; horizon 10 reproduced the prior rollout exactly. No arm-joint
clipping or bilateral target grasp in either run. Horizon 1 was slower and did
not rescue stacking. Expert-observation probes show remaining joint errors and
three incorrect release-adjacent gripper decisions. Eighteen tests pass.
Results/videos: `outputs/validation/smolvla_replanning_20260915`. No active jobs.

Latest authorized step: [single-training-scene overfit diagnostic](OVERFIT_VALIDATION_20260915.md),
episode 0 only, capped at 2,000 updates, fresh base initialization and one
training-scene rollout. No data collection or held-out evaluation in this step.

**Overfit diagnostic completed:** fixed nine-observation arm MAE decreased from
0.214860 to 0.044300 rad (79.4%), binary gripper agreement reached 99.33%, and
save/reload error was zero. **Training-scene task reproduction failed (0/1)**:
no bilateral target grasp, red cube displaced about 9.5 cm, no stack. No joint
clipping; 121 gripper-clipped ticks. Output:
`outputs/validation/smolvla_overfit_20260915`. Sixteen tests pass; no active jobs.
This shows improved fitting, not successful memorization or generalization.

The user approved a reusable joint-control benchmark design and a **small
end-to-end validation batch only**. No bulk collection, dataset upload or
training from a pre-fix checkpoint is authorized by this validation milestone.
Work is on `feat/reusable-validation`, in the original underscore checkout.

## Reading order for humans and agents

1. This status file: current decisions and execution evidence.
2. [Approved design](superpowers/specs/2026-09-15-reusable-validation-design.md).
3. [Implementation plan](superpowers/plans/2026-09-15-reusable-validation.md).
4. [Dataset-size research](SMOLVLA_DATASET_SIZE_ANALYSIS.md): source links,
   checkpoint revisions, comparison caveats and the supplied-text addendum.
5. [Current validation results](VALIDATION_20260915.md): measured gates and failures.
6. [Original two-episode report](RTC_TWO_EPISODE_REPORT.md): historical gate.
7. [Local-wheel environment](RTC_LOCAL_WHEELS.md): reproducible image setup.

## Decisions

- Goal: SmolVLA first, then pi0-FAST, pi0 and pi05, evaluated on the same
  immutable demonstrations and independent test scenes.
- Canonical data: 9-D measured joint/finger state, 8-D absolute joint-target /
  gripper command, two 512x512 RGB views, nominal 30 Hz. No normalized/padded
  actions or FAST tokens in the authoritative dataset.
- Preserve seed, full simulation initialization, model/source provenance,
  command/timestep trace and success diagnostics separately from policy inputs.
  Do not expose simulator object poses to policies.
- Proposed later budget: 300 training-demo pilot, conditional expansion to 600,
  120 held-out expert demos; these are recommendations, not guarantees.
- Full benchmark quotas, large splits and pi-family adapters are subsequent
  milestones. Small validation does not establish trained-policy success.
- Prior v1 smoke data is post-fix, two episodes. Prior v2 is interrupted bulk data,
  not training-ready. Preserve both during the new milestone.
- All Python execution/installations happen inside Docker. No host pip.
- Never read or print credentials; never commit datasets, wheels, videos or
  checkpoints. Commits use only the user-approved Git identity, no coauthors.
- Original Compose training auto-resumes and points at a fixed dataset: do not
  use its default command for this milestone.

## Execution status

**Current work:** the user authorized timing correction and another small batch.
See [TIMING_VALIDATION_20260915.md](TIMING_VALIDATION_20260915.md). Historical
results below refer to the old 34 ms contract; do not mix their dataset or
checkpoints with corrected-timing runs. The old source/assets were archived
beside the old output before editing. No bulk run is authorized.

Corrected timing unit tests: **15 pass**. Current bounded container:
`smolvla-validation-timing-v2`; target root
`datasets/franka_benchmark_validation_20260915_timing_v2`.
Collection completed: six successes / 3,168 frames, exit 0. Both replay modes
passed for all episodes with synchronized geometry and 30 Hz timestamps.
Alignment PASS: current 0.023667, next 0.017389. Fresh training/rollout container
`smolvla-learning-timing-v2` completed with exit 0: 100 finite updates, actual
weight changes and zero seeded save/reload error. **Task result remains 0/2**.
All 600-tick rollouts now span exactly 20 simulator seconds. No arm joints were
clipped; gripper clips were 204/207 ticks, bilateral target grasp counts were
zero and target lift was numerical zero. Raw/applied traces and new videos are
under `outputs/validation/smolvla_20260915_timing_v2`. No running jobs remain.
See the timing report for the next exact gate order; do not duplicate the job.

Implementation is on `feat/reusable-validation`. Eight unit tests pass, covering
simulator snapshots, deterministic conditions, invalid actions, dataset lifecycle,
fingerprints, fresh scoring, training-only statistics and chunk boundaries.

- The first foreground six-episode attempt was interrupted with exit 130 during
  terminal reuse. Its root `datasets/franka_benchmark_validation_20260915` is
  incomplete and preserved; do not train on it.
- Completed replacement: `smolvla-validation-6-retry`, exit 0, wrote
  `datasets/franka_benchmark_validation_20260915_retry`. Six successes from eight
  attempts, 3,120 frames, seed 20260915, four train/two validation episodes.
- Public model assets downloaded to `.cache/benchmark_models`, with pinned
  policy and VLM-metadata revisions. No separate VLM weights are needed when
  loading the full SmolVLA policy snapshot.
- Strict loading of `lerobot/smolvla_base` with our 9-D state, 8-D action and
  two-camera schema passed: 450,046,176 total / 99,880,992 trainable parameters.
- LeRobot 0.4.4 config loading must use `PreTrainedConfig.from_pretrained`, not
  `SmolVLAConfig.from_pretrained`, to decode the tagged `type` field correctly.
- Replay PASS: six episodes in both reference and saved-action modes; exact
  reference drift zero, maximum saved-action policy-state drift about 1.27e-7.
- Alignment PASS: current error 0.023539, next error 0.017174.
- `smolvla-learning-smoke` exited 0: 100 updates, batch 2, finite gradients,
  action-projection weight change 0.00345515, exact seeded save/reload agreement.
- **Policy task result: 0/2 successes.** Both held-out rollouts reached the
  600-tick limit. Range clipping occurred on 310 and 252 ticks respectively.
  This is a completed integration pipeline, not a successful benchmark policy.
- Outputs: `outputs/validation/smolvla_20260915` contains checkpoint, processors,
  training-only statistics, machine-readable metrics and two rollout videos.
- No containers remain running. No bulk collection/upload/new commit was made
  during this milestone. Changes remain on the working branch for review.

## Before the next milestone

1. Timing and clipping attribution were addressed in the timing-v2 repeat.
  The old-run notes above are historical; use the timing report for new results.
  Gripper clipping before binary thresholding does not alter the final binary
  command, so it is not an established explanation for failed grasps.
2. Overfit and paired replanning diagnostics completed without successful task
  reproduction. Consolidate findings and audit remaining training reconstruction
  errors before choosing another bounded fitting test; do not keep launching
  horizon sweeps or training extensions without explicit approval.
3. Add final benchmark quotas, independent seeded evaluation and consistent
   released-stack criteria. Validate pi-family adapters separately.
4. Choose a proper training budget only after these checks; 100 updates on four
   demos is not evidence that SmolVLA cannot learn the task or needs more data.

Bulk collection is still on hold. Read the evidence report before proposing
next experiments. Never resume a pre-fix checkpoint.