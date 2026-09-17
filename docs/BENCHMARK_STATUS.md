# Franka benchmark: start here

## Current X-VLA full run — 2026-09-18

Corrected two-update micro4xaccum2preflight PASSED exactFP32model/Adam/prediction
reload(error0) and data binding. User authorized deleting only failedscratch
weight/optimizer artifacts for space; all reports and other checkpoints preserved.
Freshbase5k+final13rollouts job `xvla-single-task-5000-20260918` launched into
`outputs/train/xvla_single_task_20260918`. See [current launch](XVLA_TRIAL_20260917.md#current-launch--2026-09-18).
No source/admission edits, duplicates, recipe changes, uploads or extensions
while running. Prior blocked messages below are historical. No task result yet.

## Current X-VLA trial — 2026-09-17

User approved bounded X-VLA comparison, then explicit5k micro4xaccum2 fullFP32/
BF16 recipe+13rollouts. Publicpinnedassets downloaded. Initialmicro1two-updateGPU
preflight passed exactweights/Adam/predictionreload. Exactbatch4scratch completed
2updates peak21.33GB, then failed assistant checker on action=None duringreload.
Checkerfixed with3regressions;47CPUtests passed. Failedscratch/evidence preserved,
CPUadmission now intentionally stale; nofulltraining orrollouts launched.
See [X-VLA status and correction](XVLA_TRIAL_20260917.md). Needapprovalforonefresh
corrected scratch underno-retryagreement; do notoverwritefailureroot or bypass
admission. SmolVLA/planner/recollection remainpaused.

## Single-task experiment complete; stop for decision — 2026-09-17

5,000 updates and13rollouts complete, exit0/noOOM. Capability **failed**:
seen1/9, held-out0/4 (required8/9,3/4). Six seen trials lifted requested target;
only one stacked; all four held-out trials lacked target bilateral contact.
Fresh model/data/source/cache/stat/inventory checks pass; program integrity is
not task success. See [completion findings](SINGLE_TASK_20260917.md#completed-capability-gate-failed).
No retries/extensions/newtraining/collection/planner work. The agreed stopping
rule applies; wait for a decision before starting another experiment.

## Current priority: single-task capability — 2026-09-17

User paused planner debugging and approved one simplified existing-data run:
red-on-green/two-cube train0–8, held-out300–303, freshbase5kupdates batch8x4,
FP32trainable/BF16autocast,500warmup1e-4decay1e-5, final-only13rollouts.
219CPUtests andtwo-updateGPUpreflight passed, reloaderror0. Detached job
`smolvla-single-task-5000` launched; see [single-task protocol](SINGLE_TASK_20260917.md).
No recollection, planner changes, extensions or final-bank evaluation. Do not
duplicate job or edit its sources. Historical v3/v4 and paired-learning plans
below are paused, not instructions for further work.

## Adaptive subdivision numerical stop — 2026-09-17

Approved bounded subdivision executed one saved-state numerical route, no physics.
It passed the original 0.069763rad continuity rejection, completed93/113 original
waypoints, then failed lowering IK backtracking at1.707mm position residual
(required<=1mm), joints1/3atinteriorbounds. Limits unchanged, no retry. 188 tests
passed and independent review confirmed narrow numerical claims. See
[v4 result](ADAPTIVE_PLACEMENT_V4_20260917.md). No complete path or dynamic task
success. Next requires route/posture redesign rather than tolerance relaxation;
no new collection/training is authorized by this failed gate.

## Bounded placement v3 stopped — 2026-09-17

Approved v3 implemented in isolated package; 168 CPU tests and independent review
passed. One actual case (episode4) stopped after 197 pick/lift ticks at
waypoint_branch_jump: max joint displacement0.0697633rad > approved0.05rad.
Rejected waypoint not executed, no intentional release, no remaining four cases
or retry. Reference replay qposerror0/fullstatesexact; old hashes unchanged.
Container exit1/noOOM. See [v3 report](BOUNDED_PLACEMENT_V3_20260917.md).
Expert placement still unresolved; new adaptive-subdivision design would need
approval. No collection/training or final-scene rollout follows this failure.

## Pose-copy correction tested — 2026-09-17

User approved a versioned endpoint-copy fix and one same-layout test. New
`experiments/pose_snapshot_v2` preserves old bound sources. 129 existing +8 new
CPU tests passed; exactly one corrected expert invocation on failed episode4
still failed placement (exit1/noOOM): saturation202->169ticks, finalXY80.95->97.77mm,
stable0. Alias defect fixed, task not fixed. No retries/recollection/training.
See [results and external-advice assessment](POSE_FIX_AND_EXTERNAL_ADVICE_20260917.md).
Next proposed bounded change: joint-limit-aware continuous placement planning
and measured release gating; design approval required. Existing paired data stays
failed/not-training-ready. Old source and artifact hashes were preserved.

## Placement investigation — 2026-09-17

Bounded CPU IK found joint-valid solutions for all three exact failed placement
endpoints; no continuous-path feasibility claim. Independently confirmed a
planner bug: FK returns a mutable frame pose and TaskSpaceTrajectory retains
it, so later FK calls move its stored start. Minimal reproduction moved t=0
start 6.13 cm; previous-tick-start reconstruction matches failed trace velocities
exactly. See [investigation](PLACEMENT_REACHABILITY_20260917.md).
No source fix, expert rerun, recollection or training was performed. Recommend
versioned immutable-pose correction and bounded same-layout test first, not
disabling limits or replacing the layout. Correction requires approval.

## Paired collection stopped — 2026-09-17

`smolvla-paired-collect-36` record exited 1 after four successful saved demos;
fifth `train/red-blue/0/forward` (seed 488840035) failed placement. Saved-control
CPU replay qpos error 0: grasp/lift succeeded, joint 4 target lower-bound
saturation persisted 202 ticks, time-driven release occurred off-target;
final XY gap 8.09 cm, no stack. Full validator and training never started.
Preserve all partial artifacts; no retry/resample/resume or loosened criteria.
See [failure diagnosis](PAIRED_LEARNING_PREPARATION_20260917.md#current-stopped-on-expert-placement-failure).
Next proposed step is bounded same-layout expert planning diagnosis, not model
training or automatic recollection. Historical running banners below superseded.

## Current paired collection — 2026-09-17

User approved the exact three-legacy-episode geometry exception. Fresh 129-test
CPU suite passed. Container `smolvla-paired-collect-36` launched into
`outputs/paired_demonstrations/paired_20260917`, automatically running full paired
validation only after successful collection. Startup confirmed; no completed-demo
or training-readiness claim yet. See [live preparation handoff](PAIRED_LEARNING_PREPARATION_20260917.md).
No duplicate jobs, source/spec edits, retries, new training or uploads while this
gate runs. Two-model training/evaluation remain gated subsequent stages.

## Paired stage one prepared — launch held for review, 2026-09-17

User explicitly approved the narrow exception for two legacy v1 episodes and one
v2 episode without initial geometry. Implemented test-first: literal-message
allowlist only, visible **unchecked** gaps, approval reason and actual metadata
file hashes in inventory/manifest/results, fresh validation and readiness-gate
rechecks. Unknown gaps and all non-geometric integrity errors still fail.
Old sources/artifacts remain unchanged; all new train/eval uses new paired data.

**129 CPU Docker tests pass. Actual read-only CLI preflight exits 0:** configuration
passed, collection `not_started`, 1,906 files / 1,164 geometry hashes, both approved
missing messages retained, zero expert trajectories and `training_ready: false`.
See [approved amendment and current preparation evidence](PAIRED_LEARNING_PREPARATION_20260917.md).
**No collection, GPU jobs or commits. Main launch follows review.** The 36-demo
record/validate commands are prepared, not executed; downstream two-model training
and 72-rollout evaluation implementation remain pending.

## Historical paired preparation blocker — superseded by approval above

User approved the paired-demonstration two-model design. Isolated stage-one
collection/validation implementation is prepared; 124 CPU tests reported passing.
Fresh configuration-only preflight passes configuration but blocks collection:
two old v1 episodes and one incomplete v2 episode lack full initial geometry
needed by the approved historical duplicate check. No expert trajectories or
training jobs executed. See [preparation and proposed narrow amendment](PAIRED_LEARNING_PREPARATION_20260917.md).
Do not bypass this gate or claim training readiness. Downstream training/evaluation
implementation remains pending; new stage-one files are uncommitted.

## Paired diagnostic complete — 2026-09-17

48/48 trials, exit 0/no OOM, 12/12 original traces bit-exact, all 48 saved-action
replays and unchanged final fingerprints. Correct stacks: original 2/12,
reversed 1/12, alternate seed 1/12, seen training scenes 1/12. Four original/
reversed scene pairs contacted a cube in both runs and chose the same cube;
eight pairs had no contact in either run. No demonstrated object-choice switch.
Poor performance on seen scenes rules out an exclusively unseen-layout problem.
See [results and next design recommendation](PAIRED_DIAGNOSTIC_20260917.md#verified-completion-and-interpretation).
No job is active from this diagnostic. Proposed small contrastive two-cube
learning intervention needs exact design/budget approval; no bulk collection,
training, upload or final-scene evaluation automatically follows completion.

## Current authorized diagnostic — 2026-09-17

User approved **48 policy rollouts**: 36 same-scene original/reversed/seed-control
tuning comparisons plus 12 two-cube training-scene reproductions. Container
`smolvla-paired-diagnostic-48` launched with unchanged 020000 checkpoint,
40-second timeout and horizon 10. See [protocol and handoff](PAIRED_DIAGNOSTIC_20260917.md).
90 tests passed; original-condition reproduction is a live gate, not yet a
completion claim. No new training, demonstration collection, upload or final-scene
evaluation. Do not change bound sources or duplicate the running diagnostic.

## Full tuning evaluation complete — 2026-09-17

**4/120 correct stacks (3.33%), 0/120 without any clipping.** All four successes
are two-cube scenes (4/40); three/four cubes each 0/40. Container exited 0/no OOM,
post-run fingerprints unchanged. Independent all-120 saved-action replay matched
recorded states exactly, all pairwise 30-tick scoring checks agree, 141,121 video
frames decoded. 110 trials had no bilateral instructed-target contact; first
contact was wrong-object in 17, requested-target in 9, absent in 94. Eight wrong
stable ordered-pair stacks correctly failed. See
[full review and decision tree](FULL_EVALUATION_REVIEW_20260917.md).

Additional inspection found 300 unique training layouts, identical gripper
transition frames 107/347 in all train demos, and original offline probe blind
spots. No new collection/training/policy rollouts started. Recommended next:
bounded paired-instruction comparison, optionally training-scene reproduction,
then one evidence-selected intervention. Exact follow-up budget needs confirmation;
final 240 scenes remain sealed. Historical running notes below are superseded.

## Instruction audit — 2026-09-17

User reported mismatched instruction/video behavior. [Detailed audit](INSTRUCTION_AUDIT_20260917.md)
confirms wrong-pair terminal stacks in trial_002 (green on red), trial_008
(green on yellow), trial_035 (yellow on red); all correctly scored failures.
At the 41-completed snapshot only trial_022 succeeds, correctly red on yellow.
No label/color/normalizer/scoring mismatch demonstrated; all 221,760 row tasks
match, saved stats exact, three full saved-action replays exact. Controlled
fixed-observation instruction swaps suggest weak immediate language sensitivity,
not a proven sole cause. No training/scoring/source changes or new policy rollouts.
Authorized 120-scene baseline continues unchanged; final bank remains sealed.

## Current status and authorization — 2026-09-16

**CURRENT AUTHORIZED JOB:** user approved rollouts. Selected 020000 checkpoint
is running on exactly 120 tuning scenes in `smolvla-pilot-eval-tuning-120`,
40-second timeout/horizon 10; final 240 scenes are not evaluated. Model loaded,
first trial started; launch is not a success score. See
[evaluation handoff](PILOT_EVALUATION_20260916.md). No training/upload; preserve
bound sources and do not duplicate this job. The pending-approval note below
is historical and superseded by this authorization.

**TRAINING COMPLETE:** `smolvla-pilot-train-20000` exited 0/no OOM at
21:16:00 UTC. Verified 20,000 consecutive finite updates, all 11 checkpoint
model hashes and fresh dataset/runtime/training-source bindings. Selected
checkpoint **020000** by predefined full-chunk held-out arm MAE: **0.016913 rad**;
first-ten **0.008329 rad**, gripper accuracy **93.60%**. No closed-loop success
claim. See [completion report](SMOLVLA_PILOT_TRAINING_20260916.md#verified-completion).
No active training remains from this job. Next proposed task: selected-checkpoint
evaluation on 120 independent tuning scenes, 40 seconds/horizon 10; approval
required. Final 240 scenes remain untouched. No restart, collection or upload.

### Historical training launch — completed

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