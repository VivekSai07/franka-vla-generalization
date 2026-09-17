# Paired learning preparation — 2026-09-17

## Current: stopped on expert placement failure

Container `smolvla-paired-collect-36` stopped with record exit 1. Manifest is
failed, training_ready false. Four successful demonstrations (both directions
on the first two red/green layouts) were saved; the fifth failed transcript is
preserved. Full validation did not start; no training is permitted on this root.

Failed episode: `train/red-blue/0/forward`, expert seed 488840035, state
`states/layout_02.npz`, transcript `transcripts/episode_000004.npz`. Reason:
not_released_and_stable_one_second, 528 recorded trace ticks, stable streak 0.

Read-only CPU replay of all saved float64 controls reproduced every qpos exactly
(maximum error 0). Red was grasped and lifted successfully: bilateral contact
28 grasp ticks, 60 Verify Lift ticks, 90 transport ticks and 60 lowering ticks;
maximum lift about 23.91 cm. Placement missed, rather than initial grasp failing.

Joint 4 commanded target saturated at its lower bound (interior target
-3.071799 rad; limit -3.0718) for **202 ticks**, first at tick **233** during
Move to Place. Saturation counts by phase: transport 54, lowering 58, release
30, retract 60. No other joint target saturated. At end of lowering the cube
was 7.52 cm off in XY and 8.71 cm above the destination cube center. The
time-driven expert still released at tick 347. Final cubes both rested on the
table, separated 8.09 cm in XY, Z difference effectively zero.

This establishes a placement failure with prolonged joint-limit saturation and
release before a placement-accuracy gate. It does not prove the layout physically
unreachable or prove that changing a bound, orientation, duration or posture will
fix it. The governor correctly prevented out-of-range commands; it does not
solve constrained Cartesian planning. Do not disable it or relax stack scoring.

SVT preset mapping and MP4 moov messages are encoder notices, not this failure.
The stop-on-first-failure behavior worked as specified. No retries, resampling,
deletion or metadata repair were performed. All 18 original reset artifacts and
the partial run must remain preserved.

Recommended next design decision: bounded same-layout expert reachability/IK
diagnosis followed by an explicitly versioned placement-feasibility correction,
including a no-release-until-placement-valid gate. A release gate alone only
prevents a bad demonstration; it does not solve the stuck approach. Verify any
correction on this exact failed layout before seeking approval for a fresh batch.
Do not jump directly to model retraining or a replacement random layout.

Earlier running statements below are historical and superseded.

## Current: collection launched under approved exception

Main verification reran all **129 CPU tests**, passing in 26.026 seconds.
Launched `smolvla-paired-collect-36`, ID
`843f8c80fb9edeb24f2fb0512b3bfd8123a72ba4f26a0c0c5dcca924eb5953bf`, with
the exact pinned image, CPU-only runc, no network, read-only repository and
old data, writable new paired-output namespace. Approximately 85 GB free.

Run root: `outputs/paired_demonstrations/paired_20260917`. Startup confirmed
container running, `PHASE record`, new plan/inventory/manifest files created.
This is startup evidence, not successful demonstration or validation evidence.
The detached driver runs `record` then `validate` only if collection exits 0,
and stops on any nonzero exit. No automatic retry, training or upload.

Do not edit the paired module directory or approved spec while collection or
validation runs: both are provenance-bound. Watch the retained container logs
and new manifest/validation report. Training must wait for the full 36-demo
validation PASS, not merely a recording-complete count. Downstream two-model
training and 72-rollout evaluation implementation remains a subsequent stage.

The earlier held/no-job statements below describe historical preparation.

## Latest: approved narrow exception implemented; launch held for review

The user explicitly approved the [three-episode spec amendment](superpowers/specs/2026-09-17-paired-demonstration-learning-design.md#approved-historical-geometry-amendment--2026-09-17).
Only the literal v1-two-episode and v2-one-episode missing-initial-geometry
messages are allowed. The gaps remain visible and **unchecked**; overlap with
those three legacy layouts cannot be ruled out. No broad ignore flag, root
exclusion, recovered/fabricated poses or legacy artifact changes were added.

Machine-readable `historical_geometry_exceptions` now accompanies the historical
inventory, new manifest, preflight and validation results. Each record includes
the exact dataset/count/message, `reason: userapproved20260917`,
`geometry_checked: false`, exclusion from new train/eval, and the actual existing
inventory metadata file SHA256. Validation and `require_paired_pass` reconstruct
the policy and compare fresh inventory hashes, manifest and result evidence.
Failure reports retain the manifest caveat and separately state whether the
historical inventory was rechecked; a failed report never grants readiness.

No old samples can enter the unchanged new-only schedule: 24 paired training,
12 control training (subset of those 24), 12 held-out demonstrations; all eventual
policy evaluations use the 18 new reset states. Additional missing geometry,
changed counts/paths, metadata/replay count mismatch, missing/corrupt/nonfinite/
unindexed artifacts, hash mismatch and all other integrity errors remain fatal.

### Fresh measured evidence

- TDD RED: exact approved gaps were blocked; exception evidence absent; synthetic
  collector could not reach its intentional first failure. These failures were
  observed before production edits. GREEN: **24 focused CPU tests**, 12.282 s;
  **129 full CPU tests**, 26.075 s, `OK`, no failures or skips. Native deprecation/
  encoder warnings and the intentionally corrupt-video decoder error remain.
- Actual CLI `experiments.paired_demonstrations.record --preflight`: **exit 0**;
  `configuration_status: passed`, `collection_status: not_started`,
  `inventory_error: null`, **1,906 historical files / 1,164 geometry hashes**.
  Both original missing messages remain present. Zero expert trajectories,
  `training_ready: false`. No collection, replay/rendering validation or GPU job.
- Actual exception metadata file hashes (not geometry hashes):
  - [v1, two episodes](../datasets/franka_libero_stacking_v1/meta/info.json):
    `7561ab94498836e48a26e182f98aa9d59ed94fcb09b2bd5e64b64c5edc18d649`.
  - [v2, one episode](../datasets/franka_libero_stacking_v2/meta/info.json):
    `67f2bee9ece73ad6a85cf08b40b5a0b901d3e299a3aaf44831dc36fabe26229d`.
- Pinned image:
  `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`;
  `--runtime runc`, no GPU devices/visibility, no network, repo read-only.
  All Python/tests ran in Docker; synthetic artifacts existed only in container
  temporary directories. No persistent paired run or validation PASS was created.
- Post-check: no running containers and no paired output namespace exists. Both
  record/validate `--help` commands exit 0. Editor diagnostics and tracked diff
  whitespace checks are clean. The protected tracked-source SHA256-list digest
  (data collection, robot/controllers/assets, training, validation and inference)
  is unchanged before/after:
  `468f1bed62ffc317869506996966bcb80dd4b8c8996fdbae2524c3532ee45e9f`.
  The new expert/plan/media files and both legacy metadata hashes are also unchanged.
  Spec/status documentation changes are intentional and are not included in this
  protected-source digest. Review was inline; main review remains pending.
- [Record/validate commands and intended outputs](../experiments/paired_demonstrations/README.md#exact-commands--review-first-do-not-launch-yet)
  are ready for main review. The proposed fresh run is named `paired_20260917`;
  record then validate only after launch approval. All-36 feasibility/replay/
  media/alignment gates remain unexecuted. Downstream training/evaluation remain
  unimplemented. No commits, uploads or source-bound legacy changes.

## Historical preparation — blocker superseded only by the amendment above

User approved the written two-model experiment specification. Stage-one
implementation adds isolated plan, expert, recording, media and validation
modules under experiments/paired_demonstrations; old bound sources are unchanged.
Downstream two-model training and 72-rollout evaluation are not implemented yet.

## Review and verification

- Worker test evidence: 124 CPU Docker regression tests passed after corrections.
- Independent review caught omitted prior paired-output resets in duplicate
  detection; regression tests and inventory handling now include partial/prior runs.
- An unapproved lossless-H264/exact-decoded-pixel requirement was removed before
  collection. Media now uses the existing AV1/yuv420p encoder contract. Full
  decode/dimension/FPS/interval/frame-count checks remain; lossy decoded pixels
  are not required to equal raw RGB exactly.
- Actual configuration-only preflight: configuration passed, collection blocked,
  zero expert trajectories executed, training_ready false.

## Historical blocking geometry coverage (before explicit approval)

The approved specification requires exact duplicate-layout comparison against
previous demonstrations and banks. Two legacy v1 smoke episodes and one
incomplete v2 episode lack full initial geometry. Joint positions alone cannot
recover historical object configurations. Do not fabricate poses or silently
claim those three episodes were checked.

Proposed narrow amendment for user review: check every recoverable historical
layout (including the complete 420-demo pilot, scene banks and prior paired
resets), list the three unverifiable legacy episodes explicitly, and exclude
those legacy episodes from all new training/evaluation. This preserves all
current benchmark split isolation checks but cannot guarantee no geometric
overlap with those three older episodes. No other integrity gate is relaxed.

Alternative: obtain authentic initial geometry for the three episodes before
proceeding. Do not delete old datasets or infer missing object poses from arm state.

No collection, training, GPU preflight or evaluation job has started for this
experiment. New stage-one implementation is uncommitted pending resolution of
the scope amendment; specification commit 6c9d4f6 remains the approved design.