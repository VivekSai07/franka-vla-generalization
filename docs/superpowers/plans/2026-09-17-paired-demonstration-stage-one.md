# Paired demonstration stage-one implementation plan

> **For agentic workers:** Execute inline with executing-plans and test-driven-development. No commits or experiment execution before main review.

**Goal:** Implement only fresh paired demonstration planning, recording and independent validation.

**Architecture:** A stdlib-only deterministic plan, a versioned explicit expert adapter, a recording module, and a separate paired validator under experiments/paired_demonstrations. Import unchanged physical helpers; never rewrite or monkeypatch the bound collector. Separate reports cannot impersonate legacy pilot gates.

**Tech Stack:** Python 3.10 in pinned Docker image, stdlib, existing MuJoCo/Pinocchio/LeRobot/NumPy/PyAV/Pillow. No new dependencies.

## Execution status

### Latest approved exception work — 2026-09-17

The user approved the exact three-episode amendment; the historical blocker and
lossless-media notes below describe earlier implementation stages. AV1/yuv420p
has already been restored. Current authoritative scope/evidence is in
[preparation](../../PAIRED_LEARNING_PREPARATION_20260917.md).

- [x] Read current record/validate/tests/spec and verify the two literal missing
	messages and existing metadata file hashes; preserve old sources and artifacts.
- [x] RED: exact v1-two/v2-one acceptance, evidence truthfulness/hash binding,
	additional gaps/counts/corruption rejection, collector manifest preservation,
	validation and final-readiness rechecks. CPU Docker only.
- [x] Implement the literal-message allowlist, no flag/root bypass, retain missing
	messages as unchecked, copy inventory evidence to manifest/results and compare
	it at validation and the final gate. No plan/expert/media contract changes.
- [x] GREEN: 24 focused / 129 full CPU Docker tests, actual CLI preflight exit 0, protected-source
	hash comparison and documentation review. Report measured outputs, not a
	collection or all-36 validation PASS. Main launch remains held for review.
- No collection, GPU jobs, commits, legacy artifact writes or downstream jobs.

### Historical initial implementation status

- [x] Task 1: immutable stdlib plan and complete-only sampling interface implemented/tested.
- [x] Task 2: versioned expert, lossless media, exclusive recording and first-failure preservation implemented/tested with synthetic fixtures.
- [x] Task 3: separate paired validator/replay/readiness interfaces implemented; corruption and synthetic three-mode replay tests pass.
- [x] Task 4: full 118-test CPU suite (90 old +28 new) passed; pinned configuration/CLI preflight checked. New README supplies exact commands and handoff.
- **HOLD:** No real expert trajectory, data collection, GPU job, 36-episode validation, commit or downstream implementation. Main review pending. Historical geometry audit binds 1,906 files/1,164 unique initial cube configurations but cannot cover three old episodes without saved geometry; collection fails closed.
- The step lists below preserve the pre-code implementation sequence. Completed implementation is not evidence that the deferred real-data gates have passed.

## Implementation review notes

- Pinned LeRobot's generic encoder rejects libx264rgb and its metadata heuristic rejects gbrp. The new serial subclass therefore uses public PyAV encoding and truthful per-instance stream metadata, without global patches. Exact RGB roundtrip through both PyAV and LeRobot's policy-facing decoder passes.
- Historical bank directories with missing manifests and unindexed NPZs now fail the audit, rather than silently disappearing. A failing regression test demonstrated the omission before correction.
- Review was performed inline; no independent reviewer/subagent was available. User/main review remains mandatory before any execution.

## Global constraints

- Approved specification: [paired design](../specs/2026-09-17-paired-demonstration-learning-design.md).
- Master seed 20260917; six unordered color pairs in red/green/blue/yellow combination order; two train and one held-out layout per pair, 18 layouts, 36 true expert demonstrations.
- Save exact split/layout/direction seeds and all 18 complete integration reset states before executing any expert. Same state, appearance, cameras and reset controls for both directions.
- Preserve 600 Hz physics, 30 Hz control, 20 substeps, 0.2 s settling, 9-D state, 8-D action, 512x512 front/wrist views, bounded_joint_integration_v1 and released-stack success sustained 30 ticks.
- First failed task/integrity error stops permanently; preserve partial transcript. No retry/resume/resampling/condition substitution.
- New exclusive run root with dataset and states children; no existing source/config/artifact edits, .env reads, uploads, GPU jobs, collection, expert-trajectory execution or commits in this implementation session.
- CPU tests and configuration/media preflight only; all Python in image sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc.

## Findings affecting implementation

- The expert is embedded in main. Manually extract its control loop into expert_v1.py with explicit per-direction RandomState, importing unchanged generate_task_states, IK, governor, trajectory and timing helpers. Base collector SHA256: 35209e08f874689e643f01ac5ff3db6ce493ac81a0e17270511e4470448869cf. Hash both base and adapter, shared sources/assets, runtime and image into each run. Do not reconstruct source by text substitution.
- Legacy BenchmarkRecorder and pilot validator enforce incompatible plan/count/uniqueness assumptions. New manifests deliberately do not live at the legacy replay/manifest.json entry point. Implement separate reference and persisted-float32 replay using the same physical primitives/tolerances, checking all 36 episodes. Never emit legacy replay_result.json/validation_result.json PASS artifacts.
- Full qpos hash includes robot state and is not a cube-only geometry identity. Hash ordered free-joint positions/quaternions for all four cubes, including parked cubes, independently of robot/time; reject exact duplicates against all available prior demo/bank states and other new layouts.
- Early legacy datasets franka_libero_stacking_v1 (2 episodes) and v2 (1 episode) have no replay manifests. Audit must enumerate these as missing geometry; recording remains blocked until genuine historical geometry is supplied/recovered or the user approves a revised coverage requirement. No bypass flag or fabricated independence claim.
- Use lossless RGB H264 encoding through a small pinned serial LeRobot subclass hook. This changes compression only, not image inputs. Actual decoded pixels must hash-identically to observed/rendered frames. CPU synthetic-video tests must prove the pinned hook/codec works; otherwise report blocker rather than loosen pixel checks.

## Task 1 — Deterministic plan and integrity rules

Create plan.py and tests/test_paired_demonstrations.py. Interfaces: make_plan() -> dict, validate_plan(plan), digest(value), instruction(pick, place), complete_starts(length, horizon=50). Plan contains all seeds, layout and ordered episode identities, paired/control/heldout indices; validation compares to the immutable approved schedule, including exact types.

- [ ] RED: test 18/36/24/12 counts, literal first/last identities, control forward/reverse balance, unique streams, mutation rejection, and complete-only 528 -> 479 starts with no temporal padding.
- [ ] Run new tests in the pinned CPU Docker image; confirm missing functionality failure.
- [ ] GREEN: implement stdlib deterministic SHA256-derived 32-bit streams and strict schedule checks. CLI prints plan without writing artifacts.

## Task 2 — Expert and safe recording

Create expert_v1.py, record.py and media.py. Interfaces: Expert(model).run(initial_state, episode, sink) -> outcome; sink receives observation/action plus complete per-tick transcript; collect(new_root, prior_workspace) saves plan/inventory/states then runs exactly the scheduled episodes; preflight(prior_workspace) checks configs only.

- [ ] RED: test cube-only hashes ignore arm but reject changed cube quaternion; unknown/missing prior geometry blocks; output exists/unsafe nesting rejected; failed first sink/expert stops without a second attempt and leaves incomplete ledger; both directions receive bit-identical reset arrays.
- [ ] RED: CPU synthetic media roundtrip checks exact RGB, timestamps and serial pinned LeRobot schema without executing expert physics.
- [ ] GREEN: explicit RNG reset placement with unchanged bounds/clearance/settle and unchanged expert loop with per-tick synchronous observe-command-record-step. Save failed partial frames/transcript before raising; do not finalize a failed dataset as complete.
- [ ] Record complete integration state, qpos, high-precision controls, requested/applied velocity, saturation, body/camera poses, render pixel hashes/time, phase and jitter targets, substeps, final state and success. Save seeds/state artifacts before any expert call; recheck their hashes before restore.
- [ ] CPU configuration preflight verifies model dimensions, governor bounds, base source identity, image/runtime versions and old geometry coverage, without trajectories or states collection.

## Task 3 — Paired validation and replay

Create validate.py. Interfaces: validate(new_root) -> bound paired report; require_paired_pass(new_root) -> report for future stages. No legacy gate files. Validate schedule and successful one-attempt ledger, exact split/count/task mapping, numeric/schema boundaries and complete chunks; restore saved state and compare across directions.

- [ ] RED: corruption tests reject incomplete batch, seed/task/split changes, duplicate or missing initial states, malformed/nonfinite/out-of-bound commands, timestamp/substep errors, temporal padding, false/truncated video and shifted RGB hashes.
- [ ] GREEN: reconstruct every governor command from Q_HOME and requested velocity; compare float32 rows to actual sent commands and measured state; verify camera order and RGB hashes from actual video decode.
- [ ] Reference replay re-runs the true seeded expert against every transcript, including generated commands/phases/jitter, and verifies rendered observations. Independent high-precision replay and persisted float32 replay then check all 36 full traces under original tolerances (reference full qpos <=1e-5, float32 policy state <=1e-3), stable success and timing. Do not waive a failed gate.
- [ ] Invoke unchanged original action alignment script on the new child dataset. Require its exit 0 and explicit PASS. Bind report to new sources, old sources/assets, runtime, inventory, states, dataset, transcripts and alignment output; invalidate prior paired PASS before every validation attempt.
- [ ] Training readiness is possible only after custom paired validation, all replay modes/all episodes and alignment. No completed-data claim during code preflight.

## Task 4 — Review evidence and future interfaces

- [ ] Run old 90 CPU unit tests plus new tests in pinned image with network disabled and NVIDIA visibility void; no real expert trajectory run.
- [ ] Run CPU CLI/config preflight (report missing old geometry honestly), inspect editor errors, check whitespace and unchanged tracked-file fingerprint.
- [ ] Write new README with exact Docker commands/new root, artifacts/interfaces, evidence, unresolved coverage blocker and authorization hold. No existing status/source edits or commits.

## Future stages — design only, not implemented here

1. After stage-one review and all 36-data gates, implement two fresh base-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205 training runs, control then paired: 5,000 updates each, microbatch 8 x accumulation 4 = effective 32, 160,000 chunks/model, FP32 trainable weights/moments and BF16 autocast. Seed 20260917; AdamW (0.9,0.95), wd 1e-10, clip10; 500 warmup to 1e-4, cosine floor1e-5. Separate train-only statistics, complete chunks, stock loss and frozen VLM. CPU/data tests and separately authorized tiny GPU save/reload preflight precede optimization. Save update1 and 1000..5000; final5000 primary endpoint. No implementation of trainer until paired dataset interface is reviewed.
2. Implement 72 policy evaluations: 36/model, both directions of 12 seen and 6 held-out layouts; seed20260917+layout ordinal, 1200 ticks, chunk50/horizon10, same saved resets and strict30-tick predicate. Report all scheduled denominators, control trained/opposite directions separately, both-direction layouts, contacts/lifts/wrong stacks and intervention-free score. No retries/model selection/hyperparameter adjustment. Independent CPU trace replay before conclusions.