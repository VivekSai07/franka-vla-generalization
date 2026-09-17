# Pose snapshot v2 implementation plan

User explicitly approved this design and its single-execution budget on 2026-09-17.
No additional approval gate, commit, or broader experiment is implied.

**Goal:** Correct endpoint aliasing, then measure—not promise—the outcome of the
same failed fifth paired demonstration once.

**Architecture:** A new `SnapshotTaskSpaceTrajectory` subclasses the old planner
and passes `pin.SE3` copies of both endpoints to its unchanged constructor.
A new `Expert` subclasses the old expert, inherits initialization, changes only
the reported version, and explicitly copies `run` with exactly one constructor
identifier replacement. The old loop has no injection hook; overriding the loop
is preferable to modifying bound sources or global monkeypatching. An AST check
and unified diff must prove the copied method differs only at that constructor.

**Tech:** Existing pinned Docker image, Python unittest, Pinocchio, MuJoCo, NumPy,
Pillow contact sheets. No installations, host Python, LeRobot dataset, or GPU.

## Constraints

- Preserve every pre-existing source/config/document file and all 1,090 paired
  artifacts; capture a fresh inventory because old documents have since changed.
- Add only a new plan, `experiments/pose_snapshot_v2/`, a new test module, and
  exclusive `outputs/diagnostics/pose_snapshot_v2_20260917/` artifacts.
- Exact root `paired_20260917`, episode index 4, seed 488840035,
  `train/red-blue/0/forward`, `states/layout_02.npz`; verify saved file hashes.
- Same owned RNG, phase times/jitter, IK/posture, governor, gripper, 30 Hz/20
  substeps and success scoring. No global monkeypatch or dynamic source rewrite.
- At most ONE corrected actual expert invocation, with persistent exclusive
  started marker. No second fix, retry, layout, resampling, collection or training.
- CPU Docker image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`,
  runtime runc, network none, GPU visibility void, root/repository read-only;
  only new diagnostics writable, with ephemeral tmpfs.

## Execution checklist — completed, task failure preserved

Result: alias fix verified, single actual expert still fails placement. Final
129 existing + 8 new CPU tests pass. Actual diagnostic exit 1/no OOM, 528 ticks;
saved-control replays exact, 193 old files and 1,090 paired artifacts unchanged.
No second intervention or commit. See the
[measured report](../../../outputs/diagnostics/pose_snapshot_v2_20260917/REPORT.md).

- [x] Capture current old-source and all paired-artifact hashes before code.
- [x] Add real-FK regression tests: old endpoints mutate; corrected start and end
  stay frozen after FK; frozen linear/cubic translation and rotation agree at
  clamped endpoints and intermediate times. Run the same stability assertions
  against the old class to record the expected RED failures.
- [x] Add minimal planner and explicit expert adapter; assert normalized `run`
  AST equality, one constructor replacement, inherited initialization contract,
  and unchanged legacy module identities. Save textual diff and source hashes.
- [x] Run the existing 129 CPU tests and new tests; no actual expert in unit tests.
- [x] Preflight exact saved state/trace/manifest/runtime and exclusive destination.
  Reuse prior baseline replay evidence; optionally repeat saved-control reference
  replay only for quantified contacts/scoring (not old expert execution).
- [x] Invoke corrected expert once; sink saves every trace/frame action/state and
  selected unmodified camera frames in contact sheets. Preserve partial traces
  on errors. Evaluate contacts offline on disposable data, never extra FK calls
  on the running expert robot. No dataset creation.
- [x] Validate finite commands, bounds, exact substeps (the old failed-grasp final
  tick may legitimately have zero), alignment, scoring and final integration
  state; saved-control replay may verify fixed trace, never rerun the expert.
- [x] Compare per-phase ticks, bilateral contact, lift, saturation, final XY/Z,
  stable ticks and success. Rehash old files/artifacts, save execution exits and
  report. A remaining task failure ends this intervention.

## Advice assessment to retain

30 Hz is necessary timing context, not proof of data equivalence, deterministic
policy behavior or recovery coverage. Those remain separate hypotheses. This
actual scripted expert demonstrably is not perfect; the current baseline is
129 tests, not the historical 124. Do not add random-noise/recovery experiments.