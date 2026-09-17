# Adaptive placement v4 — numerical-only implementation plan

**Goal:** Test the approved bounded adaptive subdivision on the exact saved v3
episode-4 lift state, once, without executing the expert or physics.

**Architecture:** New planner explicitly injects the unchanged v3 solver, pose
waypoint generator, branch validator, collision checker and time parameterizer.
Only a solved, residual-valid segment exceeding 0.05 rad is bisected, along its
original desired Cartesian segment (not a drifting achieved-pose path).
Read-only sources are reused by imports/adapters, never monkeypatched or rewritten.

**Tech stack:** Existing NumPy, Pinocchio, MuJoCo and unittest in pinned offline
Docker image sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc.

## Scope and hard limits

- All new code/tests/plan live in experiments/adaptive_placement_v4 only.
- Outputs exclusively outputs/diagnostics/adaptive_placement_v4_20260917.
- Old packages, tests, specs and saved outputs stay hash-identical; no credentials.
- No commits, physics stepping, expert calls, rollouts, collection or training.
- One real numerical path attempt; exclusive start marker prevents repeats.
- <=400 solver attempts, counting failed/rejected parents and both children;
  <=100 outer IK updates per attempt, <=40,000 total attempted outer updates.
- v3 solver semantics: iterations 0..100 are residual evaluations; only
  iterations 0..99 can attempt a QP/update. Failed QP/backtracking updates count.
  Inner caps remain 2,048 coordinate sweeps and 12 backtracking trials.
- <=5 mm / 0.02 rad desired steps; solved arm steps <=0.05 rad; depth <=8.
- Refuse a split if both child translation and rotation would be below 1e-5.
  A zero-rotation translation segment may subdivide; a pure rotation may too.
- Same 1 mm / 0.01 rad independent FK residuals; reject nonfinite full q before
  FK, never use NaN comparison behavior as acceptance. v3 pose_errors itself
  checks residuals but not input q; v3 solve checks seeds and trial q.
- Same intersected bounds/interior margin, 2 rad/s scheduling and 10,000
  collision samples including the previous-control bridge; unchanged contact
  rules and discrete <=0.005 rad/joint geometry sampling.
- Preserve original seeded transport/lower/release endpoints exactly. Recreate
  RNG draws for prefix jitter without prefix commands or physics, then call the
  unchanged placement_endpoints adapter and compare saved endpoints/RNG state.
- A numerical pass would not authorize physics. Independent review and a new
  main-agent/user decision remain necessary; no independent approval fabricated.

## Tasks / verification

- [x] Preserve pre-work source/artifact hash inventory in exclusive output root.
- [x] RED: solvable toy with >0.05-rad parent must yield accepted <=0.05-rad
  children, with parent attempt still present and original endpoint exact.
- [x] RED: unreachable solver case fails finitely, with no subdivision fallback.
- [x] Test attempt/iteration/collision caps, depth/minimum movement stops,
  endpoint ownership, nonfinite q, residual rejection and no hidden attempts.
- [x] Implement planner.py with explicit dependencies, accounted candidate
  journal, immutable inputs and stop-first-failure partial journal.
- [x] Implement diagnostic.py reconstructing terminal full integration state,
  qpos, previous ctrl, measured hand/payload, fixed rotation, seed/jitter and
  original rejection from saved artifacts. Require exact matches before solve.
- [x] Run all 168 existing CPU tests plus all new tests; retain exclusive logs.
  Use runc, no network/GPU, read-only repo, 16 GiB, ephemeral HF/cache paths.
- [x] Self-review code and fresh hashes/tests before the single real attempt.
- [x] Execute the exact path once; persist candidate/segment/collision journals,
  raw reconstructed inputs, solver counts, first subdivision and final gate.
- [x] Verify no live-state mutation and all protected hashes; write REPORT.md
  with measured results and explicit remaining physics/review hold. No retries.

## Planned files

- planner.py: adaptive route, explicit dependency adapter and bounded accounting.
- diagnostic.py: saved-state reconstruction, exclusive numerical invocation,
  preservation/test evidence and report data (no stepping entry point).
- tests/test_planner.py: real toy IK and dependency-boundary regressions.
- tests/test_diagnostic.py: exact reconstruction and exclusive-output behavior.

Testing reconstruction may read the real saved state but must not solve the real
route. The actual route is invoked only by the diagnostic's explicit main entry.

## Completion — numerical gate failed, no retry

168 original +20 new CPU tests passed. One actual saved-state path used 126
attempts, 331 outer update attempts, 456 residual observations and 600 collision
samples. It completed the original rejected segment and 93/113 original waypoints,
then failed ik_backtracking_failure during lowering (position residual 1.707 mm).
No physics, commands, expert calls or commits. Source hashes remain bound to the
test/attempt evidence; 2,272 originally protected files remain unchanged.
See [full report](../../outputs/diagnostics/adaptive_placement_v4_20260917/REPORT.md).

The first launch was interrupted during dependency import before main/marker,
with no numerical attempt. Its log is preserved; unchanged code then ran the
single real numerical attempt detached from the shared terminal. A separate
concurrent v4 package was left untouched; final regression discovery used the
original 168-test hash inventory, not the concurrently added tests.