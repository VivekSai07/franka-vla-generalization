# Bounded placement v3 Implementation Plan

> **For agentic workers:** Execute inline using writing-plans/TDD; independent top-agent review is required before any actual expert invocation. No subagent tool is available in this session.

**Goal:** Implement the exact approved bounded-placement contract without changing any existing source, spec or artifact.

**Architecture:** Separate finite numerical optimization, geometric/collision path validation, command scheduling/release predicates, unchanged-prefix control, and diagnostic persistence. Reuse physical helpers and the v2 snapshot planner explicitly. Never execute a full v2 episode and rewind.

**Tech Stack:** Existing NumPy, Pinocchio, MuJoCo, unittest, CPU-only pinned Docker. No installs.

## Global Constraints

- Approved source of truth: docs/superpowers/specs/2026-09-17-bounded-placement-design.md; do not edit it.
- New package and tests only; exclusive artifacts under outputs/diagnostics/bounded_placement_v3_20260917. No commits, training, collection, network, credentials or host Python.
- Pinned image sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc, runtime runc, network none, GPU devices absent, ephemeral HF_HOME. Tests use 16 GiB memory or more.
- 30 Hz / 600 Hz / 20 substeps, intersected bounds with 1e-6 rad interior margin, 2 rad/s commands, gravity compensation unchanged.
- <=5 mm and <=0.02 rad pose spacing, <=400 waypoints; <=100 IK iterations/waypoint; <=0.05 rad/iteration and consecutive waypoint; independent FK <=1 mm and <=0.01 rad.
- True box-constrained quadratic subproblem, finite projected iterations and backtracking, no solve-then-clip, no home-nullspace term, no restarts.
- Measured geometric start and previous-target command bridge checked separately. Collision sampling <=0.005 rad/joint, <=10,000 total samples, robot plus hand-relative payload, 1e-5 m penetration tolerance and stated contact exceptions only.
- Preserve seeded transport/lower endpoints and unjittered release, original phase durations as minima. Stop on release-height contradiction, not a changed endpoint.
- <=1,200 total ticks including prefix/scoring; final settling <=60 additional ticks; readiness 10 consecutive ticks: grasp, XY<=.01 m, Z .04..09 m, hand error<=.005 m, orientation<=.03 rad, arm speed<=.10 rad/s, no unexpected penetration.
- Tracking <=.05 rad before advancing; hold if lagging; fail after ten consecutive grasp-loss ticks before release; bounded retreat; <=90 post-open observation ticks; unchanged strict stack for 30 ticks.
- Actual cases 4(seed488840035),0,1,2,3, once each; stop first planning/task/provenance failure. No actual trials before independent review.

## Task 1: Numerical and geometric kernels

Create solver.py, path.py and tests/test_bounded_placement_v3.py.

- [ ] RED: coupled bounded quadratic with independently derived optimum [0.2, 0.4], unlike clipped unconstrained [0.2, 0]. Real Franka FK target, impossible target failure, branch-jump rejection.
- [ ] Implement projected coordinate minimization of the damped quadratic to KKT tolerance with a finite cap. Solve waypoint from previous accepted posture only; journal every candidate and residual.
- [ ] Implement copied pose spacing, waypoint-count limit, independent residual/branch gates, bridge and endpoint-preserving time scaling.
- [ ] GREEN: execute the real solver and geometric tests in Docker; no actual expert stepping.

## Task 2: Collision and state gates

Create collision.py and gates.py; extend the same test file first.

- [ ] RED: real toy MuJoCo scene with payload-offset obstacle, mid-segment collision, penetration/contact phase exceptions, sample exhaustion; readiness boundary/streak reset, timeout, no opening on failure.
- [ ] Implement scratch-data FK collision checks with measured hand.inverse()*cube pose; never modify live data. Record enabled filters/exclusions and discretization limitations.
- [ ] Implement finite command advancement, tracking holds, release streak and deadline as explicit state rather than elapsed-time opening.
- [ ] GREEN: verify payload placement and source-data immutability, per-joint spacing, finite checks and stopping behavior.

## Task 3: Prefix and expert integration

Create prefix.py, expert.py and tests/test_bounded_placement_v3_integration.py.

- [ ] RED: evaluate prefix control against saved v2 observations (no simulation or actual expert invocation), matching all 197 commands/targets/phases and RNG endpoints exactly.
- [ ] Extract explicit v2 prefix command logic, retaining its initialization, call order, target baseline and transition-before-step behavior. Stop after final Verify Lift step, without rewind.
- [ ] Plan full carried route from attained lift, execute only checked/time-scaled targets, retain closed target on failure, bounded measured release/retreat/scoring.
- [ ] Save plan failures even before placement executes. Validate full state, float64 controls, image boundary, clocks and substeps in reference replay.

## Task 4: Diagnostic authorization, evidence and review handoff

Create diagnostic.py, verification.py and README.md; diagnostic guard tests first.

- [ ] Bind exact five saved attempts and source/runtime provenance; no resampling and no alternate input flags. Exclusive invocation marker before each attempt, stop first failure.
- [ ] Save ticks, full-state terminal records, contacts, gate/phase events, candidate/residual journal and camera videos. Compare v2 baseline and first failed gate, never infer repaired stacking from intermediate metrics.
- [ ] Run all 137 legacy tests plus new tests CPU-only. Save test results and source hashes, compare protected baseline hashes before/after.
- [ ] Independent top-agent review of current hashes and full implementation before launch. This session must return code/evidence for that review rather than self-authorizing trials.

## Review checklist

Verify limits against approved spec, correct hand frame, continuous route/command bridge and payload collision sampling, prefix observational equality, no timer release, finite retreat, failure artifacts, no invocation in tests, no overwrites/retries, exact saved failed case first. Unit tests do not establish actual closed-loop success or replay agreement. Main handles any documentation commit.

## Implementation checkpoint — 2026-09-17

- [x] Tasks 1/2 kernels implemented test-first; real quadratic, Franka FK, toy MuJoCo payload collision and release/clock regressions pass.
- [x] Task 3 explicit prefix/control integration implemented; saved-observation comparison passes all 197 ticks with zero command difference and matching retained RNG. No live prefix/expert was executed.
- [x] Task 4 exact case/hash bindings, stop-first-failure orchestration, durable artifacts and replay code implemented. Configuration-only preflight passes.
- [x] Final pinned CPU suite: 137 legacy +24 new pass, zero skips; 1,784 protected source/artifact hashes unchanged; new sources match saved test hashes.
- [ ] Independent review of whole expert/orchestration and v3 placement/retreat replay branches. No subagent available; README provides top-agent handoff and review focus.
- [ ] Actual expert trial 4, then only on strict task/integrity pass cases 0..3; no budget consumed here.
- [ ] Actual v3 full-state/reference replay and quantitative task comparison. Existing short saved-v2 replay test is not an actual v3 replay gate.

The unchecked implementation steps above remain as task-level review prompts;
this checkpoint records what was actually verified. Do not infer end-to-end
completion from unit tests. First admission blocker is independent_review_required.