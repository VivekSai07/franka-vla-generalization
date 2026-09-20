# Bounded placement v3 — implementation / independent review handoff

**Status: code prepared; actual execution blocked pending independent top-agent review.**
No actual v3 expert invocation, placement plan on an authorized saved case,
policy inference, training, collection, installation, network request or commit
was performed in this implementation session. All five authorized trials remain
unconsumed. No independent subagent was available, so self-review is not treated
as the required independent review.

The approved contract is [the unchanged specification](../../docs/superpowers/specs/2026-09-17-bounded-placement-design.md).
The implementation plan is [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Exact new source/test files

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Side-effect-free package marker |
| [solver.py](solver.py) | Box-constrained quadratic optimization, bounded DLS, independent pose residuals |
| [path.py](path.py) | Copied pose waypoints, branch rejection, command bridge, collision-checked full route, time scaling |
| [collision.py](collision.py) | Scratch MuJoCo geometry, correct measured hand-relative payload, contact exceptions, shared sample budget |
| [gates.py](gates.py) | Tracking holds, grasp-loss counter, 1,200-tick deadline, ten-sample release streak, 60-tick settling cap |
| [prefix.py](prefix.py) | Explicit unchanged v2 prefix command calculation and original seeded endpoint draws |
| [expert.py](expert.py) | Single live prefix-to-placement loop, bounded measured release and upward retreat |
| [artifacts.py](artifacts.py) | Exclusive per-tick NPZ/JSON and lossless FFV1 diagnostic videos |
| [replay.py](replay.py) | Float64 saved-control replay, phase/command/readiness/contact checks, decoded video checks |
| [diagnostic.py](diagnostic.py) | Exact five saved inputs, review/test/source admission, exclusive invocation markers, stop-first-failure orchestration |
| [verification.py](verification.py) | CPU-only old/new test execution and protected-file fingerprints |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Test-first implementation tasks and exact contract limits |
| [README.md](README.md) | This handoff and review checklist |
| [tests/test_bounded_placement_v3.py](../../tests/test_bounded_placement_v3.py) | Real numerical/FK/path/toy collision/gate regressions |
| [tests/test_bounded_placement_v3_integration.py](../../tests/test_bounded_placement_v3_integration.py) | Saved-observation prefix, state-machine, media, short replay and orchestration regressions |

No pre-existing tracked file was edited. Pre-existing local offline Docker files
remain untouched and uncommitted.

## Verified evidence

Artifacts are exclusively under
[outputs/diagnostics/bounded_placement_v3_20260917](../../outputs/diagnostics/bounded_placement_v3_20260917).

- **137/137 legacy tests** passed in 34.228 seconds; **24/24 new tests** passed in
  2.130 seconds. No failures, errors or skips. Earlier RED logs are retained.
- Tests ran in image sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
  CPU-only, runtime runc, network none, 16 GiB memory, four CPUs, read-only repo,
  ephemeral /tmp HF_HOME and caches, no host Python or dependency installation.
- The actual coupled quadratic optimum is [0.2, 0.4], not clipped unconstrained
  [0.2, 0.0]. Small real Franka FK goals pass the approved residual/bound gates;
  an impossible goal fails finitely. This is not a saved-case carrying-path test.
- All **197** v2 saved prefix observation states yield bit-exact v3 prefix
  controls, requested/applied velocities, saturation flags, endpoint targets,
  phase indices and phase clocks. Retained RNG state/endpoint draws match.
  **Maximum command difference: 0 rad.** No new prefix was physically executed.
- Real toy MuJoCo checks detect a payload-only mid-segment obstacle; input live
  qpos is unchanged by collision sampling. Collision/contact limits are tested.
- A short **three-control saved-v2 replay** matches float64 states exactly and
  rejects corrupted qpos, controls and clocks. This does not validate the
  unexecuted v3 placement/retreat replay branches.
- Synthetic camera frames round-trip exactly through diagnostic FFV1. This
  format changes no LeRobot dataset format; no dataset was created.
- Read-only configuration/runtime/source preflight passes. **1,784 protected
  source/artifact files** match before/after. Current Python/test source hashes
  match the final test evidence. Old tracked SHA256-list digest remains
  44e65d579978241622c628b2bc782746840de50b83d406d6c903dcb7350a4a5a.

See [tests_passed.json](../../outputs/diagnostics/bounded_placement_v3_20260917/tests_passed.json)
and [review_handoff.json](../../outputs/diagnostics/bounded_placement_v3_20260917/review_handoff.json).

## Actual trial status and comparison

| Quantity | Saved v2 episode 4 | New v3 actual trial |
|---|---:|---:|
| Strict task success | false | Not executed |
| Final requested-pair XY gap | 97.768960794 mm | Not measured |
| Saturation ticks | 169 | Not measured |
| Stable stack ticks | 0 | Not measured |

First blocked gate: **independent_review_required**. No actual v3 planning or
task failure has been measured, and there is no repaired-expert/success claim.
Static endpoint IK does not establish that this local continuous planner can
carry the object to release.

## Reviewer focus / unverified integration

The implementation author checked code and unit results, but the following
must be independently reviewed before the first actual invocation:

1. Exact contract coverage in the full, not-yet-executed Expert.run and its
   placement/retreat replay branches. Unit tests are not end-to-end proof.
2. Lift boundary: the live loop stops prefix after tick 196's original physics
   step. There is no full-v2-then-rewind path. Episode 4's actual sink compares
   its prefix to saved v2 before any placement plan can execute.
3. Hand frame agreement between Pinocchio panda_hand and MuJoCo hand, full
   hand.inverse()*cube transform, frozen destination and original RNG endpoints.
   A predicted release-height conflict raises release_height_contract_conflict;
   it must not be waived or fixed by moving the approved endpoint.
4. Finite projected coordinate QP (fixed damping 1e-3, 2,048 inner sweeps,
   12 backtracking candidates), <=100 outer steps, no seed clipping/restarts or
   home nullspace. Solver failure is local, not physical infeasibility.
5. Enabled MuJoCo filters/exclusions and discrete <=0.005-rad sampling, including
   the previous-target bridge, robot and payload. Runtime contacts are sampled
   at control boundaries; this is not continuous or every-physics-substep proof.
6. Timing interpretation: 90/60/30 closed carrying-route stage minima, at most
   60 additional settling ticks; original 30-tick open phase followed by a
   >=60-tick retreat fits inside the 90-tick post-open window. Tracking lag can
   therefore make retreat fail. No extension beyond that approved window.
7. Planning, source/provenance, replay and task failures all stop the block;
   partial states and failure journals persist. Review nonfinite-state and
   interrupted media/I/O handling as well as normal finite solver failure.

The fixed order is episode **4, 0, 1, 2, 3**, seeds **488840035, 1232395700,
1762113100, 1121959577, 2349681391**. Cases 0..3 cannot run after any preceding
task or integrity failure. No retries, resampling or tuning in the block.

## Launch admission (not executed here)

Importing modules does not launch anything. Only diagnostic.execute_reviewed()
or a separately reviewed equivalent may start actual trials. It requires an
independent top agent to supply an approval artifact in the fixed output root
with reviewer_role=independent_top_agent, approved_for_actual_trials=true and
source_hashes exactly matching the current tested implementation. Do not create
this artifact merely to bypass the review gate. Review records should include
findings and disposition. None exists at this handoff.

The test evidence and protected hashes are also required before the exclusive
execution-start marker is created. Existing markers prohibit restart/resume.
The main agent handles documentation commits; no commit or push is authorized
as part of this implementation session.