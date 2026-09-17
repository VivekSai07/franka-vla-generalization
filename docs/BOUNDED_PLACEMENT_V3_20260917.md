# Bounded placement v3: implementation and first-case stop

The user approved the bounded waypoint/release-gate specification. Implementation
is isolated under experiments/bounded_placement_v3; old expert, planner, data and
model sources were not changed. No collection/training or policy rollouts.

## Verification before execution

137 legacy plus 31 new CPU tests passed: **168 total**, no errors or skips.
Independent review identified a nonfinite failure-reporting bug; corrected before
execution with seven regression tests. Raw terminal arrays are now preserved
before JSON, and summaries mark nonfinite values without weakening rejection.
Independent re-review checked the fix and current hash-bound evidence, approving
only the bounded diagnostic, not claiming task success.

All 197 saved v2 prefix observations reproduced identical requested commands,
targets, phase clocks and RNG state in the prefix comparison. This test is distinct
from the subsequent actual prefix execution. Review/test artifacts remain in
[diagnostic output](../outputs/diagnostics/bounded_placement_v3_20260917/REPORT.md).

## Actual execution: one case, no retry

Container `smolvla-bounded-placement-v3-five`: **exit 1**, no OOM.
The first case was saved failed episode 4, seed 488840035, same layout 02.
The expert grasped and lifted red, then stopped before transport because the
waypoint planner rejected a **0.0697633 rad** maximum per-joint waypoint
displacement against the approved **0.05 rad** continuity limit.

Failure label: `waypoint_branch_jump`. This label denotes the implemented
displacement gate; it does not prove a topological IK-branch change or physical
infeasibility. A 5 mm Cartesian waypoint can require more joint motion near an
ill-conditioned configuration. The accepted endpoint IK studies do not establish
that this particular discretization/path succeeds.

- Exactly **one actual expert invocation**, **197 control ticks**, **3,940
  physics substeps**. Cases 0–3 were not invoked.
- No rejected placement waypoint executed; no intentional release.
- At stop the requested cube remained in finger contact, about 0.21724 m above
  the destination center, XY separation 0.40608 m: it is still at lift completion,
  not a placement attempt that missed by 40 cm.
- Saved-control reference replay: maximum qpos error **0**, full integration
  states exact. Both camera videos contain 197 frames.
- Protected source/artifact hashes unchanged. The old partial dataset stays
  failed and not training-ready.

Evidence: [first failure](../outputs/diagnostics/bounded_placement_v3_20260917/episode_000004/failure.json),
[integrity](../outputs/diagnostics/bounded_placement_v3_20260917/episode_000004/integrity.json),
[execution summary](../outputs/diagnostics/bounded_placement_v3_20260917/execution_summary.json).

## Interpretation and next decision

The v3 safety/continuity stop worked, but **placement is not fixed and the
five-case acceptance gate did not pass**. Dynamic placement, release readiness
and retreat were not reached and cannot be claimed verified end-to-end.

Do not raise the joint displacement limit, disable collision checks or retry
silently. Recommended next design amendment: bounded adaptive waypoint
subdivision when a Cartesian segment requires too large a joint step, preserving
the existing joint-step, velocity, collision and total-compute/time limits.
Check the saved rejected segment numerically first; subdivision may still fail
if the local route/posture is unsuitable. Failure would justify reconsidering
the path/posture planner rather than accumulating arbitrary tolerance changes.

This is a proposed follow-up, not executed or authorized by the earlier
stop-on-first-failure block. No additional trial, recollection or training started.