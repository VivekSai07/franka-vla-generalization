# Adaptive placement v4 — numerical result

User approved bounded adaptive waypoint subdivision, tested numerically first.
No physics stepping, expert invocation, policy rollout, collection or training
was performed. The saved v3 lift state and original placement endpoints were used.

## Result

**The original continuity rejection was resolved, but full-path planning failed.**

- Original rejected joint increment: 0.06976332094 rad, exceeding 0.05 rad.
- Adaptive Cartesian subdivision completed that original segment; rejected
  parent solutions were never used as subsequent IK seeds.
- Completed **93 of 113** original waypoints. These are original path waypoint
  counts, not a measured percentage of Cartesian distance or task completion.
- The next original waypoint, during lowering, failed `ik_backtracking_failure`.
- Failure position residual **1.7071095 mm** exceeds the unchanged 1 mm gate;
  orientation residual **0.00544553 rad** is within its 0.01 rad gate.
- Joints 1 and 3 reached interior bounds at this local failure. This is no
  longer simply joint 4's original command saturation.
- Maximum accepted consecutive joint displacement: **0.04911981 rad**.
- No complete path, command schedule or physics-success result was produced.

## Bounded work and interpretation

| Budget | Actual | Limit |
|---|---:|---:|
| Solver attempts, including rejected parents | 126 | 400 |
| Outer IK update attempts | 331 | 40,000 |
| Maximum outer updates in one solve | 16 | 100 |
| Collision samples | 600 | 10,000 |
| Subdivision depth | 2 | 8 |

Attempt breakdown: 109 accepted, 16 subdivided parents, one failed. The final
failure occurred because all 12 local backtracking candidates failed descent,
not because a global budget was exhausted. The saved `residual_evaluations`
counter counts 456 logged outer observations, **not** every function evaluation;
independent review counted 925 pose-error calls including backtracking and
post-solve checks. No bounded outer-update undercount was found.

All 600 sampled configurations across 110 checked segments, including the
previous-command bridge, stayed within the specified sample resolution (maximum
0.00499835 rad per joint). Independent review reclassified 21,600 saved contact
records and found the existing collision exceptions unchanged. These are
discrete geometric samples, not proof of collision safety at every physics
substep or along a continuous trajectory. No dynamics were executed.

## Verification and evidence

168 existing +20 new tests passed; main rerun after cleanup: **188/188** in
30.889 seconds. Independent read-only review checked source, numerical journal,
state/endpoints, candidate sequences, collision records and hash inventories.
All 2,272 originally protected files and four tested/executed Python files
matched recorded hashes. A current regression pass does not make the failed
real numerical route a passing test.

Retained implementation: `experiments/adaptive_placement_v4`.
Evidence: [full numerical report](../outputs/diagnostics/adaptive_placement_v4_20260917/REPORT.md),
[verification](../outputs/diagnostics/adaptive_placement_v4_20260917/final_verification.json).

An assistant orchestration mistake dispatched two overlapping workers. Only the
retained placement implementation executed a numerical route. The unused
`adaptive_waypoints_v4` source/test package was removed before commit; all its
diagnostic outputs remain preserved. References in those unused historical
outputs to removed source are not an executable verification record. They are
not counted as another real route attempt or included in the 188-test claim.

## Decision

Do not loosen the 1 mm residual threshold, increase budgets, interpolate through
rejected configurations or rerun expert collection. Subdivision fixes excessive
joint steps caused by discretization but does not supply a different redundant
posture or a route around an unsuitable local IK basin. Failure here is not proof
that the target or all paths to it are infeasible.

The accumulated evidence now warrants reconsidering **route/posture planning**,
not another incremental tolerance change. A next separately approved design
should connect the attained lift posture to a collision-checked alternative
placement posture, considering the carried object and command/time constraints.
Candidate paths may require moving away from the direct Cartesian interpolation.
Keep release gating and all existing bounds; require a complete numerical path
before any new dynamic verification. No particular planner or route has been
shown successful, and no further experiment was launched.