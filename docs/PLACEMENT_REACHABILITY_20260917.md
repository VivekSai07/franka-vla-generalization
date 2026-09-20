# Failed paired-expert placement: reachability investigation

## Scope

User authorized investigation of the exact fifth failed demonstration, not
source corrections, retries or new training. All analysis ran in CPU-only
offline Docker. Original source and paired artifacts remained unchanged.
Detailed numerical methods and evidence are in the local
[diagnostic report](../outputs/diagnostics/placement_reachability_20260917/REPORT.md).

## Findings

### 1. The exact placement endpoints have bounded IK solutions

A fixed search of 12 starting postures for each of the three recorded targets
found valid poses for transport (9/12), lowering (8/12) and release (8/12).
All 25 successful attempts were independently checked with MuJoCo FK: maximum
position residual 8.05 micrometres, orientation residual 1.34e-5 radians.
These are successful attempts, not necessarily distinct solution branches.

Static collision checks found no non-grasp robot penetration in 23/25 accepted
attempts. Two self-intersected and cannot be treated as valid robot postures.
Endpoint feasibility does not prove a continuous collision-free, velocity-feasible,
grasp-preserving path, especially within the original phase durations.

### 2. The trajectory start pose is mutable: confirmed implementation defect

`FrankaPanda.forward_kinematics()` returns the live Pinocchio frame placement;
`TaskSpaceTrajectory.__init__()` stores it directly rather than taking a snapshot.
Later FK updates mutate the object retained by the trajectory. Its interpolation
start therefore moves instead of remaining the original phase-start pose.

An independent two-FK-call reproduction confirmed that the stored trajectory
position at t=0 moved **0.0613016 m** after changing joint 1 by 0.2 rad and
calling FK again. No trajectory execution or source modification was required.

Reconstructing the failed trace with previous-tick FK as interpolation start
reproduced all requested joint velocities exactly across 528 ticks. A fixed
phase-start reconstruction differed by up to 3.813442 rad/s. This establishes
the observed aliasing behavior, not that removing it alone will solve placement.

### 3. Position-limit saturation and timed release explain the observed miss

At release onset, the hand was **59.839 mm** from the phase endpoint with
**0.215125 rad** orientation error. The held cube was **75.234 mm** off laterally.
The controller still requested motion into joint 4's lower position limit;
the bounded command governor correctly prevented that motion. This was not
the controller's +/-2 rad/s velocity cap limiting the saturated interval.

The cube sits roughly 101 mm along hand-local Z, so hand orientation errors
also contribute to lateral cube-placement error. The recorded release occurred
on time, not after confirming convergence to the destination.

## Next recommended correction, requiring approval

First fix the independently reproduced pose-aliasing defect in an explicitly
versioned planner/paired-expert path with a regression test for immutable
trajectory endpoints. Preserve the old source version and its provenance.
Do not modify the old benchmark in place and rewrite hashes to retain a PASS.

Test only that change on this same saved failed layout, preserving controller,
timing, seeds, limits and scoring. If it still saturates, then evaluate a
joint-limit-aware trajectory/posture solution as a separate intervention.
Add placement-convergence gating before release as an explicit later expert
behavior change; the guard can prevent a bad release but cannot solve a stuck path.

No new 36-demo batch is authorized by successful static IK. No resampling or
resume occurred, and the four saved demos plus failed trace remain preserved.
The historical 420-demo action labels/replay remain valid evidence of what was
actually commanded; the newly discovered planner defect affects intended
trajectory semantics and expert reliability, not proof that those stored labels
were fabricated. Any new planner version needs new validation and provenance.