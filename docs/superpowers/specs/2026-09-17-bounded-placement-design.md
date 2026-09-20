# Bounded placement expert v3 — proposed implementation contract

Date: 2026-09-17. User approved proceeding with the next expert-placement
priority. This specifies the numerical/path and execution budget for review
before implementation. No corrected trajectory has been executed under v3.

## Goal

Make placement feasible under joint limits and prohibit opening the gripper
before measured placement readiness. Preserve v1/v2 and the failed paired batch.
Fixing endpoint ownership improved some tracking errors but did not yield a
stack; v2 final XY separation was 97.77 mm. Static endpoint IK solutions exist
but are not proof of a valid carrying path.

## Options considered

1. **Selected: bounded waypoint IK with explicit joint-path checks and measured
   endpoint settling.** Plan from the attained lift posture; do not jump to a
   remote endpoint solution. This provides inspectable continuity, bounds,
   payload and collision checks before moving.
2. Constrained differential IK: smaller online controller change, but a local
   solver may still stall in the current posture basin. A genuine bounded
   optimization is required; solving unconstrained DLS then clipping is not
   constrained least squares and largely repeats the existing governor.
3. General joint-space search/RRT: can address wider routing problems, but adds
   scope beyond this bounded placement correction. Not included as an automatic
   fallback. Failure of the selected local planner must be reported honestly.

## Versioning and unchanged contract

New source lives outside the existing paired_demonstrations and pose_snapshot_v2
packages so their provenance stays valid. Old source, partial data and diagnostic
records remain byte-identical. Explicitly reuse the fixed endpoint-copy planner
and existing physical helpers without global monkeypatching or runtime rewriting.

Preserve initial states, RNG seeds, pick/descend/grasp/lift behavior, gains and
timing through the end of Verify Lift. Preserve 30 Hz control, 600 Hz physics,
20 substeps, intersected URDF/actuator position bounds, command-governor 1e-6 rad
interior margin, 2 rad/s command-increment limit and gravity compensation.
No changes to cube dimensions/friction, cameras, geometry, action/state schema,
post-release success definition or workspace. All operations are simulated only.

## Placement path

At successful lift completion, capture the actual full state and copied hand
pose. Measure hand-to-held-cube transform for payload collision diagnostics;
this is privileged expert planning information, never a learned-policy input.

Use the existing destination geometry, fixed hand orientation and seeded phase
jitter. Consume random draws in the same order; retain the original transport,
lower and unjittered release endpoints. Plan the full carried-object route to
the unjittered release pose, not independent unconnected endpoint solutions.

1. Interpolate copied hand-pose waypoints from measured lift pose to transport,
   lower and release endpoints, with translation increments no larger than
   5 mm and rotation increments no larger than 0.02 rad. Cap at 400 waypoints.
2. Solve each waypoint from the preceding accepted joint posture using bounded
   damped least squares with a finite active-set/projected solver and backtracking.
   Max 100 iterations/waypoint. Cap per-iteration joint displacement at 0.05 rad.
   Do not add the old unconstrained home-nullspace velocity afterward.
3. Require independent FK position error <=1 mm and orientation error <=0.01 rad.
   Require consecutive waypoint joint displacement <=0.05 rad per joint. A
   solver iteration limit, residual failure or branch jump stops planning;
   do not silently relax residuals or try random restarts.
4. Use measured joints as the path's geometric start, but retain the previous
   actuator targets as the command-increment baseline. Explicitly validate the
   bridge between previous targets and first commanded waypoint; no target jump.
5. Check path segments at <=0.005 rad per-joint increments against enabled
   MuJoCo collision geometry, with at most 10,000 checked samples total. Check
   both robot geometry and the cube transformed by its measured hand-relative
   pose. Gripper/held-cube contact is allowed; other penetration beyond 1e-5 m
   is rejected. Intended held-cube/base contact is permitted only during final
   lowering/settling and only with the same penetration tolerance. Report
   collision-filter and discretization limitations; not a continuous proof.
6. Time-parameterize the accepted continuous joint path at 30 Hz so all command
   increments respect 2 rad/s and endpoints are not crossed by interpolation.
   Preserve original phase durations as minima; permit explicit extension
   within the total deadline rather than pretending all demos remain 528 frames.
   If the path cannot meet the bound within the deadline, fail without executing
   an unchecked route.

The planner may fail to find a route even if one exists. That is a local
planning failure, not proof of physical infeasibility. No general-search or
orientation-relaxation fallback is authorized by this design.

## Execution and release gate

Total episode deadline: **40 simulated seconds / 1,200 ticks**, including the
unchanged pick/lift prefix and post-release scoring. No clock reset between
phases. Track measured state and actual contacts throughout execution.

- Require tracking error <=0.05 rad per arm joint before advancing the commanded
  waypoint; hold the current bounded target if lagging. Reject nonfinite values,
  unexpected collision, or loss of bilateral grasp sustained for ten ticks
  during transport before intentional release. Log any brief contact loss.
- On arrival, hold the final joint target for at most 60 additional control ticks,
  subject to the total deadline. No indefinite convergence wait.
- Require all pre-release readiness conditions for ten consecutive ticks:
  requested cube remains grasped; cube XY offset from base <=0.01 m;
  cube-center Z difference between 0.04 and 0.09 m; Euclidean hand endpoint
  error <=0.005 m; hand orientation error <=0.03 rad; maximum absolute measured
  arm velocity <=0.10 rad/s; no unexpected penetrating contacts.
- The pre-release height range permits the original elevated release geometry.
  It is **not** the post-release stack test and does not imply a stable stack.
- Open only after the gate passes. Otherwise stop as placement_not_ready while
  retaining the closed command; do not release merely because a timer elapsed.
- After opening, use a bounded upward retreat from the attained posture with
  the same path/bounds/collision checks, rather than an unchecked Cartesian
  retract from the original failed loop. Keep a maximum 3-second observation
  window and require the unchanged strict requested-pair stack for 30 ticks.
  Do not count reaching the hand target, reduced saturation, a wrong-color
  stack or a 7.5 cm miss as success.

This changes the expert controller/path and phase progression, not the policy.
Measured contact-based transitions may yield variable episode lengths. The old
528-frame-only paired collector/validator must not be used unchanged for a
future v3 dataset; adaptation would be a separately versioned collection stage.

## Tests and bounded verification

Unit-test endpoint snapshots, constrained-solver bounds/residuals, branch-jump
rejection, command-bridge bounds, time scaling, collision rejection with payload,
release predicate boundaries/streak reset, total timeout and no-open-on-failure.
Compare the v2/v3 prefix through lift end for same state/commands/random draws.
Check all legacy tests without GPU/host installs. Existing NumPy/Pinocchio/MuJoCo
are sufficient; no dependency or optimizer package installation assumed.

Maximum **five actual expert trials**, each once, no retries or resampling:

1. Saved failed episode 4 / layout 02 / seed 488840035.
2. Only if trial 1 achieves strict success and all integrity checks pass, run
   saved episodes 0, 1, 2, 3 (both directions of two red/green layouts), in order.
3. Stop at the first failed planning, task or integrity gate. Do not tune the
   parameters and retry in the same authorized block.

Save plan candidates/residuals, command/measured states, contacts, gate decisions,
all raw actions, phase clocks, videos and terminal state. Validate recorded
commands and reference replay, count actual substeps, and hash old files before
and after. Success requires **5/5 instructed released stacks**, no unexpected
collisions or unbounded commands, no release without readiness, and replay
agreement. This is a small regression gate, not reliability across all 18 layouts.

## Exclusions and follow-up

No new demonstrations in LeRobot, full 36-demo retry, training, policy rollouts,
final-scene access, changed joint limits, changed success rules or uploads.
On 5/5 pass, review evidence and propose a fresh versioned paired collection;
do not launch it automatically. On failure, identify the first failing gate
and preserve the result, without interpreting improved intermediate metrics as
a repaired expert. Local commits only; no push without explicit permission.