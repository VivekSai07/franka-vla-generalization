# Pose-copy fix and assessment of external advice

## Verified correction and remaining failure

The user authorized the versioned pose-copy correction and exact-layout test.
New `experiments/pose_snapshot_v2` code snapshots both SE3 endpoints and retains
the original interpolation. The copied expert loop differs by one constructor
substitution; configuration differs only by explicit expert version. Original
bound planner, robot, expert and dataset files remain unchanged.

129 existing and eight new CPU tests passed. Exactly one corrected scripted
expert execution used failed episode 4, seed 488840035 and saved layout 02.
The container exited 1, no OOM: **the alias defect is fixed, task success is not**.

| Metric | Original | Pose-copy version |
|---|---:|---:|
| Frames | 528 | 528 |
| First bilateral contact tick | 110 | 110 |
| Maximum lift | 239.14 mm | 239.51 mm |
| Joint-4 saturation ticks | 202 | 169 |
| First saturation tick | 233 | 266 |
| Release tick | 347 | 347 |
| Release-onset hand endpoint error | 59.84 mm | 47.51 mm |
| Final cube XY separation | 80.95 mm | 97.77 mm |
| Stable-stack ticks | 0 | 0 |

Saved-control replays of both versions matched exactly, including terminal
integration states. All corrected commands were bounded and finite, with 20
substeps per tick. All 193 enumerated old files and 1,090 paired artifacts stayed
unchanged during execution/review. No second correction, retry, collection or
training followed the failure. Earlier test-sandbox failures (memory limit and
read-only cache setup) are preserved separately and were resolved without
changing old tests or claiming them to be successful runs.

Evidence: [detailed execution report](../outputs/diagnostics/pose_snapshot_v2_20260917/REPORT.md),
[test evidence](../outputs/diagnostics/pose_snapshot_v2_20260917/tests_passed.json),
[execution record](../outputs/diagnostics/pose_snapshot_v2_20260917/execution_exit.json).

## Assessment of the supplied ChatGPT discussion

The discussion was based on an abbreviated project description. Its hypotheses
must be updated with our actual measurements; they are not additional evidence.

### Points that apply

- Equal 30 Hz rates do not make datasets equivalent: command semantics,
  observation alignment, controller, camera setup and task coverage matter.
- Behavior cloning can compound small errors on policy-induced states absent
  from demonstrations. Recovery coverage is a plausible missing capability.
- Fixed phases and timing can allow shortcuts. All 300 pilot training demos
  used gripper transitions at frames 107/347, but the model is not explicitly
  given a clock or expert phase. It can infer phase from visual/proprioceptive cues.
- Task complexity and data needs cannot be inferred from ten successful SO101
  demos on an unspecified task. Paired roles/distractors add ambiguity.
- Failure-stage diagnosis is more useful than overall loss alone. Our 120-trial
  audit found 110 without bilateral target contact and eight wrong stable stacks.
- Wall-clock training duration alone is not a learning budget. Actual pilot:
  20,000 updates, effective batch 32, about 4.45 passes, 300 training demos;
  total container runtime about 4 h 51 min, not seven hours.

### Claims to qualify or reject

- A state machine is not inherently open-loop. Our expert uses measured joint
  state, feedback IK, object poses at phase start and a lift/contact check.
  Its problematic phase progression and release remain time-driven.
- Human teleoperation does not guarantee useful recovery examples, and smooth
  scripted trajectories are not intrinsically bad. Inspect what corrections
  were actually demonstrated rather than assuming one collection method wins.
- The controller is not demonstrably "too good": the fifth paired task fails
  even after the alias correction. Earlier recording selected successful
  attempts; an accepted-success dataset does not prove a universally reliable expert.
- Observation/action timing and normalization are important generic suspects,
  but our current data passed direct replay, timestamp and saved-processor checks.
  Those tests do not prove universal correctness, but repeating generic checks
  or recalculating correct statistics is not the leading next intervention.
- Cartesian expert planning with joint-target labels is valid if labels are the
  commands actually sent and deployment executes the same interface. Changing
  to EEF actions is a separate architecture/benchmark experiment, not a necessary fix.
- Two cameras can be tested, but no evidence identifies the second camera as
  harmful here. Motion blur is not established in our instantaneous MuJoCo renders.
- Adding perturbations before confirming expert recoverability risks collecting
  failed or misleading labels. Geometry-feasible endpoints do not establish
  safe, velocity-feasible, grasp-preserving paths.
- The actual broad pilot covered 12 ordered color pairs, not just two tasks;
  the current paired experiment covers six unordered pairs, two cubes at a time.
  The RTX 6000 Ada is Ada-generation hardware, not Blackwell. Neither correction
  changes the observed task failure, but accurate context matters.

## How to tackle the current failure

Separate two problems:

1. **Expert reliability:** valid joint-limited poses exist, but current tracking
   saturates joint 4 and releases off-target. Pose copying alone did not solve it.
2. **Learned policy:** wrong object roles, missed grasps and poor seen-scene
   reproduction persist. Reliable expert data is necessary but not proven sufficient.

Recommended next bounded expert design, requiring approval before execution:

- Keep the pose-copy version; preserve the original failed run as evidence.
- Plan a continuous joint-limit-aware transport/lowering trajectory, not merely
  a valid isolated IK endpoint. Check joint position/velocity bounds, collisions,
  orientation and the held cube's offset. Consider posture selection and warm-
  started waypoint IK or constrained differential IK as alternatives to compare
  during design; do not blindly add a home-posture gain or lengthen all phases.
- Advance to release only after measured placement/grasp conditions are satisfied,
  with a bounded timeout that fails closed. This guard prevents invalid release
  but is not itself a reachability solution. Thresholds must account for hand-to-
  cube offset and intended pre-release height, not reuse the post-release stack
  criterion while still holding the cube.
- Test this exact saved failure first and then both directions on the two
  previously successful red/green layouts. Count this as a new explicitly bounded
  expert verification, not training data or a resumed partial collection.
- Only after that passes, propose a fresh versioned paired batch with an honest
  same-layout reuse/provenance policy. Do not rewrite duplicate checks or resample
  difficult layouts silently to obtain a pass.

After the expert is reliable, execute the already designed small paired/control
learning comparison. Consider additional recovery supervision only if the failure
analysis supports it, and keep it separate from the paired-data comparison.

For recovery data, distinguish the deliberately perturbed behavior action from
the corrective expert target at the resulting observation. Preserve both in an
explicit contract; blindly storing noisy commands as "expert" labels teaches
noise, not necessarily correction. Position perturbations must be kinematically
consistent and collision-checked, not arbitrary teleports during a grasp. Split
layout/perturbation families together and retain all failed-attempt accounting.

No new recovery, camera, action-space, training or recollection experiment is
authorized by this report. No promise of full success follows from the generic
advice or the pose-copy correction. Use measured closed-loop gates, not loss alone.