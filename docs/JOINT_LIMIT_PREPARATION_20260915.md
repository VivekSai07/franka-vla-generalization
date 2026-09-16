# Joint-limit-aware expert and preparation repeat

User authorized this correction and a fresh 12-slot preparation gate on
2026-09-15. No pilot launch or learned-policy training.

## Correction

The previous expert integrated capped velocities into an unbounded joint target.
The first failing seed commanded joint 4 below its lower actuator limit.
The new bounded integrator constrains each command increment using the
intersection of the actuator and robot position ranges, and the existing
2 rad/s velocity cap. Commands stay 1e-6 rad inside position limits for
float32 recording. Constrained target generation occurs before actuator send;
exactly that sent target is recorded. No after-the-fact label clipping occurs.

This is a command governor, not a constrained Cartesian IK optimizer or proof
of reachable waypoints. Saturation can change Cartesian tracking and success.
It is recorded per tick/per joint along with requested/applied velocities.
The target does not accumulate windup beyond the bounds and can move away
when the requested direction reverses. Invalid prior targets fail closed.

Initialization is an explicit exception: reset controls during the existing
0.2-second settling are retained; the first integrated target uses Q_HOME as
its baseline, not the reset actuator command. The increment bound applies to
the target sequence from that declared baseline, not the reset-to-first-command
transition. The manifest records this exception and baseline. No claim of a
globally velocity-limited startup or hardware safety is made. Changing settling
control is a separate dynamics change, deliberately excluded from this repair.
Validation independently reconstructs the expected robot/actuator limits and
all governor outputs, including tick zero, then checks both saturation summaries.

Acceptance, scene ranges, seed schedules, retry limits and replay tolerances
are unchanged. A saturated trajectory is accepted only if the actual task
passes released-stack criteria and subsequent full replay validation. Failed
tasks retain their slot and use the next planned retry seed.

## Preflight

48 tests passed. The failing seed 4281541513 now completes its 60-tick initial
approach with zero invalid targets and 32 constrained ticks. That preflight
does not establish full-task success; the preparation run must do so.
The original-source snapshot is preserved in
`outputs/benchmark_preparation_20260915/source_before_joint_limits.zip`.

## Planned fresh artifacts

- Plans/banks: `outputs/benchmark_preparation_limits_20260915`.
- Recording: `datasets/franka_pilot_preparation_limits_20260915`.
- Container: `smolvla-pilot-preparation-limits-12`.

Regenerate banks under the corrected source provenance using identical plans
and seeds. Then run the 12-slot collector and the complete pilot validator.
Do not overwrite the earlier three-episode partial recording. Outcomes pending.

## Active handoff

**Completed:** the corrected preparation exited 0 with 12 episodes / 6,336
frames from 16 attempts. The full pilot validator passed with split 9 train /
3 validation, both 6,336-frame video streams decoded and 120 tuning / 240 final
scenes checked. All 12 reference and recorded-action replays passed, reference
drift zero. Alignment PASS: current-state error 0.022387, next-state 0.016457.
The result is persisted in the dataset's `validation_result.json`.
The earlier pending-run text below is retained as execution history.

Both fresh plan/bank sets were generated successfully. Plan hashes are unchanged
from the prior preparation (same conditions/seeds); source/bank provenance is
new. Container `smolvla-pilot-preparation-limits-12` is running, ID
`d45ed4d0126145be07ffe0fa8153d185ab3786559410bf8fb82346f0c37385bb`.
A terminal is waiting on `docker wait`; do not start a duplicate or change
hashed source during recording.

After completion, inspect the real exit code, count and attempt ledger. If exit
is 0 and the manifest is complete, run `validation/validate_pilot.py` on the
new dataset with `outputs/benchmark_preparation_limits_20260915/preparation_scenes`.
Require all quotas, video/row metadata, command reconstruction, replay, timing,
source/runtime and original alignment checks. If any fails, stop and preserve
the output; do not publish a pilot-ready claim or relax tolerances.