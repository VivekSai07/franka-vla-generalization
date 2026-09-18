# Fixed-layout stacking baseline

Date: 2026-09-18

## Direction and first-stage scope

The user approved the fixed-position approach on 2026-09-18 after discussing
their SO101 experience. It supersedes the unexecuted 12-layout proposal.
This written specification fixes the proposed initial data budget for review:
**24 repeated expert executions, 20 training and 4 trajectory-validation episodes**,
all from one identical saved reset. No training budget is authorized by this
data-stage specification. Choose that separately after examining the collected
data, duplicate trajectories and available storage.

Goal: establish a reliable, reproducible fixed-layout data source before training
a learned policy to stack at that one arrangement. Neither the data gate nor
later fixed-reset policy success establishes spatial or language generalization.

## Scene selection and limits

Use pilot training episode 7, selected because the old recorded expert and both
single-task policies succeeded there. This selection is intentionally optimistic
and must be disclosed; it is not a representative or unbiased benchmark sample.
The original expert succeeded once on this layout; repeat reliability with the
pose-snapshot expert is unmeasured.

- Source: `datasets/franka_pilot_20260915/replay/episode_000007.npz`.
- SHA256: `ca5c19eda5d089a4a33a838a3fc3e55b44895f77aa40e8d92d9490953290221b`.
- Source scene hash: `88060815ffd36e578284aeda158cbaac38e0149db398f09c4cf22bad48150efd`.
- Red settled XYZ: (0.2820318976205449, -0.38303709520664875, 0.020994666209306827) m.
- Green settled XYZ: (0.2676253941656954, 0.3512266692018041, 0.020994666209306827) m.
- Restore the complete saved initial integration state for every attempt; do
  not respawn, resettle or randomize object/robot positions. Verify reset qpos,
  controls and derived body/camera geometry against source data before recording.
- Use world coordinates rather than ambiguous camera-relative left/right labels.
- Other objects, camera poses, lights, physics and robot start remain unchanged.
- This deliberately duplicates a known training geometry. Use a separate
  fixed-layout validator, not a bypass of older distinct-layout validators.
- No old held-out scene or sealed final scene is used. Four new validation
  episodes share the training geometry and are **not spatial holdouts**.

## Expert and data contract

Use the existing pose-snapshot-v2 expert unchanged. Keep timed phases, IK,
gripper schedule, bounded joint governor and its existing 5 mm Gaussian
trajectory-waypoint jitter. Only the expert RNG seed changes across episodes;
object positions and starting robot state do not. This preserves modest execution
variation without introducing spatial-layout randomization. Gaussian jitter is
not bounded to +/-5 mm. It acts in the existing pick/place waypoint phases.

Instruction: "pick up the red cube and place it on top of the green cube".
State9 measured joints/fingers; action8 absolute joint commands plus gripper
0 closed / 255 open. Observe-command-record alignment, two 512x512 RGB cameras,
AV1/yuv420p, 30 Hz control, 600 Hz physics / 20 synchronized substeps remain.
Ground-truth poses, contacts and phases belong only in audit sidecars.

Master seed 20260918, namespace `fixed_layout_baseline_v1`. For episode index i,
derive seed from first four SHA256 bytes, big endian, of
`fixed_layout_baseline_v1:20260918:expert:i` where i is decimal without padding.
Freeze all 24 seeds and IDs before any trajectory. Train IDs0-19, validation20-23.
The first three training IDs are the preparation subset, counted in the 24 total.

Store independent genuine executions, never copied recordings. Report hashes of
every action/state sequence and pairwise action RMS differences. If exact duplicate
trajectories occur, disclose them and stop before training-design decisions;
do not claim that episode count alone adds new information. Preserve all outputs.

## Budget and gates

1. CPU software tests and provenance/reset-only checks first; these do not run
   an expert or policy. Restore/checking state must not advance physics.
2. Collect first three executions, validating each before the next. If all pass,
   finish the remaining 21 under the same frozen plan without a new budget.
3. Maximum 24 expert executions, one per slot. Stop on first expert, media,
   integrity or replay failure. No retry, alternate layout, new output relaunch,
   changed seed, planner repair or relaxed threshold without a new decision.
4. Exactly two validation replays per completed successful execution: reference
   controls and persisted float32 commands. Do not re-run the expert as a third
   replay or accidentally exceed the expert-attempt cap.
5. Maximum three hours wall time for recording/validation. Minimum 30 GiB free
   before collection and checked between episodes. If insufficient, block/stop;
   do not delete old artifacts or automatically reduce the reserve.
6. Separate admission: >=30 consecutive closed-command ticks; >=5 consecutive
   post-step bilateral target-contact ticks; target lift >50 mm while bilateral
   contact is present; final strict released stack for >=30 consecutive ticks.
   These contact/closure thresholds are operational data gates, not universal
   definitions of a valid grasp.
7. Strict stack geometry unchanged: XY<20 mm, relative Z35-45 mm, gripper open,
   no top-cube finger contact. Reference replay full-qpos tolerance1e-5 and
   float32 replay policy-state tolerance1e-3 remain; both must stack successfully.
8. Validate persisted action/state rows, instruction, timestamps, finite values,
   command reconstruction, 528-frame episode length, all video frames and source/
   runtime/data bindings. Correct hand-origin geometry for distance reporting.

First-three success is expert data readiness only. It does not establish learned
policy success. The full data gate requires 24/24 and 48/48 validation replays.
No automatic model training follows collection. Later policy evaluation should
use predeclared distinct sampling seeds at this exact reset and describe the
limited source of variability, not count deterministic repeats as independent
physical-world evidence. Train-only normalization must exclude IDs20-23.

## Implementation and failure isolation

New package: `experiments/fixed_layout/`, tests within that package. Reuse small
existing helpers and the unchanged expert, not old hardcoded paired scheduling.
Keep all old provenance-bound source/report bytes unchanged. Preserve the original
single-scene overfit failures; this experiment is not a relabeling of them.

Output root: `outputs/fixed_layout/fixed_layout_20260918`, exclusive creation,
no resume/overwrite. Freeze source reset, plan and source/runtime inventory before
execution. Save per-attempt raw observations, actual commands, pre-step full state,
terminal state and failure reason, including incomplete/failed attempts. Final
dataset and report may be marked ready only after all checks. Never insert failed
attempts as successful imitation targets or replace a failed slot silently.

Run all Python inside the existing pinned Docker image, CPU/runc/OSMesa,
network disabled. Repo and existing data read-only; only the new output root
writable. No GPU or model loading needed for this stage. Do not touch unrelated
containers. Preserve secrets; do not read environment credential files.

## Review and current resource status

- [x] Conceptual fixed-layout direction approved by user.
- [x] Existing source episode and both policy success records inspected read-only.
- [x] Scope separated from spatial generalization and future training.
- [x] Self-review: budgets, seeds, split, duplicates and stop rules explicit.
- [ ] User review of this exact written data-stage budget and implementation mode.
- [ ] Implementation plan, tests and independent source review.
- [ ] Collection storage gate: currently blocked (about19 GB free on host).
- [ ] Actual expert/collection/replay gates.

No visual companion is required for this contract. The known scene has exact
coordinates and existing videos; no unmeasured reachability claim is made.
