# M2 — wrist force/torque and finger forces in the robot state — design

**Date:** 2026-10-30. **Status: draft for external review; Tasks 1, 2 and 4 of the plan are built on the CPU (2026-10-31), training and
the GPU gates are not.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). Depends on S0 (sealed sidecar) and M0 (evaluator, A0).
The code is `experiments/xvla_p2_force/`; D2's constants are derived at run time from the sealed sidecar and pinned by digest.

## Question

With everything else identical to P2's X-VLA run (data, images, instructions, recipe, sample order), does adding the wrist
force/torque and the two finger contact forces to the robot state change success on M0's comparison tier?

## What is established

| fact | source |
|---|---|
| A0's failures at 101,520 are dominated by "never lifted": 12 of 30 report failures, all 12 closed the gripper, 9 held the target without lifting; milk is 7 of the 12 and fails at the hold, not the approach | PROPS_P2_DIAGNOSTICS_20261019 |
| milk succeeds 57.6% on 118 fresh trained-bin layouts | PROPS_P2_RETEST_20261020 |
| the state already holds both finger positions, so a closed gripper is already visible to the policy | `observation.state` names in the P2 dataset |
| X-VLA takes 20 proprioception values; P2 uses 9 and pads with zeros, and the shared check asserts the padding is zero | `experiments/xvla_single_task/contract.py` `check_model_inputs` lines 187–204 |
| P2's runner is deterministic: `--equivalence` gave byte-equal metrics, weights and Adam moments over 30 updates | P2 training package, 2026-10-13 |
| **P2's X-VLA recipe does not normalise the state:** `normalization_mapping['STATE']` is `IDENTITY` (enforced by `xvla_single_task.make_config`, which raises otherwise; actions are `MEAN_STD`), and `check_model_inputs` asserts `proprio[:, :9]` equals the raw state. Joint radians enter the model as they are | `experiments/xvla_single_task/contract.py` lines 140–151, 187–204; LeRobot `configuration_xvla.py` default; checked 2026-10-30 |
| S0: training-row 0.1/99.9 percentiles of the 8 values (wrench up to ~47 N and ~2.8 N·m, finger forces up to ~17.8 N); tails heavy (max \|z\| 17–58); **mean wrist force is dominated by the hand's own ~10 N and is not ordered by object mass**; per-finger force in the lifted hold separates objects (cereal 4.1 N to can 6.6 N) | [VLA_INPUTS_S0_20261030.md](../../VLA_INPUTS_S0_20261030.md) |

## Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | State = 9 P2 values + `wrist_wrench` (6) + `finger_force` (2) = **17**, in that order, from the S0 sidecar for training and from the same S0 `signals.py` at rollout | approved signal set; 17 ≤ 20 needs no architecture change |
| D2 | The 8 extra raw values are **clipped to the training-row 0.1 and 99.9 percentiles recorded in S0's report**, then **standardised with fixed per-dimension mean and std computed on the clipped training rows** (at `--check`, from the sealed sidecar). This transform is applied by the data loader and by the rollout state builder, not by LeRobot's normaliser, which stays `IDENTITY` for the state exactly as in P2; the first 9 values stay raw | the P2 recipe feeds raw radians; raw forces of tens of newtons would dwarf them, and contact spikes would dominate without clipping; constants are fixed from demonstrations before training, never from rollouts |
| D3 | **Recipe unchanged:** 126,900 updates, effective batch 8, the same checkpoint grid, the same learning-rate schedule, the same sampler seed and sample order, the same base weights | one variable: the state's contents |
| D4 | New package `experiments/xvla_p2_force/`, a fork of `experiments/xvla_props_p2/` with: the sidecar join in the data loader (validated against the S0 seal), `OBS` and the statistics for 17 values, and a **local** `check_model_inputs` that asserts proprio `[:, :9]` equals the raw state, `[:, 9:17]` equals the D2-transformed extras and `[:, 17:]` is zero. The shared `xvla_single_task` check is not edited (it is bound by sealed runs) | fork, do not edit |
| D5 | **Zero-equivalence gate** before training: with the 8 extra values forced to exactly 0 after the D2 transform, the new runner reproduces P2's runner byte for byte over 30 updates (metrics, weights, Adam moments). Then the real run differs from P2 only by those 8 values | proves the fork changed nothing else |
| D6 | **Train/rollout parity gate:** replaying 3 sealed episodes' actions through the evaluator's state builder gives 17-value states byte-equal to the training rows the loader builds for those frames | the policy must see at test time exactly what it saw in training |

## Review requirements

[Audit R3/R6](../../VLA_INPUTS_AUDIT_20261030.md): 30 Hz describes the signal/command clock. Ten queued actions mean
**3 Hz policy replanning**; intermediate force samples do not alter queued actions. This tests instantaneous force
conditioning at chunk boundaries, not reactive force control. A null does not rule out causal history or faster feedback.

Compute extra-state statistics after clipping, using training rows only; preserve the original nine statistics exactly.
Report per-dimension clipping rates on demonstration validation rows and rollouts without retuning the frozen ranges.
Use explicit episode/frame keys and reject shifted, duplicate or missing sidecar rows. Train/rollout parity includes the
complete normalized and padded policy input, not just raw concatenation. The wrist signal is a sensor-frame interaction
wrench including inertial/gravitational contributions, not an external-load estimate.

Put `state_force` and its adapter in `experiments/xvla_p2_force/eval_adapter.py`, not M0's frozen core. Follow M0's
single-seed claim limits and matched-phase secondary comparison.

## Primary result

M0's primary test of A2 against A0. **Secondary (exploratory):** the count of `never_lifted` failures (paired), milk success.
Descriptive: wrist force and finger force traces of A2's and A0's failing milk trials (M0 records the signals for every arm).

## Prediction (2026-10-30, before any run)

Small or no effect is plausible. S0 showed that the object's weight is **not** readable from the mean wrist force (the hand's
own ~10 N dominates and the hold-minus-before-close difference is not ordered by mass), so the new information is grip force
(which separates objects in the hold) and contact transients; the
closure itself is already in the state. The expert never fails a grasp, so the demonstrations show "force present → lift"
but never a recovery. If the hold failure is about the timing of the lift relative to a firm grip, A2 should reduce
`never_lifted` on milk; if it is about grip geometry, it should not.

## Costs

About 32 h of training (P2: 0.524 s/update raw, ~0.9 s effective with thermal pauses); disk guard 65 GiB (checkpoints 29.7 GB
plus the snapshot working set); M0 evaluation ~2.5 h.

## What M2 cannot show

- That a real wrist sensor would help (these are exact simulator forces).
- Whether a history window, filtering or another normalisation would do better (each is its own variable).
- Which of the 8 values matters.

## Out of scope

Force history, a force output, changes to the expert, gripper-force control.

## Plan

[../plans/2026-10-30-vla-inputs-m2-force.md](../plans/2026-10-30-vla-inputs-m2-force.md).
