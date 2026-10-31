# X-VLA force/torque arm adapter for the M0 evaluator (arm A2)

`adapter.XVLAForceAdapter(checkpoint, state_builder=state_force)` evaluates the X-VLA checkpoints trained by
`experiments/xvla_p2_force/` (the robot state holds 17 values: 9 joint positions, then the wrist wrench and the finger forces,
clipped and standardised). M0 loads it as `experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter`.

The code lives in `experiments/xvla_p2_force/eval_adapter.py`, which is **not** in the training contract's `SOURCE_FILES`: fixing
an evaluation adapter after training does not invalidate the training binding. This package only re-exports it, because M0's
loader accepts arms from `experiments.vla_inputs_arm_*` only and its arm manifest binds this package (its non-test modules and
this README), every other repository module the import loads (the training contract and `forces.py` among them) and the
directories in `INPUT_DIRS`: the X-VLA base bundle and the sealed S0 sidecar `outputs/vla_inputs_signals/p2_signals_20261030`,
which `forces` verifies against its seal before it derives the clip and standardisation constants.

It subclasses A0's `experiments/vla_inputs_arm_xvla/adapter.XVLAAdapter`: loading, processors and action handling are A0's; only
`act` differs, in validating a 17-value state with `xvla_p2_force.contract.observation`. M0's shared core
(`experiments/vla_inputs_eval/`) is not touched and its hash equals A0's (`tests/test_closure.py`).

Tests (robosuite image, CPU): `tests/test_closure.py`. The train/rollout parity tests and the adapter's stub tests are in
`experiments/xvla_p2_force/eval_tests/`. Loading a real checkpoint needs the GPU and is exercised by M2's evaluation (Task 6).
