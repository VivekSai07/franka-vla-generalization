# X-VLA force/torque + drawn target box arm adapter for the M0 evaluator (arm A3)

`adapter.XVLAForceBoxAdapter(checkpoint, state_builder=state_force)` evaluates the X-VLA checkpoints trained by
`experiments/xvla_p2_force_box/` (M2's 17-value robot state, and the target object's box drawn as a 4 px magenta band on both
camera frames). M0 loads it as `experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter`.

The code lives in `experiments/xvla_p2_force_box/eval_adapter.py`, which is **not** in the training contract's `SOURCE_FILES`:
fixing an evaluation adapter after training does not invalidate the training binding. This package only re-exports it, because
M0's loader accepts arms from `experiments.vla_inputs_arm_*` only and its arm manifest binds this package (its non-test modules
and this README), every other repository module the import loads (the training contract, `boxes.py`, `overlay.py` and M2's
`forces.py` and adapter among them) and the directories in `INPUT_DIRS`: the X-VLA base bundle and the sealed S0 sidecar
`outputs/vla_inputs_signals/p2_signals_20261030`, which `forces` and `boxes` verify against its seal before they read it.

It subclasses M2's `XVLAForceAdapter` (itself A0's `XVLAAdapter`): loading, processors and action handling are A0's. Two things
differ. Its state builder returns M2's 17-value state and also stores the live box of that tick (`signals.target_box`, the
function that wrote the sidecar); `act` draws that box with the training overlay on the frames after the division by 255 and
raises if no box was stored since the last `act` or `reset`, so a stale box is never drawn. M0 calls the builder and `act`
at every tick on the same simulation state (`vla_inputs_eval/rollout.py`). M0's shared core (`experiments/vla_inputs_eval/`)
is not touched and its hash equals A0's (`tests/test_closure.py`).

Tests (robosuite image, CPU): `tests/test_closure.py`. The adapter's stub tests and the train/rollout parity tests on replayed
sealed episodes are in `experiments/xvla_p2_force_box/eval_tests/`. Loading a real checkpoint needs the GPU and is exercised
by M3's evaluation (plan Task 7).
