# X-VLA continuation-pilot arm adapter for the M0 evaluator

`adapter.XVLAPilotAdapter(checkpoint)` evaluates the checkpoints trained by `experiments/xvla_p2_pilot/` (the robot state holds
19 values: 9 joint positions, then the variant's extras, then zeros). M0 loads it as
`experiments.vla_inputs_arm_xvla_pilot.adapter:XVLAPilotAdapter`.

The code lives in `experiments/xvla_p2_pilot/eval_adapter.py`, which is **not** in the training contract's `SOURCE_FILES`: fixing
an evaluation adapter after training does not invalidate the training binding. This package only re-exports it, because M0's
loader accepts arms from `experiments.vla_inputs_arm_*` only and its arm manifest binds this package (its non-test modules and this
README), every other repository module the import loads (the training contract, M2's `forces.py` and `eval_adapter.py` among
them) and the directories in `INPUT_DIRS`: the X-VLA base bundle and the sealed S0 sidecar
`outputs/vla_inputs_signals/p2_signals_20261030`.

**The variant is read from the checkpoint:** the marker (`checkpoint_result.json`, one of the checkpoint's hashed files) names the
variant and the sha256 of its file; the adapter loads `experiments/xvla_p2_pilot/variants/<name>.json`, refuses it if it changed
since training, and takes the state builder from it (`control`, `force`, `box`). So one identifier serves every variant, the
variant files are not in the arm closure (adding one never changes a prepared arm), and a checkpoint cannot be evaluated with the
wrong builder.

It subclasses A0's `experiments/vla_inputs_arm_xvla/adapter.XVLAAdapter`: loading, processors and action handling are A0's; only
`act` differs, in validating a 19-value state with `xvla_p2_pilot.contract.observation`. In a pilot campaign root the arms are
named by the plan: `A0` the shared continuation control (this adapter), `A1`..`A8` the variants (this adapter), `A9` the untouched
bundle checkpoint 101,520 (A0's own adapter, `experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter`). M0's shared core
(`experiments/vla_inputs_eval/`) is not touched and its hash equals A0's of the M0 campaign (`tests/test_closure.py`).

Tests (robosuite image, CPU): `tests/test_closure.py`. The train/rollout parity tests and the adapter's stub tests are in
`experiments/xvla_p2_pilot/eval_tests/`. Loading a real checkpoint needs the GPU and is exercised by the dry night (plan Task 6).
