# X-VLA G1 arm adapter for the G0 and M0 evaluators (arm A1 of the G1 evaluation)

`adapter.XVLAG1Adapter(checkpoint, state_builder=state_g1)` evaluates the X-VLA checkpoints trained by `experiments/xvla_g1_rand/`
on the sealed G1 randomized dataset: the robot state holds 17 values, the 9 joint positions, then the wrist wrench (6) and the
finger forces (2), clipped and standardised. It is loaded by both evaluators as
`experiments.vla_inputs_arm_xvla_g1.adapter:XVLAG1Adapter` (G0's `experiments/gen_eval/`, core `512806282b1f...`, and M0's
`experiments/vla_inputs_eval/`, core `c4f6c303f2ae...`; neither core is touched, `tests/test_closure.py`).

This is A2's adapter (`experiments/xvla_p2_force/eval_adapter.py`, arm `experiments/vla_inputs_arm_xvla_force/`) with one
difference: **the constants are G1's.** The clip bounds (0.1 / 99.9 percentiles) and the mean and standard deviation come from
G1's 2,700 training episodes (`experiments/xvla_g1_rand/forces.py`, pinned by `CONSTANTS_SHA256 b6e229658c04d6462cfe96bf71425b314608066835a4d21b33054ce5df2a2d15`);
A2's describe P2's distribution. Importing this arm never loads `experiments/xvla_p2_force/` (`tests/test_closure.py`).

Unlike A2's package, the code lives here and not in the training package: `xvla_g1_rand` is frozen, and its `SOURCE_FILES` bind the
training run. `experiments/xvla_g1_rand/` is imported read-only (`contract.observation` and `forces`).

## What the arm manifest binds

`prepare` of either evaluator records this package's non-test modules and this README, every other repository module the import
loads that the core does not bind (`xvla_g1_rand/contract.py` and `forces.py` among them, not the training `runner.py`), and:

* `INPUT_DIRS`: the X-VLA base bundle `.cache/xvla_trial/bundle` and G1's sealed signals
  `outputs/gen_g1_data/g1_20261110/signals` (3,000 files, every one hashed), which `forces` validates and derives the constants from;
* `EXTRA_SOURCE`: `outputs/gen_g1_data/g1_20261110/seal.json`, which `forces.verify_seal` compares with the pinned record.

Every container that evaluates this arm therefore mounts `outputs/gen_g1_data` read-only. **This README is bound by the arm
manifest**: it is written once and not edited after `prepare`.

## State builder

`state_g1(model, data, target)` is called at the observation instant of the rollout and returns a float64 vector of 17:
`data.qpos[:9]`, then `forces.standardise(concat(wrist_wrench, finger_force), forces.constants())` with S0's
`signals.wrist_wrench` and `signals.finger_force`. It reads `data` without writing it.

## Tests

| file | image | what it proves |
|---|---|---|
| `tests/test_closure.py` | robosuite, CPU | both evaluators accept the arm; the closure and the inputs; neither shared core changes (hashes equal A2's G0 and A0's M0 records); A2's force module is never loaded |
| `tests/test_adapter.py` | robosuite, CPU | the interface, the 17-value state handed to the processors, held-out instructions, the state builder uses G1's constants (stubs, no simulator) |
| `tests/test_parity.py` | robosuite, CPU | train/rollout parity: G0's own rollout loop, driven by the recorded actions of sealed G1 episodes (nominal geometry and two randomized conditions), feeds the policy the training row byte for byte; the normalised, padded proprio tensor equals the training one; P2's nominal replay gives G1-standardised raw signals |
| `tests/test_inference_boundary.py` | robosuite, GPU | the policy replans at ticks 0, 10 and 20 of 30; what is observed between inferences does not change the queued actions |

Loading a real checkpoint needs the GPU and is exercised by the evaluation's smoke stage.
