# X-VLA arm adapter for the M0 evaluator

`adapter.XVLAAdapter(checkpoint, state_builder=state_p2)` is the control arm A0 (X-VLA 101,520 from the P2 bundle) and
the base for any arm that changes only the state contents or the loss. It loads a checkpoint as
`xvla_props_p2_eval.evaluate.load_policy` does, builds the processors from the checkpoint directory, and per call does what
P2's rollout does inline (`pre(observation(frame))`, `select_action` under `torch.inference_mode` and bf16 autocast, `post`).

It is **separately bound** from the shared core `experiments/vla_inputs_eval/`: the core loads it lazily as
`experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter` and never imports it, so an environment without X-VLA still runs the
core. Its source closure, weights, processors and runtime are recorded by `prepare` as the arm-specific manifest.

Tests (robosuite image, GPU, the P2 bundle at `/bundle`): `tests/test_parity.py` — the adapter's actions equal P2's own code
path byte for byte for 12 consecutive steps (two chunk inferences), and are sensitive to the image order and observation.
