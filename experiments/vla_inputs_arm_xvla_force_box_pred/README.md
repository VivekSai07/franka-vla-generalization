# vla_inputs_arm_xvla_force_box_pred: arm A5, A3's frozen policy fed a box predicted from pixels

M3b ([spec](../../docs/superpowers/specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md) D2, D6, D7). **No policy is trained.** A5 is A3 (X-VLA at checkpoint
101,520, trained on M2's 17-value force state with the target's box drawn on both camera images) run on M0's comparison tier with one change: the box that
is drawn comes from a pixel-only detector (`experiments/vla_inputs_box_detector`), not from the simulator.

* `adapter.XVLAPredictedBoxAdapter` extends A3's `XVLAForceBoxAdapter`. At every `act` the detector gets the **raw 512 x 512 uint8 renders** and the instruction, returns the (2, 5)
  box pair (front camera, then gripper camera; an absent box is all zeros and draws nothing), and A3's frozen `overlay.draw_frame` draws it; the policy then runs exactly as A3's.
* The oracle box the state builder computes is moved at once into the diagnostic `oracle_box`. `act` never reads it (a test inspects the source), and
  `tests/test_adapter.py` shows the drawn frames do not change when it does and that the same check fails for an adapter mutated to draw it.
* With a detector that returns the oracle box, A5's inputs equal A3's byte for byte (D7 ii).
* **Log:** one JSON line per `act` to `/out/arms/A5/predicted_box_logs/worker_<pid>.jsonl` (the campaign folder in the container): trial id, tick, the predicted box and the
  oracle box. It is diagnostic only; `experiments/vla_inputs_m3b_analysis` matches a trial's lines to its trace by the oracle sequence for the per-tick detector error (Q4).
* `INPUT_DIRS` = A3's (the X-VLA base bundle, the sealed S0 sidecar) plus the detector's `final/` directory (`detector.pt`, `detector.json`), so `prepare` hashes every
  weight file and every worker re-verifies it. The detector's constants digest is checked when it is loaded.

## What stays privileged

The box is no longer the oracle's, but **A3's state still carries M2's exact simulator forces** (wrist wrench, finger forces): A5 is "a policy that needs a force sensor and
a camera", not a deployable system. The detector is a closed-set detector for six known objects selected by the object the instruction names, not a language-grounding model.

## Confound

A3 was trained on perfect boxes, so a shortfall of A5 mixes "the predicted information is poor" with "A3 never saw noise". Arms A6-A9 (corrupted oracle) separate part of it;
a retrain on noisy or predicted boxes is a different experiment (M3b-2).

## Hash binding

M0's `prepare` binds this package (every top-level `.py` and this README), A3's arm module, the detector modules the adapter imports (`experiments/vla_inputs_box_detector`'s
`constants`, `inputs`, `model`, `predictor`), every other repository module the import loads, and the input directories. Do not edit any of it after `prepare`.

## Tests

`tests/test_adapter.py` (stub detector and stub model parts, robosuite image, CPU) and `tests/test_closure.py` (the arm closure, the core hash `c4f6c303f2ae`, A2's, A3's and A6-A9's recorded
closures unchanged).
