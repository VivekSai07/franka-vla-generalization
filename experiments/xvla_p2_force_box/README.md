# X-VLA fine-tune on the props P2 dataset with force/torque in the state and the target's box drawn on both cameras (M3)

Training package for M3 of the VLA-inputs programme: X-VLA fine-tuned from the base checkpoint on the sealed props P2
collection `outputs/props_p2_data/props_p2_20261013` (540 scripted episodes in the robosuite scene: 480 training, 60
validation; 10 trained object->bin instructions; `milk->blue` and `bread->orange` never collected as targets; seal
`b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`), with M2's 17-value force state, **with one change: both
camera frames carry the target object's box, drawn as a magenta band**, from the sealed S0 sidecar
`outputs/vla_inputs_signals/p2_signals_20261030` (seal `7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c`).
Evaluated as arm A3 of the M0 protocol and tested against A2 (M2).

Design: [docs/superpowers/specs/2026-10-30-vla-inputs-m3-box-design.md](../../docs/superpowers/specs/2026-10-30-vla-inputs-m3-box-design.md)
(decisions D1-D11). Plan: [docs/superpowers/plans/2026-10-30-vla-inputs-m3-box.md](../../docs/superpowers/plans/2026-10-30-vla-inputs-m3-box.md).

A fork of M2's package `experiments/xvla_p2_force/` (frozen: it is A2's training binding and arm). **The recipe, the 17-value state
and its force transform are M2's, unchanged**: `forces` is imported from M2, never copied, and `tests/test_m2_reference.py` reads
M2's contract and compares the recipe, the schedule, the checkpoint grid, the sampler's batches, the joined state, the
statistics and the model config, so this run and A2 differ in one variable. The checkpoint writer, strict FP32 reload, snapshots,
resume, resource guards, faster loop and thermal guard are M2's.

## The overlay: the only difference

* **Source of the box:** the sidecar's `target_box`, `(u_min, v_min, u_max, v_max, in_frame)` per camera, float64 in [0, 1]
  (u right, v down, divided by 512), the front camera then the gripper camera. `boxes.py` reads it after M2's `forces` has
  verified the sidecar's seal, pins the whole table by `boxes.TABLE_SHA256`
  (`d168977668b8d3bb61acdc1aac1216564c5707c12b674740360fe82454a2a1d7`) and requires `in_frame` = 1 on every training row in both cameras
  (S0's finding), so no training frame is ever undrawn. The box is amodal: it marks where the target is, not what is visible.
* **Drawing rule** (`overlay.py`, spec D3): the pixel box is `[floor(512 u_min), ceil(512 u_max)) x [floor(512 v_min), ceil(512 v_max))`
  (half-open, so it contains every pixel the box touches); the band is 4 px wide, drawn **outside** that box (so it covers
  background, not the target), clipped to 512 x 512; magenta `(1.0, 0.0, 1.0)` exactly; nothing is drawn when `in_frame` is 0.
  Drawn on the 512 x 512 float32 frame in [0, 1] after video decoding and before LeRobot's processors, returning a copy. Front
  camera -> `observation.images.image`, gripper camera -> `observation.images.image2`. State normalisation stays `IDENTITY`; image
  normalisation is `IDENTITY` too (enforced by the shared `make_config`), so the processor's 512 -> 224 resize sees the band.
* **Loader** (`contract.JoinedDataset`): joins the 17-value state exactly as M2 does, then draws through the single module-level hook
  `contract.OVERLAY` (default `overlay.draw_frame`), looked up at call time. `runner.overlay_identity()` swaps it for the identity
  inside the zero-equivalence gate and nowhere else. A row that is not at `index == episode*452 + frame` is an error.
* **Model-input check** (`contract.check_model_inputs`): M2's checks (proprio is (n, 20); the 9 joints raw; the 8 extras standardised
  and recomputed from the sealed sidecar; the padding zero) and, new, the frames before the processors: for every sample and camera
  the exactly-magenta pixels are exactly the band of that sample's own sidecar box. An undrawn frame, a frame drawn with another
  sample's box, swapped cameras and a stray magenta pixel all fail.

What it cannot show: the box is an oracle projection of simulator geometry, not a detector's output; whether the box helps without
force (this is box on top of force); another colour, thickness or encoding. A gripper-camera box that leaves the frame at rollout
draws nothing, which never happened in training.

## What `--check` adds

* `colour_check.json` (spec D4): on **undrawn** decoded frames of every 10th training episode (frames 0, 225 and 451, both cameras;
  288 frames), the fraction of pixels within L-infinity 0.1 of magenta. It **fails above 0.01%** and the run stops: the colour is
  never changed silently, the user picks another one. The verdict is part of the CPU admission and is re-validated at every
  later admission (`contract.validate_cpu`).
* `overlay_samples/*.png` (spec D9): one sheet per object (the first training episode of each, six sheets), the front camera on
  top and the gripper camera below, three frames across (the first, the first `closing` frame, the first `closed_and_lifted`
  frame), drawn by the training loader, plus `index.json`. **The user approves them before `--preflight`**: a band in the wrong
  orientation passes pixel tests built on the same convention.

## What it binds

`current_binding()` checks, on every admission, everything M2's does (the parent's seal record, plan, substitutions, source
provenance, validation record, every file in `SOURCE_FILES`, the base model bundle and the sidecar's seal, provenance and the force
constants' digest) and, new, the box table's digest and the overlay specification. `--check` first recomputes both fingerprints
(the parent's and the sidecar's) and refuses unless they equal their seals.

`SOURCE_FILES` lists exactly the repository modules that importing `contract` and `runner` loads, plus this package's tests and
README, plus M2's `__init__`, contract and runner (`REFERENCE_FILES`: loaded only by the zero-equivalence gate, so the proof is of
equality with exactly those bytes). **The rollout side is deliberately not bound:** `eval_adapter.py`, `eval_compare.py` and
`eval_tests/` are outside `SOURCE_FILES`, so fixing an evaluation adapter or the comparison after training never invalidates the
training binding; M0's arm manifest binds the adapter.

## Recipe and horizon (M2's, unchanged)

Effective batch 8 (microbatch 4 x accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and
cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32 parameters with bf16
autocast, seed 20261007, base bundle in `.cache/xvla_trial/bundle`. 480 x 423 = 203,040 starts; **126,900 updates = 5.0
epochs**. Checkpoints at 1, 50,760, 76,140, 88,830, 101,520, 114,210, 126,900 (epochs 0, 2, 3, 3.5, 4, 4.5, 5); about 29.7 GB. Disk
guard 65 GiB, 28 GiB of free GPU memory. Offline metrics never select a checkpoint, stop a run early or support a claim.

## Gates

In order, each with its own output directory, in the admission image
(`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`), `--ulimit nofile=65536:65536` on every GPU run
(FAILURES #42), a large `/tmp`, and the sidecar mounted read-only.

| gate | output | proves |
|---|---|---|
| `--check` | `outputs/train/xvla_p2_force_box_v1_admission` | both fingerprints, binding, train-only statistics, the 17-value input check with the overlay check, `colour_check.json`, `overlay_samples/` (CPU) |
| `--preflight` | `..._v1_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot |
| `--resume-proof` | `..._v1_resumeproof` | a resumed process reproduces update 2 exactly |
| `--equivalence` | `..._v1_equivalence` | the fast loop equals the serial loop byte for byte over 30 updates |
| `--zero-equivalence` | `..._v1_zeroequivalence` | M2's own loop (its runner, imported lazily, bound as `contract.REFERENCE_FILES`) on M2's dataset class and this loop with the overlay replaced by the identity agree byte for byte over 30 updates (metrics, weights, Adam moments); and this loop with the overlay ON is reported unequal (the mutation check) |
| `--train [--resume] [--stop-at T]` | `outputs/train/xvla_p2_force_box_v1` | the run; refuses without the other five proofs, each bound to the same sources and data |

```bash
python -m experiments.xvla_p2_force_box.runner --check | --preflight | --resume-proof | --equivalence | --zero-equivalence | --train [--resume] [--stop-at T]
```

`--train --stop-at <unix time>` pauses cleanly at a snapshot; `--train --resume` continues from the newest verified snapshot. A
thermal guard sleeps at 83 °C until 75 °C (no effect on the math). A fix that needs a file in `SOURCE_FILES` changed is a different
run, not a resume.

## Files

| file | content |
|---|---|
| `overlay.py` | the band: `pixel_box`, `band`, `draw`, `draw_frame`, `near_magenta`; torch and numpy only, shared with the rollout |
| `boxes.py` | the sealed sidecar's `target_box` table, its pin, the labels for the sample sheets |
| `contract.py` | parent and sidecar binding, `JoinedDataset` with the `OVERLAY` hook, the overlay check in `check_model_inputs`, the colour check, horizon, schedule, `SOURCE_FILES` |
| `runner.py` | the gates, `colour_check`, `overlay_samples`, `overlay_identity`, `one_update`, `verify_precision_fast`, `Prefetcher`, the pause and the thermal guard |
| `eval_adapter.py`, `eval_compare.py`, `eval_tests/` | rollout side: the adapter that draws the live box for M0, the A3-against-A2 test, and the train/rollout parity tests (robosuite image); not bound by the training contract |
| `tests/` | contract, runner, overlay, boxes and the join, input check, zero-equivalence wiring, fork hygiene, and the reference comparison with M2 |
