# M3 — the target's box drawn on both camera images, on top of M2's force state — design

**Date:** 2026-10-30, **redesigned 2026-11-03** (see "What changed"). **Status: approved by the user 2026-11-03** (approach B,
decisions D1–D11). Part of the [VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). Depends on S0 (sealed boxes),
M0 (evaluator, frozen core `c4f6c303f2ae`) and M2 (package `experiments/xvla_p2_force/`, trained and evaluated, frozen and
hash-bound; imported read-only, never edited). No code exists.

## What changed on 2026-11-03, and why

The 2026-10-30 design put the box into the robot state as 10 numbers (19-value state, P2's 9 + box) and tested A3 against A0.
It was replaced before anything was built, for four reasons:

1. **M2 changed the baseline.** A2 (force in the state) scored 523/573 = 91.3% against A0's 78.7% (+12.6 points, p = 1.1e-10).
   The user chose to build the next arm on the strongest one, so A3 = A2 + box and the control is A2.
2. **A numeric box cannot be stacked on M2's state.** X-VLA's pretrained weights fix `max_state_dim` at 20; M2 uses 17, so
   10 box values give 27 (25 even without the two constant `in_frame` flags). A box drawn on the image uses no state slot.
3. **GraspVLA, the user's reference, does not draw boxes into its input.** Its Progressive Action Generation has the VLM
   *predict* the target's 2D box (then a grasp pose) as tokens, and the action expert conditions on those predictions; the
   boxes in its figures and playground visualise predictions. A drawn oracle box is a visual prompt, a different and simpler
   intervention. The GraspVLA-style variant is scoped as **M3b** (below), to follow M3 if the box information helps.
4. **No pilot.** M2 calibrated the overnight pilots: the force pilot measured +3.2 points (dropped by its +5 rule), the full
   run +12.6. A pilot may promote but must not drop, so a box pilot cannot spare the full run.

What was believed and what replaced it is kept here so the history is readable: the numeric-state design and its prediction
(wrong-object choices toward zero, "about 6 points") were written against A0's failure mix, which A2 has since changed.

## Question

With everything else identical to M2's run (data, the 17-value state with its frozen force transform, instructions, recipe,
sample order), does drawing the target object's box on both camera images change success on M0's comparison tier, **against A2**?

## What is established

| fact | source |
|---|---|
| A2 at 88,830 fails 50 of 573: never lifted 33, timeout while carrying 8, dropped outside 3, wrong object 4, unstable in target 2, wrong bin 0 | [RESULTS.md](../../RESULTS.md) |
| so removing every wrong-object failure is worth at most 4/573 = **0.7 points**; any larger gain has to come from acquisition and alignment (never lifted is 33 of 50) | arithmetic on the row above |
| A0's gripper-camera target box left the frame at some tick in 26 of 122 failures and 0 of 451 successes (association, not cause) | [VLA_INPUTS_M0_20261031.md](../../VLA_INPUTS_M0_20261031.md) |
| X-VLA's state has 20 slots (`max_state_dim` 20 in the base config); M2 uses 17 | checkpoint `config.json`; `xvla_p2_force.contract.check_model_inputs` |
| P2's frames, M0's rollout frames and S0's segmentation check are all rendered by `mujoco.Renderer` at 512×512 with the same camera names, so S0's `(u, v)` (u right, v down, /512) index the stored image directly | `props_scene/expert.py`, `vla_inputs_eval/rollout.py`, `vla_inputs_signals/tests/test_boxes.py` |
| S0 `target_box`: `(u_min, v_min, u_max, v_max, in_frame)` per camera, float64 in [0, 1], amodal (occlusion ignored); `in_frame` = 1 on every training frame in both cameras; every target pixel inside the box ±1 px; IoU with the segmentation box mean 0.949, minimum 0.626 | [VLA_INPUTS_S0_20261030.md](../../VLA_INPUTS_S0_20261030.md) |
| X-VLA's image normalisation is `IDENTITY` (enforced by `xvla_single_task.make_config`); the processor resizes 512 → 224 (`resize_imgs_with_padding`), and the third view slot is zero and masked | `xvla_single_task/contract.py`; checkpoint config |
| M0 calls the arm's `state_builder(model, data, target)` immediately before `adapter.act(obs)` at every tick, after `synchronize_observation`, and the builder comes from `adapter.state_builder` | `vla_inputs_eval/rollout.py`, `evaluate.py` |
| M0's `compare` is hard-wired to A0 as control, and its core is frozen | `vla_inputs_eval/evaluate.py` (`CONTROL = "A0"`) |

## Decisions (approved 2026-11-03)

| id | decision | reason |
|---|---|---|
| D1 | **State = M2's 17 values, unchanged:** `qpos[:9]`, wrist wrench 6, finger forces 2, transformed by M2's own `experiments.xvla_p2_force.forces` (imported, constants pinned by `forces.CONSTANTS_SHA256`), not copied | the only new variable is the image overlay |
| D2 | **Overlay:** the target object only, on both cameras (front → `observation.images.image`, gripper → `observation.images.image2`), from the S0 sidecar's `target_box` (float64 as stored) in training and from `signals.target_box(model, data, target)` at rollout | approved: target only, both cameras |
| D3 | **Drawing rule:** a solid magenta band `(1.0, 0.0, 1.0)` of **4 px** drawn **outside** the box. Pixel box `x0 = floor(512·u_min)`, `x1 = ceil(512·u_max)`, `y0 = floor(512·v_min)`, `y1 = ceil(512·v_max)` (half-open); band = `[x0−4, x1+4) × [y0−4, y1+4)` minus `[x0, x1) × [y0, y1)`, clipped to the image. Nothing is drawn when `in_frame` is 0. Drawn on the 512×512 float32 CHW frame in [0, 1] (training: after video decoding; rollout: after the adapter's `uint8 / 255`), before LeRobot's processors, returning a copy | **changed from the 1 px proposed on 2026-11-03:** 1 px at 512 is under half a pixel after the resize to 224 and survives only as a faint tint; 4 px is about 1.75 px at 224. Drawn outward so the band covers background, not the target (S0's box contains every target pixel to ±1 px) |
| D4 | **Colour check before training:** `--check` reports, on undrawn decoded frames of every 10th training episode (frames 0, 225, 451, both cameras), the fraction of pixels within L∞ 0.1 of magenta. If it exceeds 0.01% the run stops and the user picks another colour; the colour is never changed silently | the band must not look like the scene |
| D5 | **Recipe unchanged** (M2's, which is P2's): 126,900 updates, microbatch 4 × accumulation 2, chunk 30, ten executed actions, LR schedule, checkpoint grid (1, 50,760, 76,140, 88,830, 101,520, 114,210, 126,900), snapshots every 2,600, sampler seed and order, base weights; derived from M2's contract and asserted equal in a test | one variable |
| D6 | **Package** `experiments/xvla_p2_force_box/`, forked from `experiments/xvla_p2_force/` (frozen): `overlay.py` (pure drawing), `boxes.py` (sealed sidecar `target_box` reader, seal verified before a row is read), the loader draws in `JoinedDataset.__getitem__`, the local `check_model_inputs` additionally asserts every sample's band pixels are exactly magenta for its own sidecar box. Rollout side `eval_adapter.py` and `eval_tests/` outside `SOURCE_FILES`; thin arm package `experiments/vla_inputs_arm_xvla_force_box/` (M0 loads only `experiments.vla_inputs_arm_*`, FAILURES #67) | fork, do not edit |
| D7 | **Zero-equivalence gate:** with the overlay replaced by the identity, the new runner reproduces **M2's runner** (`experiments.xvla_p2_force.runner`, bound as `REFERENCE_FILES`) byte for byte over 30 updates: metrics, weights, Adam moments. Mutation: overlay on → unequal (recorded) | proves the overlay is the only difference from A2 |
| D8 | **Train/rollout parity gate:** on sealed episodes 0, 18, 271 and 539 replayed with S0's `replay.frames`: the live `signals.target_box` equals the sidecar row byte for byte at every frame; the adapter's drawn frame differs from the undrawn render exactly on the band computed from the sidecar box, where it is magenta; the processed `image_input` the policy receives equals the training processor's output for that drawn frame; the 17-value state parity is M2's, re-run | the policy sees at test time what it saw in training |
| D9 | **Visual check:** before `--train`, PNGs of drawn frames (6 training episodes × frames 0 and the first closed-gripper frame, both cameras, from the training loader; the same for one replayed rollout) are written to the admission directory and shown to the user, who approves them | a wrong-orientation band can pass pixel tests built on the same convention |
| D10 | **Comparison.** Primary: A3 at its selection-tier winner against A2 at its selected checkpoint 88,830 on the 573 comparison layouts; matched-phase secondary A3 101,520 against A2 101,520 (A2's comparison-tier results exist for both). Exact McNemar and the stratified paired bootstrap from M0's `stats`, computed by `experiments/xvla_p2_force_box/eval_compare.py` (outside the frozen core and the training binding), which refuses unless both arms share core `c4f6c303f2ae`, the same layout set and the same dropped layouts. Written, tested and committed **before** A3's comparison tier runs. M0's own `compare --arm A3` (against A0) is run too and reported as secondary | M0's `compare` only knows A0 |
| D11 | **Multiplicity, amended before any M3 result:** the fixed four-test Holm family becomes {A1 vs A0, A2 vs A0, **A3 vs A2**, A4 vs A0}, unrun arms at p = 1. With A2's p = 1.1e-10 ranked first, A3's test is effectively at α/3 ≈ 0.0167. M0's core still prints its own family (A3 vs A0); both are reported, and the amended one decides | the comparator changed; recorded before the result |

**Pre-registered criterion:** the box helps on top of force iff A3's paired effect against A2 is positive and its
amended-Holm-adjusted p < 0.05. **Exploratory (unadjusted, paired):** never-lifted count, milk success, held-out pairs,
wrong-object count, and the number of failures in which the gripper-camera box leaves the frame (M0 records `target_box`
every tick).

## Review requirements

[Audit R7](../../VLA_INPUTS_AUDIT_20261030.md) still holds: `in_frame` is geometric projected-box validity, not visible pixels;
boxes are amodal. No claim about detector robustness follows from an oracle box. A gripper-camera box that leaves the frame at
rollout draws nothing, a case never seen in training (S0: `in_frame` = 1 on every training frame); report how often it happens.
The overlay is drawn after decoding in training and on the raw render at rollout, so it is pixel-exact on both sides; the
codec difference between decoded training frames and raw rollout frames is pre-existing and shared by every arm.

## Power (exact, N = 573, paired)

| discordance | +3 pts (α .05 / .0125) | +4 pts | +5 pts |
|---|---|---|---|
| 0.10 | 0.58 / 0.36 | 0.84 / 0.67 | 0.97 / 0.90 |
| 0.15 | 0.42 / 0.23 | 0.66 / 0.45 | 0.86 / 0.70 |
| 0.20 | 0.33 / 0.16 | 0.54 / 0.33 | 0.74 / 0.54 |

A2 against A0 had discordance 128/573 = 0.22. A2 fails 50 layouts, so the largest possible gain is 8.7 points. Effects
under about 4 points are unlikely to reach significance; read the interval, not the verdict alone.

## Prediction (2026-11-03, before any M3 run)

Small: 0 to +4 points, probably not significant. Wrong-object can add at most 0.7. A gain would have to come from the
approach and alignment, the box marking where to go in the gripper camera; never-lifted would then fall. A null is plausible
and would mean "the box adds nothing on top of force", not "the box is useless". (M2's prediction was wrong by a wide margin
in the other direction; this one is recorded, not leaned on.)

## Caveats

- **Box given force, not box alone.** Box-only on A0's 9-value state is a separate, optional arm (not scheduled).
- The box names the target: a wrong-object reduction is partly by construction.
- Privileged at test time: an upper bound on what a perfect detector could give through this encoding.
- One training seed; layout is the unit.

## Costs

Training as M2 (measured 28.6 h; disk guard 65 GiB, 233 GB free on 2026-11-03; GPU guard 28 GiB). Evaluation as M2: selection
tier, then the comparison tier at two checkpoints about 5.4 h with 6 workers.

## What M3 cannot show

- Whether a numeric box, boxes of all objects, another colour or thickness, or a detector's boxes would do the same.
- Whether the box helps without force.
- Whether the gain survives detector noise (M3b's question).

## Out of scope

Box-only arm on A0, distractor boxes, a detector, predicted boxes (M3b).

## Follow-on: M3b — predict the box, then act on it (approach C, scoped only)

GraspVLA-style: the policy produces the target's box itself and the action is conditioned on that prediction, so no oracle is
needed at test time. **Its spec is written only after A3's result. Start condition:** A3 meets the pre-registered criterion, or
the user overrides with the reason recorded. Candidate forms, to be chosen then: (i) two stage: a box predictor (a head on
X-VLA's features or a separate detector, trained on S0's boxes) whose prediction is drawn onto the frames for A3's policy,
which measures the cost of predicted against oracle boxes with no new policy training; (ii) end to end: one model predicts the
box and conditions its actions on it (X-VLA does not run its Florence-2 decoder today, so this is an architecture change).
Relation to M4a: M4a predicts the box and discards it; M3b predicts and uses it. If M3b runs, M4a is reconsidered (it can
become M3b's first stage). Programme order: M3 → M3b → M4 → M1.

## Plan

[../plans/2026-10-30-vla-inputs-m3-box.md](../plans/2026-10-30-vla-inputs-m3-box.md).
