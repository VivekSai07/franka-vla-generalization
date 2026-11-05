# M3b — the box from pixels instead of the oracle: how much of A3's gain survives — design

**Date:** 2026-11-05. **Status: approved by the user 2026-11-05 (direction, two-stage form, the questions below, decision rule
shape). The numeric defaults (D3, D5, the thresholds in D10) are proposed and are fixed in the code before any run they govern.**
Part of the [VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). Depends on M3's result ([report](../../VLA_INPUTS_M3_20261105.md):
A3, the policy trained with the target's box drawn on both camera images, 571/573 = 99.65%, +8.4 points over A2), on S0 (sealed boxes) and on
M0's frozen core (`c4f6c303f2ae`). Nothing is built.

## Why this exists

M3's box is an oracle: it is computed from simulator state (`signals.target_box`), perfect, amodal and available on every frame. A real
robot has none of that. M3b measures what remains when the box has to come from the camera images and the instruction alone, and
what box errors the policy tolerates. Everything is read on one scale: **A2 (no box, 523/573 at 88,830) is the floor, A3 (oracle box, 571/573 at
101,520) is the ceiling**, a gap of **48 layouts** (8.4 points). Because the question is a position inside that gap, A3's saturation does not
limit M3b.

The same scale applies to M4a later: an auxiliary box head (no box input) is judged by its recovery between A2 and A3.

## Questions (the answers M3b is built to give)

| id | question | evidence | the answer takes the form |
|---|---|---|---|
| Q1 (primary) | How much of A3's gain survives a box predicted from pixels? | arm A5: A3's frozen policy fed by a pixel-only predictor, on the 573 comparison layouts | recovery **R = (S − 523) / 48**, S = A5's successes, with a paired interval; and whether A5 beats A2 (exact McNemar) |
| Q2 | What box error does A3 tolerate? | arms A6–A9: A3 fed the oracle box with one controlled corruption each | a dose-response read-out: success against A3 (571) per corruption, and the measured error the predictor must stay under |
| Q3 | Does A3 follow the box or the language? | A9: a distractor's box is drawn in place of the target's | success and wrong-object count under conflict; separates "the box names the target" from "the box marks where to reach" (M3's mechanism reading is untested) |
| Q4 | Where does a predicted box fail? | A5's failures, and the predictor's per-tick error against the live oracle box (logged, never used) | counts per failure class, the box IoU distribution, the share of ticks with no box in the gripper camera |

## What is established

| fact | source |
|---|---|
| A3 at 101,520: 571/573; A2 at 88,830: 523/573; A2 at 101,520: 517/573; A0: 451/573. A3 vs A2: +8.4 [6.1, 10.6], b = 50, c = 2, p = 6.1e-13 (Holm-amended 2.4e-12). Every one of A2's 50 failures became a pass; A3's two failures (ordinals 288, 300) were passes for A2 | [VLA_INPUTS_M3_20261105.md](../../VLA_INPUTS_M3_20261105.md) |
| A3's frame overlay is a 4 px magenta band outside the pixel box, nothing drawn when `in_frame` is 0; `in_frame` was 1 on every training frame, so **A3 never saw a missing box** | `experiments/xvla_p2_force_box/overlay.py`; S0 report |
| X-VLA's base weights hold only Florence-2's vision tower and language **encoder**: 196 language-model tensors, all `encoder.*`, no decoder layer and no `lm_head` (checked 2026-11-05 on `.cache/xvla_trial/bundle/model/model.safetensors`) | so GraspVLA-style box tokens need decoder weights and an architecture change: out of scope here (M3c) |
| M0 calls the arm's state builder, then `act(obs)`, at every tick; `obs["images"]` are the raw 512 × 512 uint8 renders; X-VLA runs the policy only every ten ticks (3 Hz replanning) and ignores the other observations | `vla_inputs_eval/rollout.py`; M2 `eval_tests/test_inference_boundary.py` |
| M0 records the live oracle `target_box` at every tick in each trial's `trace.npz`, so a predicted box can be scored against it after the fact | `vla_inputs_eval/rollout.py` |
| M0 loads an arm by `module:Class` with only the checkpoint path, from modules named `experiments.vla_inputs_arm_*`; arm names are `A0`–`A9`; its `compare` is hard-wired to A0, so M3b has its own analysis module | `adapters.py`, `evaluate.py` |
| S0's box is amodal and tight to the rendered geometry: against a segmentation render every target pixel lies inside it (±1 px), IoU mean 0.949, minimum 0.626, so a visible-pixels predictor cannot reach IoU 1 | [VLA_INPUTS_S0_20261030.md](../../VLA_INPUTS_S0_20261030.md) |
| the gripper camera lost the target in 26 of A0's 122 failures and 0 of its 451 successes | [VLA_INPUTS_M0_20261031.md](../../VLA_INPUTS_M0_20261031.md) |

## What stays privileged, stated plainly

M3b removes the **box** oracle only. A3's state still carries M2's exact simulator forces (wrist wrench, finger forces). A5 is therefore "a policy
that needs a force sensor and a camera", not a deployable system. The target is also one of six known objects, so the predictor is a
closed-set detector selecting by the object the instruction names, not a general language-grounding model.

## Decisions

| id | decision | reason |
|---|---|---|
| D1 | **The policy under test is A3 at checkpoint 101,520, untouched** (weights, M2 state builder with its frozen force transform, processors). Only the source of the drawn box changes. No policy is trained in M3b | one variable; 101,520 is both A3's selection winner and the matched-phase checkpoint |
| D2 | **Arms**, all on M0's comparison tier (573 layouts) at checkpoint 101,520 only, in the main campaign `outputs/vla_inputs_eval/m0_20261030`: **A5** predicted box; **A6** `jitter_small`; **A7** `jitter_large`; **A8** `dropout`; **A9** `swap`. New arm packages `experiments/vla_inputs_arm_xvla_force_box_pred` (A5) and `experiments/vla_inputs_arm_xvla_force_box_noisy` (A6–A9, one class per condition, M0 passing only the checkpoint), importing A3's frozen adapter and overlay read-only | M0 constructs an adapter from the checkpoint path alone, so a condition must be a class; A5–A9 do not collide with A1 (pi0.5) and A4 (aux head) |
| D3 | **Corruptions**, applied to the oracle box at every `act`, deterministically seeded by sha256 of (condition name, the oracle box's float64 bytes, the tick) so a trial reproduces whatever the worker order or a resume. Box format `(u_min, v_min, u_max, v_max, in_frame)` per camera, /512. **A6** each of the four coordinates of each camera plus Gaussian noise σ = 6 px; **A7** the same with σ = 18 px; both then ordered, given a minimum side of 2 px and clipped to the image. **A8** the gripper-camera box is not drawn (`in_frame` = 0) with probability 0.5 per `act`, the front camera unchanged. **A9** on both cameras the box of the **distractor nearest the target in 3D** is drawn instead of the target's, for the whole episode | the four error types the predictor can make: imprecision (two sizes), a missing box (what A3 never saw), a confident wrong object. σ and p are proposed; the plan fixes them in the contract before the first trial and records each condition's mean IoU with the true box |
| D4 | **The predictor** (`experiments/vla_inputs_box_detector/`): per camera it outputs `(u_min, v_min, u_max, v_max)` and a presence logit for the object named by the instruction (parsed from `put the <object> in the <colour> bin`; six objects). It is trained **only** on the 480 training episodes of the sealed P2 dataset (every k-th frame, k fixed in the contract), with S0's `target_box` as labels, from scratch or from weights already inside the pinned images (no downloads, containers stay offline), at most 50 M parameters. **Absence is learned from exact geometric augmentation** (translate the frame, transform the label, clip, absent when no area remains) because S0's own `in_frame` is constant 1. Checkpoint chosen by validation IoU on the 60 validation episodes of the plan split; **never by rollouts, never on any M0 tier layout** | S0 has no absent examples; offline IoU selects a perception module, not a policy checkpoint, and is not used for any capability claim |
| D5 | **Detector gate, fixed in the contract before training, applied before any A5 GPU trial:** per camera, mean IoU ≥ 0.85 and at most 3% of frames below IoU 0.5, on (a) the validation episodes' decoded frames and (b) **raw renders** of replayed validation episodes (S0 replay, robosuite image; the rollout sees raw renders, the training frames are video-decoded), plus presence F1 ≥ 0.90 on synthetic out-of-view frames. A failed gate means no A5 run; the detector is fixed using training and validation data only. If A5 later disappoints, a retrained detector is a **new arm** reported as such, never a replacement | a rollout is expensive; thresholds are not moved after seeing results |
| D6 | **A5 inference:** in `act`, the detector runs on `obs["images"]` (the raw renders) and the instruction; its box (presence < 0.5 → `in_frame` 0) is drawn with A3's frozen `overlay.draw_frame`; the policy then runs exactly as A3's. A5 **never** reads `signals.target_box` for drawing. The adapter also logs, per `act`: tick, predicted box, and the live oracle box kept in a separate diagnostic attribute | no privileged box in the loop; the diagnostic log yields Q4 |
| D7 | **Equality gates before any GPU trial counts:** (i) with every corruption off, the A6–A9 adapters give A3's drawn frames and actions byte for byte on replayed frames of sealed episodes 0, 18, 271, 539; (ii) a stand-in "detector" that returns the oracle box makes A5 equal A3 byte for byte; (iii) a leakage test: A5's drawn frames do not change when the oracle diagnostic is altered, and **the test fails when the code is mutated to draw the oracle**; (iv) a live re-run of two comparison trials with a corruption off reproduces A3's recorded traces bit for bit | the only difference from A3 must be the box |
| D8 | **Evaluation:** M0's frozen evaluator (`prepare` with checkpoint 101,520 as the single and matched-phase checkpoint, `smoke`, then the comparison tier with 6 workers × 7 cores); no selection tier and no `select`. Detector weights are an arm input directory bound by `prepare`. M0's `compare` is not used | M0's own A0 comparison is irrelevant to this question |
| D9 | **Analysis** (`experiments/vla_inputs_m3b_analysis/`, outside the frozen core, written and tested before the first A5 trial): R with a layout-resampling bootstrap stratified by object (numerator and denominator resampled together, 10,000 resamples, seed 0, M0's own resampling convention); exact McNemar of A5 against A2 (primary, one test, α = 0.05) and against A3 (descriptive); each of A6–A9 against A3 (exact McNemar, Holm over the four, flagged exploratory); failure classes; the per-tick detector error from the logs (IoU against the oracle, share of ticks with no box, jitter between consecutive replans). It refuses unless the arms share core `c4f6c303f2ae`, the same layouts and dropped layouts, and A3's and A2's recorded stages | the comparator changed; recorded before any result |
| D10 | **Decision rule, fixed before any result.** With S = A5's successes: **R ≥ 0.75** (S ≥ 559): a pixel-only box keeps most of the gain, M3b closes and M4a is reconsidered as a lower-priority route. **0.25 ≤ R < 0.75**: partial; M3b-0's dose-response and Q4 say which error costs the gain, and a retrain on noisy or predicted boxes (**M3b-2**, about 30 h, its own spec) is considered. **R < 0.25, or A5 not above A2 (p ≥ 0.05):** the gain needs near-perfect information, the oracle result stays an upper bound, and M4a (learn the box inside the policy) becomes the next route. The point estimate picks the branch; claims are made with the interval, and a straddled threshold is reported as such. Secondary: R against A2 and A3 at 101,520 (517 and 571: gap 54) | the thresholds are on the pre-registered quantity |
| D11 | **Multiplicity:** M3b adds no arm to the programme's family {A1 vs A0, A2 vs A0, A3 vs A2, A4 vs A0}. A5 vs A2 is one pre-registered test; A6–A9 against A3 are exploratory with Holm reported | |

## Prediction (2026-11-05, before any M3b run; ranges per failure class, as FAILURES #69 requires)

Reference: A3 = 571 successes (2 failures); A2 = 523 (50 failures: never lifted 33, timeout carrying 8, dropped outside 3, wrong object 4,
unstable 2); the headroom for any M3b arm above A2 is 48 layouts, below A2 it is bounded only by misleading boxes.

**A5 (predicted box), R: most likely 0.6 to 0.95 (S about 552 to 569).** By class, relative to A2's 50 failures: never lifted (the class the
box fixed): most of the 33 stay fixed if the box is within a few pixels, partly un-fixed if the gripper-camera box is often wrong or missing
(range: 3 to 20 still failing); wrong object: 0 to 12 (a predicted box on a distractor is the new risk, A2 had 4 and A3 0); timeout carrying
and unstable: 0 to 8; dropped outside: 0 to 3. The headline risks are the gripper camera at close range, boxes missing there (never seen
in training), and frame-to-frame jitter between replans. **R below 0.25 would surprise me.**

**A6 `jitter_small` (σ 6 px):** 560 to 571, almost no loss. **A7 `jitter_large` (σ 18 px):** 500 to 565, a measurable drop; box size at 512 px
is tens of pixels, so 18 px is a large fraction of it. **A8 `dropout` (gripper box missing half the time):** 480 to 571, the widest range,
because A3 never saw a missing box. **A9 `swap` (distractor's box):** below 80% (< 458) with wrong-object as the main cause, i.e. A3
follows the box; a result at or above 90% would mean the language dominates and the box is supplementary localisation. Reading of A9:
≥ 90% language dominates; 40 to 90% mixed; < 40% the box is followed. These are predictions, not criteria; none of the ranges is a gate.

## Review requirements

- **No privileged box in A5**, proved by D7 (iii) with a mutation. The forces remain privileged (see above).
- **Detector offline metrics are never capability.** The detector gate is a go/no-go for GPU time and a choice between detector checkpoints; the
  only capability measure is the rollout result. Offline IoU that passes the gate does not predict R.
- **Codec shift:** training frames are video-decoded, rollout frames are raw renders. D5 (b) measures it before the run, and the P2 policy
  already lives with the same shift.
- **A5 and A3 differ only in the box**, but A3 was trained on perfect boxes, so a shortfall in A5 mixes "the predicted information is poor"
  with "A3 never saw noise". M3b-0 separates the two partly; M3b-2 is the clean test of the second.
- Single training seed, one trial per layout, layouts the independent units; intervals describe layout noise only.

## Costs

GPU: about 2.7 h per 573-trial condition with 6 workers; five conditions about 14 h. The detector: frame extraction (decode the sealed videos of 480
episodes, every k-th frame, at the detector's input size; the cache must keep at least 65 GiB of free disk), then a short training. No policy training.
Disk for evaluation outputs about 4 GB. Sequence by default: corruption arms first (CPU and GPU busy overnight), detector after, A5 last.

## What M3b cannot show

- That the policy works without a force sensor (forces stay exact simulator quantities).
- Anything about open-vocabulary grounding, other objects, or a learned detector's behaviour outside these six objects and this scene.
- Whether training on noisy or predicted boxes would close a shortfall (M3b-2), or whether the box can be predicted and used inside one model (M3c, needs a decoder).
- Seed-to-seed variation.

## Follow-ons (not built here)

**M3b-2:** retrain on noisy or predicted boxes; considered under D10's middle branch, own spec. **M3c:** end to end, one model that predicts the box and
conditions its actions on it (GraspVLA-style); X-VLA's checkpoint has no Florence-2 decoder, so it needs weights and architecture work. **M4a:**
reconsidered after M3b under D10.

## Plan

[../plans/2026-11-05-vla-inputs-m3b-predicted-box.md](../plans/2026-11-05-vla-inputs-m3b-predicted-box.md).
