# G1 — a domain-randomized dataset for X-VLA (A2's inputs) — design

**Date:** 2026-11-06. **Status: approved in outline by the user 2026-11-06 (decisions P2, P3, P4, P6, P7 of the
[programme](2026-11-06-generalization-programme-design.md)); numbers below are fixed in code before the collection starts.** Nothing is built.
Built on the CPU in parallel with G0's GPU nights; trained after G0's A2 result exists.

## Question

Trained on the same task with the training distribution randomized (lighting, table/floor/wall appearance, table height, start region, bin
positions, distractor count and type, initial arm pose), does X-VLA with A2's inputs hold G0's axes 1–4, and does it recover better on axis 5,
than A2?

## Decisions

| id | decision | reason |
|---|---|---|
| D1 | **Collector** `experiments/gen_g1_data/`, forked from `experiments/props_p2_data/` (sealed, never edited): plan, record (sharded), merge, validate, seal; scenes from `gen_scene` with the **train pools only**; the engine's forked expert | same proven pattern as P2 |
| D2 | **Size:** 3,000 episodes, 2,700 training / 300 validation (split in the plan, as P2). Cells: P2's 12 instruction pairs with **milk→blue and bread→orange held out** (as P2), sides swapped | comparable instructions; P2's held-out pairs stay a test of language |
| D3 | **Per-episode randomization (training ranges):** lighting — every light's diffuse scale uniform in [0.25, 2.8], ambient [0.2, 1.5], tint per channel [0.55, 1.0], one extra directional light with probability 0.5 at elevation [15°, 80°] and random azimuth, shadows on with probability 0.5; textures — table, floor, walls each from the **train pool** (or a flat random colour with probability 0.2); table height offset uniform in [−7, +12] cm; target start region the union of P2's region and the reachable ring; bins shifted in y uniform in [−5, +5] cm; distractors 0 to 3 (props and **pool A** primitives); initial arm pose: home plus uniform jitter up to ±10° per joint (P2: ±1°) | covers G0's continuous levels (dark ×0.3, bright ×2.5, tints, low angle, ±5 and +10 cm, outer start) **inside** the training range, while G0's textures and pool-B distractors stay **novel**; 4 and 5 distractors are a count extrapolation |
| D4 | **Expert gate before collection:** on a pilot of 96 episodes across the randomization, the engine's expert must succeed in ≥ 95% overall and ≥ 90% at each table-height quintile and each distractor count; failed attempts are rejected and re-drawn by the collector as P2 did, with the rejection budget recorded. Contact sheets of 24 pilot episodes approved by the user | the data is only as good as the expert across the new ranges |
| D5 | **Signals recorded live** for every frame (S0's `signals.wrist_wrench`, `finger_force`, `target_box`, imported read-only) and sealed with the dataset; the force state's clip bounds (0.1/99.9 percentiles) and standardisation constants are derived from **G1's own training rows** and pinned by digest before training (A2's constants describe P2's distribution, not this one) | A2's inputs, consistent with this data |
| D6 | **Training package** `experiments/xvla_g1_rand/`, forked from `experiments/xvla_p2_force/` (frozen): only the dataset binding, the sidecar source and the derived constants change. **Recipe identical to A2's** (126,900 updates, effective batch 8, chunk 30, ten executed actions, the LR schedule, checkpoint grid 50,760 … 126,900, snapshots every 2,600). With about 2,700 × 452 frames this is **below one epoch** (≈ 0.8); comparison with A2 is at a matched update budget, not matched exposure, and every figure says so | user decision P6; compute is the binding constraint |
| D7 | **Equality gate:** pointed at P2's sealed data and the S0 sidecar with A2's constants, the G1 runner reproduces M2's runner byte for byte over 30 updates (metrics, weights, Adam moments). Plus the usual `--check`, `--preflight`, `--resume-proof`, `--equivalence` | proves the fork changes only the data binding |
| D8 | **Arm package** `experiments/vla_inputs_arm_xvla_g1/` (the `vla_inputs_arm_` prefix, so both M0 and `gen_eval` can load it); its state builder is A2's with G1's constants | |
| D9 | **Evaluation:** checkpoint chosen on **G0's selection tier** (all conditions, the six candidates) by total successes, ties to the later checkpoint; then G0's benchmark tier at the winner and at the matched-phase checkpoint 101,520. Secondary: M0's comparison tier (nominal, 573 layouts) at the winner, against A2 | the selection never touches the benchmark tier |
| D10 | **Primary test:** per axis 1–5, G1 against A2 on that axis's perturbed trials (paired by layout), exact McNemar, **Holm over the five axes**, α 0.05. **Criterion per axis:** G1 holds the axis (G0 D13 definition) and beats A2 at Holm p < 0.05. Nominal: G1 against A2 on G0's nominal trials and on M0's comparison tier, reported with intervals (a drop of more than 5 points on nominal is stated as a cost, not hidden) | pre-registered before any G1 run |

## Prediction (2026-11-06, before any G1 run; A2's G0 numbers will exist first and anchor these)

- **Lighting:** G1's paired drop 0 to 10 points (held), A2's large drop recovered almost fully. **Background:** 0 to 15 (held or nearly);
  novel textures are the hardest test, `both` worst. **Distractors:** 0 to 15, wrong-object still the main class at `five`.
  **Spatial:** `table_up5/down5` 0 to 10, `table_up10` 0 to 20, `outer_start` 0 to 15. **Disturbance:** `move` recovered more often than A2
  (state diversity from the wider start pose and positions), drop 5 to 35; `drop` little better than A2 (no regrasp in the data): 25 to 70.
- **Nominal:** −6 to +2 points against A2 (less exposure per condition, below one epoch).
- If `drop` stays far from held, that is G2's motivation; if `move` is held already, G2 narrows to re-grasping.

## Costs

Collection: about 3,000 × 1.1 min of admitted-episode work in 16 shards, about 3.5–4 h CPU, plus rejections; dataset about 17 GB.
Training about 28–30 h (65 GiB disk guard). Evaluation: selection ~4 h (six candidates in parallel), benchmark tier ~4–5 h per checkpoint,
M0 comparison ~2.7 h.

## What G1 cannot show

- Which randomized factor produced which gain (a bundle by design).
- Behaviour under camera-pose changes, categories (G3), or physical-parameter changes.
- Whether more data or more updates would close a remaining gap (one size, one budget, one seed).

## Plan

[../plans/2026-11-06-g1-randomized-data.md](../plans/2026-11-06-g1-randomized-data.md).
