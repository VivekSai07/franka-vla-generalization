# G1 — a domain-randomized dataset for X-VLA (A2's inputs) — design

**Date:** 2026-11-06, **revised 2026-11-10 after G0 (R1-R4, before any G1 episode)**. **Status: approved in outline by the user 2026-11-06 (decisions P2, P3, P4, P6, P7 of the
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

## Revision 2026-11-10, after G0 and before any G1 episode exists

G0 ([report](../../GEN_G0_20261110.md)) measured A2 (paired drops, points): background 7.0, lighting 16.9 (`low_angle` 37.1), distractors 36.2
(`three` 27.1, `five` 50.0), disturbance 38.6 (`move` 49.3, `drop` 28.2), **spatial 88.7** (4/142 succeed at every level); nominal 91.5%.
No axis held. These amendments are made before any G1 data exists and supersede the rows they name; the original rows are kept below.

| id | amendment | reason |
|---|---|---|
| R1 | **D3 distractors: 0 to 5 per episode** (props of the six and pool A primitives), drawn with weights 0:0.05, 1:0.15, 2:0.25, 3:0.25, 4:0.20, 5:0.10. Was 0 to 3 | G0's `three` already cost 27 points and `five` 50: P2's one or two is the likely cause, and GraspVLA trains in clutter. Counts 3-5 at test become in-range; **novelty stays in the types** (pool B primitives never reach training) |
| R2 | **D3 is expressed only through `gen_scene.Condition(pool="train")` and `compose.reset(home_jitter_deg)`**, because `gen_scene` and `gen_eval` are bound by G0's core `512806282b1f…` and cannot change without a new protocol. Mapping: diffuse scale [0.25, 2.8], specular scale [0.5, 2.8], ambient scale [0.2, 1.5], headlight scale [0.3, 1.5], tint per channel [0.55, 1.0]; with probability 0.5 `light_dir` replaced by a direction at elevation [15°, 80°] and random azimuth, `light_shadows` with probability 0.5; table, floor and wall textures from the train pool (each with probability 0.85, else P2's default); `table_offset_m` [−0.07, +0.12]; `bin_shift_m` [−0.05, +0.05]; target `start` from P2's region or the outer ring (probability 0.3); home jitter ±10° per joint. **Dropped from D3** because `Condition` cannot express them: the *extra* directional light (replaced by the direction change above) and flat random-colour surfaces. Anything else the collector needs lives in `gen_g1_data`, never in `gen_scene` | the engine is frozen with G0's evaluator |
| R3 | **D4 gate per object:** ≥ 90% expert success for each of the six objects as well (G0's feasibility dropped 15 lemon trials the expert could not grasp); the collector's rejection rate per object and per table-height quintile is reported, so a skew in the admitted data is visible | the admitted data must not under-represent an object or a height |
| R4 | **Prediction replaced** (below, written 2026-11-10 before any G1 run). The 2026-11-06 prediction is kept for the record; it was written before G0 and FAILURES #72 shows its kind of reasoning (appearance over geometry) was wrong | |

### Mapping note, 2026-11-10 (written with the plan, before any G1 episode; where R2 was silent or the engine differs from what R2 assumed)

Checked against `conditions.py`, `levels.py`, `compose.py` and the compiled scene (robosuite image):

- **`ambient_scale` moves only the headlight's ambient.** The scene light's ambient is 0 (diffuse 0.7, specular 0.3, direction
  (-0.19, -0.19, -0.96), 74° above the horizon, no shadows); the headlight is 0.4 / 0.5 / 0.1 (diffuse / specular / ambient). R2's ranges
  are applied as written; the effective ambient is 0.1 x [0.2, 1.5] x headlight scale [0.3, 1.5] x tint.
- **Light direction:** elevation uniform in angle over [15°, 80°], azimuth uniform; the default 74° lies inside the range.
- **Where things live.** The target's used start is `layout["start"]` (`Condition.start` is never set, `extra_props` is never used);
  prop distractors are `layout["distractors"]`, pool A primitives are `condition.primitives`. The scene is rebuilt with
  `compose.model_for(layout, condition)` all the same.
- **R1's mix:** each distractor is a prop of the six with probability 0.6 while props remain (five at most besides the target), else a
  pool A primitive (36 shape x colour combinations). R1 did not say; the result is 949 / 1,143 / 649 / 212 / 43 / 4 slots with 0-5
  primitives.
- **Regions:** distractors in G0's distractor region (x 0.25-0.52, y +-0.20, radius at least 0.285 m); the target in P2's region (70%) or
  G0's `outer_start` ring (x 0.42-0.50, y +-0.12, 30%), so G0's `outer_start` level lies **inside** the training range, as D3 intended.
  Spacing is G0's `levels.spaced` (P2's rules plus primitives as neighbours), which also keeps every object 2 cm from both bins **at the
  slot's bin shift**: a shift of -5 cm (bins together) therefore excludes some starts and distractor positions that +5 cm allows.
- **Balance by decks, the rest by draws.** Cells (135 / 15 per cell), distractor counts (R1's weights, exact: 135 / 405 / 675 / 675 / 540 / 270 in
  training), five equal-width table-height strata (540 / 60 each, uniform inside a stratum) and the start region (70 / 30) are shuffled
  decks; the light redirect (0.5), shadows (0.5) and each texture (0.85) are independent per-slot draws; floor and wall are drawn
  independently (G0's `room` level forces two different maps; here they may coincide).
- **Home pose** +-10° per joint for the planned draw **and for every retry** (P2's retries drew +-1°).
- **Substitution** (a slot that exhausts its attempts on one scene) redraws the whole scene inside the slot's stratum; the cell, count,
  height stratum and region stay, so every stratum total of the admitted data is the plan's.
- **The expert gate** (D4, R3): 96 first attempts (scene 0, draw 0), unrendered like P2's gate: 16 per target object, 16 per distractor
  count, 19-20 per height stratum, 29 on the outer ring. At these sizes a bar of 90% means at most one failure per object or count
  and at most one or two per stratum, and 95% overall means at most four.
- **Not bound:** the collection's `provenance()` does not hash this spec or the plan document (FAILURES #52), so the docs pass after the result
  cannot break the seal. The plan digest (`PLAN_SHA256` in `gen_g1_data/plan.py`) pins the plan itself.
- Shards: 16 contiguous blocks of 188 (the first eight) and 187 slots (3,000 is not a multiple of 16).

## Prediction (2026-11-10, before any G1 episode; anchored on A2's G0 result; ranges per axis and class, FAILURES #69 and #72)

G1's paired drop on G0's benchmark tier (A2's in brackets), and whether the axis is held (upper bound ≤ 10):

| axis / level | G1 paired drop, points | held? | main failure class |
|---|---|---|---|
| nominal success | 85-94% (91.5%); on M0's comparison tier −6 to +2 points against A2's 91.3% | – | never lifted |
| spatial (88.7) | 5-35; `table_up5`/`down5` 0-25, `table_up10` 5-40, `outer_start` 5-30 | unlikely | never lifted, 5-30 layouts of 142 |
| distractors (36.2) | 5-25; `five` worst | unlikely | never lifted; wrong object 2-10% of layouts |
| disturbance (38.6) | `move` 20-45 (49.3), `drop` 15-35 (28.2); no recovery demonstrations, so little change on `drop` | no | failed re-acquisition |
| lighting (16.9) | 0-12; `low_angle` worst, 5-25 | possible | never lifted |
| background (7.0) | 0-8 | likely | never lifted |

G1 beats A2 (Holm over five, spec D10) on spatial and distractors almost certainly, on lighting probably, on background and disturbance
possibly not. If spatial stays above 35, the joint-space reach hypothesis of G0 is not fixed by data alone, and that is the first thing to
diagnose. The headroom on spatial is 89 points; FAILURES #69 says treat it as the plausible upper end of the gain, not the unlikely one.

## Prediction (2026-11-06, before any G1 run; superseded 2026-11-10 by the revision above)

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

## Note 2026-11-10: lemon-only gate (pre-registered before it runs)

The 96-attempt expert gate passed every bar except `object=lemon`: 14/16 = 87.5% (bar 90%); overall 94/96 = 97.9%. Both failures are
`GenExpert` grasp failures, reproduced exactly on replay: the lemon rotates 15-18 degrees in the grip during the close phase and is
squeezed out (fingers end at 8 and 15 mm against about 21 mm on the 14 successes). No neighbour or distractor contact was involved.
P2's gate had the same weakness (lemon 22/24 = 91.7%). The expert is frozen, so there is no fix to make.

Sixteen attempts cannot tell a true lemon rate near 88% from one near 95%, so the lemon bar is re-measured with more attempts. The bar
itself does not move. Registered now, before any attempt runs:

- 96 fresh lemon-only attempts (namespace `gen:g1:lemongate`, identities disjoint from every slot and from the 96 gate attempts, asserted):
  lemon's four trained cells in equal shares, 16 per distractor count 0-5, table-height quintiles balanced, 30% outer start, every range
  as in R2, first attempt only, unrendered, the same admission as the gate.
- **Pass:** first-attempt admission of the 96 is at least 90%, i.e. at least 87/96 (86/96 = 89.6% fails). Counts per quintile, distractor
  count, bin and region are reported, not barred (n is 16-20 per stratum).
- **Pass** lets collection proceed on the committed code. **Fail** stops collection; the next step is then a decision, not another run.
  The lemon-only gate is run once; the 14/16 stays on record next to its result.
