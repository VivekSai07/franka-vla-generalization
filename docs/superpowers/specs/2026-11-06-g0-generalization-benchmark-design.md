# G0 — generalization benchmark: five perturbation axes, paired with nominal — design

**Date:** 2026-11-06. **Status: approved in outline by the user 2026-11-06 (axes and levels, decision P5 of the
[programme](2026-11-06-generalization-programme-design.md)); every number below is fixed in code before the first policy trial.** Nothing is built.

## Question

How much does each of our policies (A0, A2; A3 descriptive) lose under unseen lighting, unseen backgrounds, more and unseen distractors,
spatial changes, and disturbances during the episode, compared with the same layouts under nominal (P2) conditions? G0 is also the yardstick
every later G arm is measured on.

## Design

| id | decision | reason |
|---|---|---|
| D1 | **New evaluator core** `experiments/gen_eval/`, forked from M0's `experiments/vla_inputs_eval/` (frozen, never edited). It adds a **condition** to every trial and a disturbance hook to the rollout; trial selection, seeding, restore, the success rule (target resting in the named bin, 30 stable ticks), the failure classifier and the statistics are M0's. Frozen with its own core hash after the first arm is prepared | M0's core cannot change; a condition is a new protocol |
| D2 | **Randomization engine** `experiments/gen_scene/` (shared with G1): a JSON-serializable `Condition` (lights, table/floor/wall textures, table height offset, bin shift, extra distractors, object start region, disturbance) applied to the props scene. With the **nominal** condition it must produce P2's scene exactly (gate D9) | one engine for test and training, with separate pools |
| D3 | **Layouts:** a fresh plan in namespace `gen:g0`, disjoint from P2's layouts, its re-test, M0's tiers and the pilot tier. **Benchmark tier:** 48 cells (6 objects × 2 bins × blue slot × 1–2 base distractors, M0's cells) × 3 = **144 layouts**. **Selection tier** (for choosing a later arm's checkpoint, never for claims): 24 layouts from 24 of the cells, disjoint from the benchmark tier. **Smoke:** 2 layouts | power for paired per-axis drops of about 15 points; independent selection |
| D4 | **Every layout is run once under nominal and once under each of the five axes** (6 trials per layout; 864 per arm on the benchmark tier, 144 per checkpoint on the selection tier). Within an axis, each layout gets one **level**, assigned by a fixed stratified rotation over cells, so every level has 36–48 layouts | the drop is paired per layout |
| D5 | **Levels** (fixed below; exact numbers pinned in `gen_scene` before the first trial, with a test): **lighting** — `dark` (all lights' diffuse ×0.3, ambient ×0.3), `bright` (diffuse ×2.5, specular on), `tinted` (warm (1.0, 0.75, 0.5) or cool (0.6, 0.75, 1.0) by layout parity), `low_angle` (one directional light 20° above the table from the side, shadows on, others ×0.3). **Background** — `table` (a test-pool texture on the table), `room` (test-pool textures on floor and walls), `both`. **Distractors** — `three` (3 props), `four` (3 props + 1 novel primitive), `five` (3 props + 2 novel primitives); props from the six, primitives from **test pool B** (shapes and colours never used in any training). **Spatial** — `table_up5` (+5 cm), `table_down5` (−5 cm), `table_up10` (+10 cm, GraspVLA's height test), `outer_start` (target start in the reachable ring outside P2's start region, bins shifted ±4 cm in y). **Disturbance** — `move` (when the gripper first comes within 6 cm horizontally of the target before closing, the target is moved 8 cm to a free reachable spot, seeded), `drop` (after the target is first lifted 5 cm, the gripper command is forced open for 10 ticks, then the policy has control again) | the five core capabilities; categories are G3's axis |
| D6 | **Pools:** textures are CC0 (ambientCG / Poly Haven), downloaded on the host, sha256-pinned in a committed manifest, files under `.cache/gen_textures/` (gitignored, bound by the core as an input directory). They are split once, by a fixed hash rule, into a **train pool** (G1, G2) and a **test pool** (G0 only); the split is committed before any G1 episode exists. Primitive distractors: pool A (train) and pool B (test) differ in shape family and colour set | test appearances and distractor types are truly novel; continuous factors are tested at fixed levels |
| D7 | **Tick cap:** 900 as M0 for nominal and axes 1–4; **1,200 for disturbance** (recovery takes time) | |
| D8 | **Feasibility before any policy trial** (unrendered expert, the engine's forked expert): every layout × condition that changes geometry (distractors, spatial) must be solvable; a disturbance layout is feasible if its nominal is feasible and the expert solves a fresh episode from the post-move pose (`move`). Lighting and background share the nominal's feasibility. A layout infeasible under a condition is dropped **for that condition and for every arm**, counted by cell and level, never replaced | no arm charged for an impossible trial |
| D9 | **Equality gates before any GPU trial counts:** (i) the engine's nominal condition composes a model whose renders of three sealed P2 initial states equal props_scene's byte for byte; (ii) the forked expert (reads bins and table from the model) reproduces three sealed P2 episodes' qpos traces byte for byte under nominal; (iii) `gen_eval` with the nominal condition re-runs two of A2's recorded M0 comparison trials (the same M0 layouts, given to `gen_eval` directly) bit for bit in every trace array, under M0's worker runtime | the fork changes nothing under nominal (M3b's FAILURES lesson: share the worker runtime) |
| D10 | **Visual check:** a contact sheet of every axis level (both cameras, first frame and a mid-episode frame) is shown to the user and approved before any policy trial | a broken texture or light is cheaper to catch on a sheet |
| D11 | **Arms:** A2 (M2, checkpoint 88,830, `experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter`) and A0 (P2 101,520, `experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter`) required; A3 (101,520, the oracle box, `experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter`) descriptive, run if GPU time allows. Adapters are imported read-only; `gen_eval` accepts identifiers under `experiments.vla_inputs_arm_*` | the controls every G arm is compared against |
| D12 | **Metrics per trial:** `task_success` (M0's rule), `grasp_lift` (target lifted ≥ 5 cm at any tick), `cause` (M0's classifier), `recovered` (disturbance axis: success after the disturbance fired), `disturbance_fired`; traces as M0 plus the condition and the disturbance tick | sub-metric comparable in kind to GraspVLA's |
| D13 | **Statistics:** per arm, axis and level: success under the perturbation against nominal on the same layouts, exact McNemar, the paired drop with a 95% bootstrap interval stratified by object (10,000, seed 0). An axis is **held** by an arm iff the paired drop's interval upper bound is ≤ 10 points (pre-registered definition, used later as G1's target). No family-wise claim in G0 (it is a measurement); Holm over five axes is reported | |
| D14 | **Selection tier is never used in G0's claims**; it exists so a later arm (G1) chooses its checkpoint on all conditions without touching the benchmark tier | |

## Prediction (2026-11-06, before any run; ranges per axis and failure class, FAILURES #69)

Reference: A2 is 91.3% on M0's nominal layouts; G0's nominal should land near it (85–95%) for A2 and near 79% for A0.

| axis | A2 paired drop (points) | dominant failure class predicted |
|---|---|---|
| lighting | 20 to 70; `dark` and `tinted` worst, `bright` mildest | never lifted (misaligned approach), wrong object under tint |
| background | 30 to 80; `both` worst | never lifted, wrong bin (bin colours against new textures) |
| distractors | 5 to 25; `five` worst | wrong object 3 to 15 percent of layouts, never lifted |
| spatial | `table_up5`/`down5` 20 to 60, `table_up10` 50 to 90, `outer_start` 10 to 40 | never lifted (joint-space policy reaching the trained height) |
| disturbance | `move` 10 to 50; `drop` 30 to 80 | timeout carrying / never re-grasped |

A0 drops similarly or more. A3 (descriptive): smaller drops on lighting and background than A2, because the drawn box does not depend on
appearance; no prediction is made about its disturbance behaviour. **No axis is predicted to be held by any arm.**

## Costs

Benchmark tier 864 trials per arm, about 4–5 h with 6 workers (disturbance trials run longer): A2 + A0 one night, A3 a second if wanted.
Feasibility is CPU (about 2,000 expert runs, sharded). Texture download a few hundred MB.

## What G0 cannot show

- Robustness to camera pose, object categories (G3) or physical parameter changes (mass, friction): not axes here.
- Real-world robustness: rasterized rendering only.
- Which training factor would fix an axis (G1 and G2 answer that, without per-factor ablations).

## Plan

[../plans/2026-11-06-g0-generalization-benchmark.md](../plans/2026-11-06-g0-generalization-benchmark.md).
