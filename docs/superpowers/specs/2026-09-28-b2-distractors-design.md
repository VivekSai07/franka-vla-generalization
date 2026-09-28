# B2 distractors — design

**Date:** 2026-09-28
**Status:** design approved (asymmetric control/distractor split, single-run checkpoint sweep);
budgets below are estimates pending the first measured run
**Supersedes:** the one-line B2 entry in [TODO.md](../../../TODO.md#b2-distractors-second-axis)
**Corrected 2026-09-28** during implementation — see *Corrections after code review* below. The
floor-colour prerequisite is **withdrawn**; the per-cell and screening-trial arithmetic is fixed.
Implementation plan: [2026-09-28-b2-distractors.md](../plans/2026-09-28-b2-distractors.md).

## Corrections after code review

Three items in the original draft did not survive contact with the code. They are recorded rather
than silently rewritten.

1. **The floor-colour change is withdrawn — it was both a no-op and seal-breaking.** The draft
   made "blue floor → grey, `scene.xml:21`" a prerequisite that must land before collection. But
   the scene the packages actually load is `smolvla_scene.xml`, pinned in
   `experiments/fixed_layout/source.py`, so editing `scene.xml` would have changed nothing that
   runs. Worse, **both** XML files are recorded in the sealed `provenance.json` of
   `b1_language_data` and `broad_workspace_data`, so editing either would invalidate the
   provenance behind the 94.4% and 86.1% results and block any future run bound to those
   datasets — the FAILURES #17/#28 class of mistake, promoted to a prerequisite. **No scene file
   is edited.** The distractor cubes already exist in the pinned scene, and the control arm is
   therefore visually identical to B1, which *strengthens* the sanity-check comparison below
   rather than weakening it.
2. **"16 episodes per cell" was wrong; it is 32.** 128 distractor-present training episodes over
   4 background cells is 32 per cell, against B1's 40 — a ~20% thinning, not ~60%. The inference
   that B1's 33,400 updates is therefore a hard *floor* does not follow from cell thinning alone.
   The ceiling is left to the screening curve.
3. **Screening was billed for four checkpoints but lists five.** 20 layouts × 5 checkpoints is
   **100 trials ≈ 3.7 h**, not 80 ≈ 2.9 h.

## Two mistakes this design corrects

Both surfaced during design review and are recorded here so they aren't repeated.

**Mistake 1 — crossing distractors with the full B1 factorial.** The first draft proposed
balancing distractor count across all four B1 cells (instruction × arrangement) as a fully crossed
design. That reopens the language question B1 already answered, in the same dataset as a new axis —
exactly the "two new axes, one failure, not attributable" trap this repo's own rules exist to avoid.
Instruction and arrangement still vary per episode (the task still needs an instruction, and
removing that variation would reopen the position confound B1 fixed), but they are **background
randomization, not a reported factor** in B2. The one axis under test is distractor presence.

**Mistake 2 — fine-tuning the B1 checkpoint on a B1/B2 data mix.** Suggested externally (see
conversation log), rejected for two reasons. First, no package in this repo has ever continued
training from a prior experiment's finetuned checkpoint — every run starts from the same upstream
base weights (`local_varied`, `broad_workspace`, the matched-exposure rerun, B1 all did this), so
"avoid catastrophic forgetting by mixing" was solving a problem this repo's own methodology doesn't
create. Second, a data-mix ratio is an unlogged hyperparameter: if B2 underperforms, part of the
explanation becomes "how much clean data diluted it," which cannot be cleanly separated from "how
hard is the distractor task." B1 avoided this by being fully self-contained; B2 does the same.

## Design: asymmetric control / distractor-present split

**200 episodes total: 40 control (0 distractors) + 160 distractor-present (1–2 distractors,
randomized per episode).**

This is not a 50/50 ablation split. The dataset should read as "about distractors" — most of the
budget (80%) goes to the condition under test. The 40-episode control arm exists only to give a
**matched, same-training-run, same-layout-population contrast**, the same role B1's swapped
arrangement played for language and T2a's 8 reach-matched control layouts played for the placement
offset. Without it, "did distractors hurt?" could only be answered by comparing against B1's 94.4%
externally — a different layout roster (and, if the floor-colour housekeeping item lands first, a
different visual domain), which RESULTS.md already treats as "a rough sanity check, never a precise
delta." The control arm buys a same-dataset answer instead.

Distractor count (1 vs 2) is **not** a separately reported factor in this design — see Open
Decisions if that resolution turns out to matter.

| arm | episodes | role |
|---|---|---|
| control | 40 | internal baseline, same envelope/task as B1, zero distractors |
| distractor-present | 160 | the axis under test, 1–2 non-target cubes, randomized per episode |

Train/val split follows the established 80/20 convention:

| arm | train | val |
|---|---|---|
| control | 32 | 8 |
| distractor-present | 128 | 32 |
| **total** | **160** | **40** |

## Feasibility

No expert changes needed — same as B1. The scripted expert already targets only the instructed
pair; distractors are scene dressing it never interacts with.

`src/robots/franka_panda/model/scene.xml` already defines non-target-colour cube geoms
(`cube_geom_blue`, `cube_geom_yellow`) alongside the red/green target pair — and so does the
scene that is actually loaded, `smolvla_scene.xml`. These are reused directly as distractor
objects with no new asset authoring and **no edit to either file**.

Every dataset to date left those two cubes at the pilot reset's parked pose — verified across all
200 sealed B1 resets at `(1.344, 2.0, 30.319)` and `(2.656, 2.0, 30.319)`, far outside the
workspace and outside camera view. Control episodes keep exactly that pose. Parked cubes carry a
tilted quaternion, so placement must set position *and* orientation, taken from the target cubes'
own rest pose rather than from literals.

Distractor placement needs the same minimum-clearance rule already used for the pick/place pair,
applied against both target cubes and any other distractor in the scene — and, because the reset
gate `require_no_robot_cube_contacts` covers all four cubes, clear of the robot as well.

**Sequencing dependency: none.** The floor-colour item is withdrawn (see Corrections), so nothing
blocks collection.

## Collection

Fresh dataset, fully self-contained — own plan, own layout bank, own reserved evaluation layouts,
own seal. Does not read, import, extend, or re-seal `b1_language_20260925` or
`broad_workspace_data`.

At the measured 1.97 min/episode: **200 episodes ≈ 6.6 h.**

## Training

Train from the **base bundle**, not from any prior checkpoint — consistent with every package in
this repo and with the correction above.

**Budget is uncertain and should be screened empirically, not assumed.** B1 needed double
matched-exposure's updates (16,700 → 33,400) purely from splitting 160 training episodes across 4
cells (40/cell) — an empirical finding (the screening curve was monotonic: 55% → 85% → 100% and
still climbing at 33,400), not a formula. B2's finest per-condition split is thinner: the
128-episode distractor-present train arm, divided across the same 4 background cells, gives
**32 episodes per cell** — versus B1's 40/cell. That is a ~20% thinning, so it is *suggestive* that
B2 needs at least B1's horizon, not a demonstration of it. The control arm is thinner again (32
train episodes, 8/cell). Treat 33,400 as the starting point of the sweep and let the curve decide.

Recommended checkpoint schedule for one training run: **[1, 16700, 25000, 33400, 41700, 50000]** —
reusing B1's early checkpoints as calibration (do they land visibly below B1's own screen numbers,
confirming B2 is harder?), then extending 25% and 50% past B1's winner rather than assuming it
suffices.

At the measured 0.84 s/update: **50,000 updates ≈ 11.7 h** for the full run through every
checkpoint (checkpoints saved along the way in one run, not separate runs).

Checkpoints must be screened by real rollouts, never by offline loss/MAE — this repo has two
independent instances (broad-workspace, B1) of offline metrics improving monotonically while
rollout success was non-monotonic or collapsed to 0% at an intermediate checkpoint.

## Evaluation

Two-tier protocol, following B1's screen/report split (a checkpoint is chosen on one disjoint
layout set, then reported on another, so checkpoint selection never touches the number that gets
published).

**Screening.** A dedicated held-out layout set (proposed: 20 layouts), distractor-present condition
only — that's the axis whose difficulty is actually uncertain — run against each non-trivial
checkpoint (16700, 25000, 33400, 41700, 50000). 20 trials × 5 checkpoints = 100 trials ≈ **3.7 h** at
the measured 2.2 min/trial.

**Report.** A disjoint held-out layout set (proposed: 40 layouts, B2's own — not shared with B1's or
`broad_workspace_data`'s holdouts, same rationale as B1's holdout independence). Each report layout
is banked and evaluated at both conditions (control, distractor-present) so the control/distractor
contrast is **within-layout**, the same design move that made B1's C1/C2 contrast diagnostic rather
than confounded by layout difficulty. Arrangement is fixed per reserved layout and balanced across
the population, because it is background rather than a reported factor. At 2 trials/condition/layout
(T2b: aggregate rates survive at low seed counts; only per-layout attribution needs ≥8 seeds, which
this design does not claim): 40 layouts × 2 conditions × 2 seeds = **160 trials ≈ 5.9 h.**

Report **geometry-clustered intervals**, not naive binomial — established necessity since T2a found
per-layout success ranging 12.5%–87.5%.

## Confounds and controls

**Data starvation.** The distractor-present arm (128 train episodes) is thinner than B1's per-cell
training data. A lower absolute score could reflect data volume rather than distractor difficulty
in isolation. The **control-vs-distractor-present contrast is within-model and within-layout**
(same weights, same checkpoint, matched layouts), so attribution of a drop to distractor presence
holds regardless of absolute level — only the absolute "how hard are distractors" claim is exposed
to the data-starvation caveat.

**Comparing B2's control arm to B1's 94.4%.** Same status as B1's own comparison to the sealed
86.1%: same-envelope, different-layouts — a rough sanity check that the control arm reproduces
roughly B1-level performance, never a precise delta. Because no scene file is edited, the visual
domain *is* identical to B1, which removes one confound the draft had accepted.

**Expert–distractor collision.** The claim that the expert "never interacts" with distractors is an
assumption, not a measurement: the expert is scripted against the target pair and has no knowledge
of the extra cubes. Collection therefore rejects and retries any episode in which a distractor moves
more than 2 cm, billed to the rejection budget, which both keeps disturbed scenes out of the dataset
and *measures* the collision rate. The 6.6 h collection estimate assumes B1's near-zero rejection
rate and is not established for four-cube scenes — a pilot precedes the full run.

## Success criteria

- A determinate control-vs-distractor-present contrast on matched layouts, with geometry-clustered
  intervals.
- Control arm broadly in line with B1's 94.4%, as a sanity check that nothing else regressed.
- A defensible answer to "does distractor presence measurably reduce success," including a null
  result — that is a finding, not a failure of the experiment.

## Non-goals

- Resolving distractor *count* (1 vs 2) as a separate reported factor — see Open Decisions.
- Floor-colour change — **withdrawn entirely** (see Corrections): it would not have affected the
  scene that actually runs, and it would have invalidated two sealed datasets.
- SmolVLA — X-VLA only, consistent with B1 (weaker learner at every prior stage; doubling policies
  doubles the bill to answer one question).
- B3 (full colour randomisation) — explicitly depends on a measured B2 result per TODO.md.

## Open decisions

1. **Screening/report layout counts (20/40 proposed)** — adjust if GPU/collection time is tighter
   or looser than estimated below.
2. **Whether to track distractor count (1 vs 2) as a secondary breakdown** within the report, even
   though it's not a balanced/reported factor — cheap to record post hoc from the trial logs even
   if not designed as a controlled contrast.
3. **Checkpoint ceiling (50,000 proposed)** — if the screening curve is still climbing at 50,000
   the way B1's was at 33,400, a second training extension will be needed; budget is not committed
   past this ceiling until the curve is seen.

## Budgets

| stage | quantity | cost |
|---|---|---|
| pilot (rejection-rate measurement, precedes the full run) | 10–15 episodes | ~0.5 h |
| collection | 200 episodes | **6.6 h** |
| CPU admission (`--check`) | — | ~1 h (B1 measured actual) |
| preflight | 2 GPU updates | ~5 min |
| training | 50,000 updates (through all checkpoints) | **11.7 h** |
| eval — screening | 100 trials | **3.7 h** |
| eval — report | 160 trials | **5.9 h** |
| **total** | | **~29.5 h** |

Comparable in order of magnitude to B1's actual ≈33.5 h end-to-end.
