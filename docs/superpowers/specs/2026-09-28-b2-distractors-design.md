# B2 distractors — design

**Date:** 2026-09-28
**Status:** design approved (asymmetric control/distractor split, single-run checkpoint sweep);
budgets below are estimates pending the first measured run
**Supersedes:** the one-line B2 entry in [TODO.md](../../../TODO.md#b2-distractors-second-axis)

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
(`cube_geom_blue`, `cube_geom_yellow`) alongside the red/green target pair — these can be reused
directly as distractor objects with no new asset authoring. Distractor placement needs the same
minimum-clearance rule already used for the pick/place pair, applied against both target cubes and
any other distractor in the scene.

**Sequencing dependency:** if the floor-colour housekeeping item (blue → grey,
`scene.xml:21`, tracked separately in TODO.md) is done, it must land **before** B2 collection
starts, not during or after — so all 200 B2 episodes are collected against one stable visual
domain, and a floor change never lands inside a sealed dataset's collection window.

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
still climbing at 33,400), not a formula. B2's finest per-condition split is thinner still: the
128-episode distractor-present train arm, divided across the same 4 background cells, gives
**16 episodes per cell** — versus B1's 40/cell, which already required doubling. The control arm is
thinner again (32 train episodes, ~8/cell if balanced). By this pattern, 33,400 updates (B1's
winner) is a **floor for B2, not a target.**

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
checkpoint (16700, 25000, 33400, 41700, 50000). 20 trials × 4 checkpoints = 80 trials ≈ **2.9 h** at
the measured 2.2 min/trial.

**Report.** A disjoint held-out layout set (proposed: 40 layouts, B2's own — not shared with B1's or
`broad_workspace_data`'s holdouts, same rationale as B1's holdout independence). Each report layout
evaluated at both conditions (control, distractor-present) so the control/distractor contrast is
**within-layout**, the same design move that made B1's C1/C2 contrast diagnostic rather than
confounded by layout difficulty. At 2 trials/condition/layout (T2b: aggregate rates survive at low
seed counts; only per-layout attribution needs ≥8 seeds, which this design does not claim):
40 layouts × 2 conditions × 2 seeds = **160 trials ≈ 5.9 h.**

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
86.1%: same-envelope, different-layouts (and possibly different floor colour) — a rough sanity
check that the control arm reproduces roughly B1-level performance, never a precise delta.

## Success criteria

- A determinate control-vs-distractor-present contrast on matched layouts, with geometry-clustered
  intervals.
- Control arm broadly in line with B1's 94.4%, as a sanity check that nothing else regressed.
- A defensible answer to "does distractor presence measurably reduce success," including a null
  result — that is a finding, not a failure of the experiment.

## Non-goals

- Resolving distractor *count* (1 vs 2) as a separate reported factor — see Open Decisions.
- Floor-colour change as part of this dataset's manipulation — it's a prerequisite housekeeping
  step, not part of what B2 measures, and must land before collection starts, not during it.
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
| floor-colour housekeeping (prerequisite, tracked separately) | few-episode sanity check | ~15 min |
| collection | 200 episodes | **6.6 h** |
| CPU admission (`--check`) | — | ~1 h (B1 measured actual) |
| preflight | 2 GPU updates | ~5 min |
| training | 50,000 updates (through all checkpoints) | **11.7 h** |
| eval — screening | 80 trials | **2.9 h** |
| eval — report | 160 trials | **5.9 h** |
| **total** | | **~28.2 h** |

Comparable in order of magnitude to B1's actual ≈33.5 h end-to-end.
