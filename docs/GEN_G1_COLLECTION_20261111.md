# G1 randomized dataset: collection report (launched 2026-11-10)

Spec: [G1 design](superpowers/specs/2026-11-06-g1-randomized-data-design.md) (decisions D1-D10, revision R1-R4, lemon-only gate note).
Plan: [G1 plan](superpowers/plans/2026-11-06-g1-randomized-data.md). Collector `experiments/gen_g1_data/` (forked from sealed P2's; scenes from
`gen_scene`, train pools only). Branch `feat/g1-randomized-data`, merged into `main` as `2a1bfa4`. The dataset is
`outputs/gen_g1_data/g1_20261110`, **the only copy, about 33 GB; never delete**.

## Result

| | |
|---|---|
| episodes | **3,000** (2,700 training, 300 validation, split in the plan), 1,356,000 frames, 6,000 videos |
| seal fingerprint | `845ac94063fd367cc315b8d287fc141df76ea171f9227f5794eef0bd8e4e161e` (provenance `350bf9fd…`, plan `3b57cfac…`, substitutions `807b504d…`, robosuite image `sha256:54d861aa…ccf27e`) |
| attempts / rejected | 3,046 / **46** (expert did not succeed 40, distractor check 6); 9 slots needed a substituted scene (the cell, count, height stratum and region stay) |
| instructions | 12 P2 pairs minus the held-out `milk -> blue` and `bread -> orange`: `held_out_pairs_collected` = 0 |
| per object (admitted) | bottle 600, can 600, cereal 600, lemon 600, milk 300, bread 300 |
| distractor counts 0-5 | 150 / 450 / 750 / 750 / 600 / 300 (R1's weights, exact) |
| table-height strata | 600 in each of five equal-width strata; start region P2 2,100, outer ring 900 |

## Gates

- **Expert gate (D4, R3), 96 first attempts, unrendered** (recorder commit `0668ce6`): 94/96 = 97.9% overall; every stratum passed except
  `object = lemon` at 14/16 = 87.5% against the 90% bar. Both failures are `GenExpert` grasp failures (the lemon turns 15-18 degrees in the grip
  and is squeezed out), the weakness P2's own gate showed (22/24). The bar was kept, not lowered.
- **Lemon-only gate**, pre-registered in the spec before it ran (`8b86f12`): 96 fresh lemon attempts, **93/96 = 96.9%, passed**; the three
  failures are grasps and none lies in the outer start region. Determinism was checked twice and equal. **Caveat that stays with the data:** the
  14/16 is on record; the unit that passed is the lemon-only gate, not the first one.
- Collection: 16 detached shards, 3 cores each (`LP_NUM_THREADS=3`), on commit `6441299`; per-episode gates (expert success, distractors,
  frames, replay, labels, schema, signals) in `journal.jsonl`.

## Rejections by group (admitted data is balanced by construction; the rejections show where the expert is weak)

| group | attempts / rejected |
|---|---|
| object | **lemon 642 / 42 (6.5%)**, can 604 / 4, bottle 600 / 0, cereal 600 / 0, milk 300 / 0, bread 300 / 0 |
| height stratum 0-4 | 607 / 7, 618 / 18, 602 / 2, 609 / 9, 610 / 10 |
| distractor count 0-5 | 152 / 2, 453 / 3, 762 / 12, 760 / 10, 615 / 15, 304 / 4 |
| start region | P2 region 2,130 / 30, outer ring 916 / 16 |

91% of the rejections are lemons: the 600 admitted lemon episodes are the scenes the expert could grasp, so they under-represent the
geometries that squeeze the lemon out. The evaluation drops G0's expert-infeasible lemon layouts for every arm (G0 feasibility, 15 trials), so
the same blind spot is on both sides of the comparison; it is still a statement about what the policy never saw.

## Merge, verification and cleanup

`merge.json`: 16 shards merged, every episode and all 6,000 videos verified against the shards. The per-collection shards folder
(`outputs/gen_g1_data/g1_20261110_shards`) was **deleted on 2026-11-13 with the user's explicit consent** after that verification; the sealed
dataset is the only copy. The pilot, gate, lemon-gate and contact-sheet folders next to it are kept.

## Training that followed (for the record; the evaluation is in the G1 evaluation report)

`outputs/train/xvla_g1_rand_v1`, `experiments/xvla_g1_rand/` (forked from the frozen `xvla_p2_force/`): A2's recipe, only the data and the
force constants differ. Status `complete`, **126,900 / 126,900 updates, effective batch 8, 30.7 h, one resume** from a verified snapshot,
no rollouts during training. 1,220,400 training rows, 1,142,100 complete chunk starts: **0.89 epoch** against A2's 5.0 epochs of 203,040 starts.
The force state's clip bounds, mean and standard deviation come from the 2,700 training episodes (`forces.py`, `CONSTANTS_SHA256
b6e22965…5d2`), never from A2's. Checkpoints (model sha256 prefix): 1 `ad68729c1c6d`, 50,760 `a8e4d8a18b53`, 76,140 `4adc50b4b65d`, 88,830
`90e3b9148b6a`, 101,520 `314a157d7f5a`, 114,210 `c9d07c930e14`, 126,900 `ec31dbaf3c5f`. `result.json` carries
`selected_checkpoint_update = 126900`: that is the runner's default marker, **not a selection**; the checkpoint is chosen by G0's selection
tier only. The run folder, `resume/` (about 20 GB of snapshots) and the five gate folders (`_admission`, `_preflight`, `_resumeproof`,
`_equivalence`, `_m2equivalence`) are kept because the user may continue training from update 126,900 later; that continuation changes a second
variable (more exposure), the LR schedule already sits at its floor there, so it would be a new run with its own contract, not a `--resume`.
