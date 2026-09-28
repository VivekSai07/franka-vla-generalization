# B2 distractor data

Approved specification: [B2 distractors](../../docs/superpowers/specs/2026-09-28-b2-distractors-design.md).
Implementation plan: [B2 distractors plan](../../docs/superpowers/plans/2026-09-28-b2-distractors.md).
Implementation does not authorize collection. Main must independently review, rerun the CPU tests
and the non-stepping preflight before its single launch.

Module `experiments.b2_distractor_data.record` accepts exactly one of `--preflight` or
`--collect`. No count, source, output, seed, deadline, reserve, retry or resume overrides.
Preflight derives all 321 bank identities in memory and emits JSON on stdout; it creates no
experiment output. Collection exclusively claims the canonical output (an empty bind mount is
permitted) and freezes the plan, source, configuration and all 321 bank identities with their
hashes before dataset creation or any expert execution.

## The measured axis

One axis is under test: **distractor presence**. 200 admitted slots split asymmetrically, because
the dataset is about distractors and most of the budget belongs to the condition under test:

| arm | train | validation | total |
|---|---|---|---|
| control — 0 distractors | 32 | 8 | 40 |
| distractor-present — 1 or 2 | 128 | 32 | 160 |

Instruction and colour/region arrangement still vary per episode across the four B1 cells, but as
**background randomisation, not reported factors**. Removing that variation would reopen the
colour/position confound B1 closed. Cells are balanced *within each arm*, so the control versus
distractor-present contrast is never confounded with instruction or arrangement. Distractor count
is balanced 80/80 between one and two cubes, and the one-cube half splits evenly blue/yellow.

## Envelope

Targets use envelope F7 — two disjoint half-rectangles that cannot overlap by construction:

* Region A ∈ [0.20, 0.35] × [−0.45, −0.20]
* Region B ∈ [0.20, 0.35] × [+0.20, +0.45]

Which colour occupies which region is decided by the episode's cell, so colour stays decorrelated
from position.

Distractors are drawn by deterministic rejection sampling over [0.20, 0.35] × [−0.45, +0.45] —
the full y span, including the middle band the target pair never occupies. Confining them to that
band would make "distractor" predictable from position, the same class of confound as FAILURES
#27. Every pair of cubes in a scene keeps at least 15 cm of clearance.

## Distractor bodies

`smolvla_scene.xml` already defines `target_cube_blue` and `target_cube_yellow` with free joints,
so **no scene file is edited** — both scene XMLs are recorded in the sealed provenance of
`b1_language_data` and `broad_workspace_data`, and editing either would invalidate the datasets
behind the 94.4% and 86.1% results.

Every dataset to date left those two cubes at the pilot reset's parked pose, verified across all
200 sealed B1 resets at `(1.344, 2.0, 30.319)` and `(2.656, 2.0, 30.319)` — far outside the
workspace and outside camera view. Control episodes keep exactly that pose, so the control arm
reproduces the B1 visual condition rather than introducing a new one.

Parked cubes carry a tilted quaternion, so placement sets position **and** orientation, both taken
from the red cube's own rest pose in the base source rather than from a literal. The same helper
builds the derived scene and its expected-qpos check, so the two cannot drift apart.

## Gates beyond the inherited ones

* Active distractors sit at the target rest height; unused ones must remain **bit-exactly** parked.
* Clearance is enforced at plan time and re-checked on the derived scene.
* **Distractor disturbance**: if the expert moves a distractor more than 2 cm, the episode is
  rejected and the slot retried against the 100-rejection budget. This turns the design's
  unverified "the expert never interacts with distractors" assumption into a measurement and keeps
  disturbed scenes out of the dataset.

## Ordering

Episode order is permuted so the first four slots are 2-distractor, 1-distractor, control,
2-distractor. Per FAILURES #29 the opening minutes of the real run are then themselves the smoke
test, rather than a distractor-specific failure surfacing at slot 150.

## Reserved evaluation layouts

60 reserved layouts are never executed during collection: 20 for checkpoint screening and 40,
disjoint, for the report. Each is banked in **both** conditions so the reported control versus
distractor-present contrast is within-layout. Arrangement is fixed per reserved layout, balanced
30/30 across the population, because it is background rather than a reported factor.

Per-episode reset materialization is unchanged from B1: home-pose jitter changes the initial arm
qpos, so each admitted episode writes its own `resets/episode_{i:03d}.npz` (`initial_state` and
`qpos` only), with SHA256s recorded in `meta/info.json:episode_resets`.

Each admitted episode receives exactly two saved-control replays and no expert regeneration.
Infeasible home poses are redrawn up to eight times per episode; if none succeed the episode is
rejected and consumes one of the 100-rejection budget. Independent RandomState streams are derived
per split, coordinate, joint, phase, redraw, distractor and policy identity via SHA256.
