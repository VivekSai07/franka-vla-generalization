# B2 distractors — implementation plan

**Date:** 2026-09-28
**Design:** [../specs/2026-09-28-b2-distractors-design.md](../specs/2026-09-28-b2-distractors-design.md)
**Status:** package implemented, pilot pending

## What changed from the design

Three items were corrected during implementation after checking the code rather than the spec text.

1. **The floor-colour prerequisite is dropped entirely.** The spec cited
   `src/robots/franka_panda/model/scene.xml:21`, but the scene the packages actually load is
   `smolvla_scene.xml` (pinned in `experiments/fixed_layout/source.py`), so editing `scene.xml`
   would have changed nothing. More seriously, **both** XML files are recorded in the sealed
   `provenance.json` of `b1_language_data` and `broad_workspace_data`, so editing either would
   invalidate the provenance behind the 94.4% and 86.1% results and block any future run bound to
   those datasets. No scene file is touched. The control arm is therefore visually identical to
   B1, which strengthens the sanity-check comparison rather than weakening it.

2. **Per-cell arithmetic.** The spec's "16 episodes per cell" is wrong: 128 distractor-present
   training episodes over 4 background cells is **32 per cell**, against B1's 40. The thinning is
   ~20%, not 60%, so the claim that B1's 33,400 updates is a hard floor does not follow from cell
   thinning alone. The checkpoint ceiling is left to the screening curve.

3. **Screening trial count.** The spec lists five non-trivial checkpoints but bills four
   (`20 x 4 = 80`). Five checkpoints x 20 layouts is **100 trials**.

## Distractor mechanism

`smolvla_scene.xml` already defines `target_cube_blue` and `target_cube_yellow` with free joints.
Every dataset to date left them at the pilot reset's parked pose — verified across all 200 sealed
B1 resets at `(1.344, 2.0, 30.319)` and `(2.656, 2.0, 30.319)`, far outside the workspace and
outside camera view. Control episodes keep exactly that pose, so "no distractor" is the same
visual condition every prior dataset was collected in.

Parked cubes carry a **tilted** quaternion, so placement sets position *and* orientation, both
taken from the red cube's own rest pose in the base source rather than hardcoded — one source,
used by both the derived scene and its expected-qpos check.

## Composition

| arm | train | validation | total |
|---|---|---|---|
| control (0 distractors) | 32 | 8 | 40 |
| distractor-present (1-2) | 128 | 32 | 160 |

Background cells (instruction x arrangement) are balanced **within each arm**, so the
control/distractor contrast is never confounded with instruction or arrangement. Distractor count
is balanced 80/80 between one and two cubes, and the single-cube half splits evenly blue/yellow.

Distractors are drawn by deterministic rejection sampling over x in [0.20, 0.35], y in
[-0.45, 0.45] — the full y span, including the middle band the target pair never occupies.
Restricting them to the middle band would make "distractor" predictable from position, the same
class of confound as FAILURES #27. Every cube pair keeps the 15 cm clearance.

## Riskiest case first

Episode order is permuted so the first four slots are 2-distractor, 1-distractor, control,
2-distractor. Per FAILURES #29, the first minutes of the real run are then themselves the smoke
test, instead of discovering a distractor-specific failure at slot 150.

## New gates

- Distractor placement respects the same 15 cm clearance as the target pair, against both targets
  and any other distractor, enforced at plan time *and* re-checked on the derived scene.
- Unused distractors must remain **bit-exactly** at their parked pose.
- Active distractors must sit at the target rest height.
- **Distractor disturbance**: if the expert moves a distractor more than 2 cm, the episode is
  rejected and the slot retried, billed to the rejection budget. This converts the design's
  unverified "the expert never interacts with distractors" assumption into a measurement, and
  keeps knocked-over scenes out of the dataset.

## Verification steps

1. Plan balance, determinism and smoke ordering — `tests/test_plan.py`.
2. Scene bank: 321 identities, distractor placement, parked-pose preservation, clearance and
   disturbance gates — `tests/test_scenes.py`, `tests/test_distractors.py`.
3. Forked tests repointed at this package — copied tests that keep importing the source package
   pass while testing the wrong code (FAILURES #20).
4. **Pilot collection** before the full run, to measure the expert rejection rate with distractors
   present. The 6.6 h budget assumes B1's near-zero rejection rate, which is not established here.
5. Only then the full 200-episode collection.
