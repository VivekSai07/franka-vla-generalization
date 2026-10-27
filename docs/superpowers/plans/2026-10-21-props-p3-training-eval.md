# Props P3 — training and evaluation Implementation Plan

> Written 2026-10-21, after the dataset was sealed ([collection report](../../PROPS_P3_COLLECTION_20261020.md)). The packages are built
> and tested; no GPU run has started. Spec and pre-registration: [../specs/2026-10-19-props-p3-food-drinks-design.md](../specs/2026-10-19-props-p3-food-drinks-design.md).

**Goal:** train X-VLA on the sealed P3 dataset with the unchanged P1/P2 recipe, and evaluate every checkpoint on the 288 reserved layouts
under the pre-registered bin-choice rule.

**Packages:** `experiments/xvla_props_p3/` (training, admission image, hash-bound by its own `SOURCE_FILES`) and
`experiments/xvla_props_p3_eval/` (evaluation, robosuite image + GPU, not hash-bound), driver `scripts/xvla_props_p3_eval.sh`.

## Decisions made while building (2026-10-21)

1. **Recipe unchanged**: 126,900 updates = 5.0 epochs (the training split is the same 480 × 452 frames as P2's, so the count is the
   same), the dense checkpoint grid 1 / 50,760 / 76,140 / 88,830 / 101,520 / 114,210 / 126,900, snapshots every 2,600 updates. One new
   variable per phase: the data.
2. **One instruction, re-derived (collection plan Review Focus 4).** P2 matched ten instructions by text. Here the plan must carry only
   `sort the item`, the dataset's task table must be exactly `{prompt: 0}`, and `approved_instruction`/`policy_row` accept only it; the
   expert's own `put the X in the Y bin` is rejected. Each check is mutation-tested.
3. **No control tier and no cross-policy run** in the evaluation. With one object per episode and no distractors there is nothing to
   control for, and a P1 policy has no meaningful reading of a fixed prompt. The reference for "no treatment" is chance (0.5 bin choice),
   balanced by construction across the blue-bin side.
4. **Checkpoint selection on trained-object layouts only. This is a decision the spec did not state and the user may change before the
   first rollout.** The spec says "screen all checkpoints; report on those within 2 successes of the best screen; confirm once on the
   winner" but not what the winner is. P2's rule (highest report success, ties to the later checkpoint) would include the 160 held-out
   layouts and so choose the checkpoint whose held-out result is then reported. Counting trained-object layouts only keeps the exploratory
   held-out read-out unselected. Held-out successes are still recorded (`all_layout_successes`) but never decide anything.
5. **Wilson intervals over layouts** (one seed per layout, so a trial is a layout), not a clustered bootstrap.
6. **The classifier** has P2's causes minus `wrong_object`; the most specific condition (`placed_bin`) is checked before the gripper
   state (FAILURES #53). It is validated on real physics (expert replay as the policy succeeds for a drink and a food among the trained
   and the held-out objects; a `wrong_bin` outcome is produced by declaring the other bin), not on fabricated traces.

## Gate sequence before training (admission image, GPU where noted)

Work through section 1 of [../../../.github/skills/launching-a-run/SKILL.md](../../../.github/skills/launching-a-run/SKILL.md) first.

| # | step | note |
|---|---|---|
| 1 | disk | the runner's guard is 65 GiB free; checkpoints 29.7 GB plus the snapshot working set 31.5 GB. **The user decides what to reclaim.** |
| 2 | `--check` | CPU; recomputes the dataset fingerprint, binds sources and the base model, train-only statistics |
| 3 | `--preflight` | 2 GPU updates, FP32 weight and Adam round trip, one snapshot |
| 4 | `--resume-proof` | a resumed process reproduces update 2 exactly |
| 5 | `--equivalence` | the fast loop equals the serial loop byte for byte over 30 updates |
| 6 | `--train --stop-at T` | nights; `--train --resume` continues from the newest verified snapshot |

Nothing in `SOURCE_FILES` is edited between `--check` and the end of training; the training README is in it.

## Evaluation sequence (robosuite image)

`feas` (can run now, CPU) → `prepare` (needs the finished run) → `smoke` → `all`. Parallel workers pinned to cores, sized by peak GPU
memory (FAILURES #57). A stage is resumable; `rc=0` is not evidence a stage did work (check the trial folders and the wall-clock).

## Open for the user

- Decision 4 above (selection on trained-object layouts).
- What to reclaim on disk before `--preflight`.
- Whether `feas` (CPU, minutes, no GPU) may be run before training finishes: it only reads the sealed plan and writes under the evaluation
  output. *(Run 2026-10-21 without waiting: 287 of 288 feasible, `report_062` (bottle) dropped and counted; it changes nothing the policy is
  scored on, and the result is recorded in the status log.)*
