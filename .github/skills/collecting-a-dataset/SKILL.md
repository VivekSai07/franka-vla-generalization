---
name: collecting-a-dataset
description: "Use when launching, resuming, or debugging a data-collection run (--preflight/--collect) — covers the rejection budget, resume/rollback, hash-bound package files, and why a pilot must precede the full run."
---

# Collecting a dataset

Collection is its own gate sequence, separate from training's `--check`/`--preflight`/`--train`
(see [launching-a-run](../launching-a-run/SKILL.md)). A collection package accepts exactly one of
`--preflight` or `--collect`, plus `--resume` to recover a stopped run.

## 1. Gate sequence

| gate | proves | writes output? |
|---|---|---|
| `--preflight` | every bank identity derives in memory, no stepping | no — emits JSON on stdout |
| `--collect` | freezes plan/source/configuration + all bank identities with hashes, then runs the expert | yes — claims the canonical output |
| `--resume` | continues a stopped `--collect` from its journal | reuses the existing output |

Collection freezes its own source/plan/bank snapshot before any expert execution. The package's
`.py` files, its README, and the spec/plan docs it cites are re-hashed and compared against that
frozen snapshot; editing any of them **after `--collect` starts and before it finishes** raises
`'Frozen source/plan/bank snapshot changed'`. Treat the whole package as hash-bound for the run's
duration, not just its Python files.

## 2. Rejection budget, not a hard stop

A failed expert attempt, a replay drift, or a scene-specific gate (e.g. "distractor disturbed")
raises `ExpertRejection`, which is billed against a fixed rejection budget and **retries the same
slot** — never skips it. Skipping would desync the slot index from the admitted-episode index that
`attempts/`, `resets/`, and the train/val split boundaries all depend on.

If a slot exhausts every feasible redraw without an admission, collection stops with
`Slot <n> exhausted every feasible draw`. Recovery is a **layout substitution**, not a tolerance
change: add the layout to `INFEASIBLE_LAYOUT_RESAMPLES`, then `--resume`. This calls
`amend_frozen_plan`, which refuses unless no already-collected attempt used the moved layout —
verify that before amending, or the recovery silently invalidates already-collected data.

## 3. Rollback

A rejected episode that already committed to the dataset (validation runs after the LeRobot commit)
is removed by `rollback_episode`, which deletes that episode's own data/video/metadata files and
recomputes `meta/stats.json`. **Rolling back episode 0** leaves zero surviving episodes — do not
re-aggregate stats through `LeRobotDatasetMetadata` in that case, since with no local episodes left
it falls through to the Hub for version resolution and raises `OfflineModeIsEnabled` under
`HF_HUB_OFFLINE=1`. Just delete `meta/stats.json` when the episode count is zero.

## 4. Pilot before the full run

If the package moves or disturbs anything a prior dataset always left untouched (a body, a joint,
a region), that path is unverified. Watch the first ~10-15 admits interactively before committing
to the full multi-hour budget. A parked prop can carry residual velocity invisible in `qpos` —
check `qvel` too before trusting "at rest".

## 5. Monitor

Same caveat as training: `docker wait` does not reliably notify. Poll `--status` (or
`docker logs <container> | tail`) actively, especially in the first few minutes — a slot that
exhausts its rejection budget costs the same wall-clock whether it happens at slot 1 or slot 197,
and only a resumable run makes that recoverable rather than a total loss.

## See also

[diagnosing-a-binding-failure](../diagnosing-a-binding-failure/SKILL.md) for provenance/admission
mismatches, [forking-an-experiment-package](../forking-an-experiment-package/SKILL.md) if this
package was copied from another.
