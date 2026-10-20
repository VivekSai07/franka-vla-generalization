---
name: collecting-a-dataset
description: "Use when launching, resuming, speeding up or debugging a data-collection run (--preflight/--collect, --gate, --pilot, --merge) — covers the rejection budget, resume/rollback, hash-bound package files, parallel shard collection, distractor/expert gates, and why a pilot must precede the full run."
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

If this package was forked, do not trust `amend_frozen_plan`'s per-episode equality check as-is:
it was written against the source package's per-episode fields. If this fork's episodes carry a
field that legitimately depends on layout geometry (e.g. a distractor position drawn for
clearance against the layout's cube placement), that field is expected to change for the
substituted episode too, and the inherited check will wrongly reject the amendment as unauthorized
drift. Scope the allowance to exactly that field, for exactly the episode(s) whose layout moved —
never loosen the check generally. Confirm with `--resume`'s dry validation (it raises before
writing anything, so a wrong fix costs a retry, not data) before trusting it.

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

A pilot of 10-15 episodes cannot show a cost that grows with the dataset: B3 reopened the dataset
after every episode and filled an 8 GB `/tmp` with Hugging Face caches at episode 541 (FAILURES
#41). For anything done once per episode, measure it at the final dataset size (one open of a
copy is enough) and size scratch space for the sum.

## 5. Monitor

Same caveat as training: `docker wait` does not reliably notify. Poll `--status` (or
`docker logs <container> | tail`) actively, especially in the first few minutes — a slot that
exhausts its rejection budget costs the same wall-clock whether it happens at slot 1 or slot 197,
and only a resumable run makes that recoverable rather than a total loss.

## 6. Making a collection fast: parallel shards (props P2: 540 episodes in 54 min, P1 took 9 h 38 m)

**Profile one episode before choosing a speed-up.** Time each stage in the image you will run:
props P1 was physics 0.6 s, OSMesa rendering ~37 s, frame write + AV1 encode 23 s. Rendering and
encoding are per-episode CPU work with nothing shared between episodes, so the win is running
episodes side by side, not a faster renderer. GPU (EGL) rendering was 17× faster per attempt but
needs an image change, is not byte-deterministic (1-pixel differences run to run) and renders
differently from OSMesa — deferred (TODO) because it adds variables a sealed dataset cannot carry.

The pattern (`experiments/props_p2_data/`, copy it):

| piece | rule |
|---|---|
| shards | one container per shard, a **contiguous** block of plan slots (`plan.shard_slots(k, n)`), its own journal, LeRobot dataset and `--resume`; rejections retry inside the shard, so slot order holds |
| workers | 12 on 48 cores with `-e LP_NUM_THREADS=4` (OSMesa threads); prove the thread cap is pixel-neutral (two determinism runs, capped vs default, byte-equal). Never more than ~30 workers (FAILURES #50) |
| memory | robosuite scene models are ~390 MB each and `props_scene` caches them forever; cap the cache per worker and clear it when full. Prove a rebuild is bit-identical first (compare every compiled `MjModel` array) |
| merge | `lerobot.datasets.aggregate.aggregate_datasets(..., data_files_size_in_mb=0, video_files_size_in_mb=0)` copies videos byte for byte and rewrites data with shifted indices. Verify every merged episode against its shard (lengths, tasks, data columns, video sha256, timestamps) **before** validating and sealing; seal the merged root |
| gate | the expert re-test runs unrendered (rendering does not feed physics), sharded: 288 attempts in ~1 min; determinism is checked separately on a dozen rendered attempts in two processes |
| pilot | 2 shards × 6 slots through collect → merge → verify → `LeRobotDataset` load, before the full launch |
| launch | `docker run -d --name <run>-collect-K ...` per shard; poll `--status` and `docker stats` (flat memory per worker is the sign the cache cap works) |

Size the estimate from the pilot's attempt **and** commit seconds under full load (attempts slowed from
41 s to 47 s with 12 workers), not from one episode alone (FAILURES #51).

## 7. Gates that measured the wrong thing (caught before data, P2)

- **"Distractor undisturbed" by pose threshold.** The lemon rocks on its own after reset (up to 13°
  untouched; 51° and 18 mm over a full episode). Gate on *contact* with anything but the table, from a
  substep re-run that must equal the recorded trace. Run the scene with nothing acting on it first and
  see how much each object moves by itself (FAILURES #54).
- **Spacing that models the fingers only.** The hand housing (±0.104 m along the finger axis, underside
  0.042 m above the grasp point) clipped tall neighbours of short targets: expert re-test 90.6%. Measure
  the collision geometry of every robot part that comes down near an object (FAILURES #55).
- **Asymmetric spacing marks the target.** A clearance applied only around the target makes "the
  isolated object" a learnable cue. Every spacing rule must be symmetric in the objects.
- When the expert re-test fails its bar, find the cause in the journal (partner body, tick, target) and fix
  the layouts or the expert; never move the bar.

## 8. Container traps on the collection path

- `--read-only` containers need `-e XDG_CACHE_HOME=/tmp/cache` even for a probe: robosuite's numba cache
  otherwise fails at import (`cannot cache function ... no locator available`).
- A run binds its spec and plan through `provenance()`: from the first `--collect` until the seal, and
  after it, do not edit them (FAILURES #52). Put later design in new files.
- Create the shards directory (`/out/<run>_shards`) in the launcher before `--collect`: the forked collector reads its free
  space before it makes it, so 16 containers died at once (FAILURES #59). Check `docker ps` a minute after a mass launch.

## See also

[diagnosing-a-binding-failure](../diagnosing-a-binding-failure/SKILL.md) for provenance/admission
mismatches, [forking-an-experiment-package](../forking-an-experiment-package/SKILL.md) if this
package was copied from another.
