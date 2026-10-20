# Props P3 collection: Food vs Drinks dataset sealed, 544 of 544 — 2026-10-20

A dated report; do not edit it to reflect later findings. Not a policy result: this is the dataset and what was checked
before it was sealed. Design and pre-registration: [spec](superpowers/specs/2026-10-19-props-p3-food-drinks-design.md) (decisions D1–D7
confirmed by the user 2026-10-20, risk of D4 recorded there), [plan](superpowers/plans/2026-10-19-props-p3-collection.md) (with its
build decisions). Package `experiments/props_p3_data/`, branch `feat/props-p3`. Robosuite image
(`sha256:54d861aa…ccf27e`, robosuite 1.5.2 / MuJoCo 3.4.0), OSMesa rendering, CPU only.

## What was collected

| | |
|---|---|
| episodes | **544 admitted, 0 rejected, 0 substitutions** (480 train, 64 validation) |
| prompt | `sort the item` in every frame of every episode and as the only entry of the task table |
| objects (trained) | can 136, milk 136 (drinks → orange bin); bread 136, lemon 136 (food → blue bin) |
| held out entirely | bottle (drink) and cereal (food): **0 episodes**, checked from the recorded arrays, not the plan |
| cells | 8 (object × side of the blue bin): 60 train and 8 validation each; each object 68 left and 68 right |
| seal | fingerprint `6430d34e…d151acf` (full: `6430d34ea2a13c9c49a6621b0a2509350cc5e84c489209154367dc27cd151acf`) |
| bound | plan sha256 `279dec81…3c7b46`, provenance sha256 `e1861e02…a328e0`, image digest above |
| location | `outputs/props_p3_data/props_p3_20261020/` (3.0 GB; sealed, never delete), shards kept beside it (`…_shards/`, 3.0 GB) |

Expert: the accepted P0 `PropsExpert`, unchanged (P2's 0.19 m carry existed to clear a standing distractor and is not used).
The bins are P0's; the container follows from the object exactly and the blue bin's side is balanced inside every object.

## Checks before the seal

- **Expert re-test (288 unrendered attempts, 6 objects × 2 arrangements × 24): 286 admitted = 99.3%.** Bars as P0/P2 (95% overall,
  90% per object, 93% per slot): all met. Per object: lemon 46/48 = 95.8%, every other object 48/48. Per arrangement: blue bin left
  142/144, right 144/144. The two misses are the same cell, lemon with the blue bin on the left (`lost_in_carry` and `grasp`; attempts
  101 and 102), which is the object the P0 gate already found hardest. The held-out objects (bottle, cereal) are 48/48 each, so the
  reserved layouts are reachable by the expert. The reserved layouts themselves were not run here; the evaluation package checks
  their feasibility before any rollout.
- **Determinism, and the thread cap:** 12 rendered attempts (one per object and arrangement), two separate processes, one with
  `LP_NUM_THREADS=3` pinned to 3 cores and one with the variable unset on 45 cores: traces **and per-frame image digests byte-equal**,
  12 of 12 successful. The 16-worker layout therefore does not change a pixel.
- **Pilot (2 shards × 6 slots, scratch root):** 12 of 12 admitted; merge verified every episode against its shard (24 videos);
  the merged dataset loads through `LeRobotDataset` (5,424 frames, task table exactly `['sort the item']`, 3 × 512 × 512 images,
  episode length 452 throughout); a mid-episode frame was viewed (bread, blue bin on the image's right for a `left` slot, as in P1/P2).
  Under no load: about 39 s per attempt and 22–23 s per commit.
- **Tests:** 110, in two images (77 that import no simulator, 13 episode tests and 20 merge tests in the robosuite image), and
  **every test was mutation-checked** (break the code, confirm a failure). Mutations that survived exposed real gaps and were closed with
  new tests: the stable-ticks and frame-count gates, image dtype and shape, the substituted-layout path in `admitted_episodes`, the
  seal's image field, `substitutions_of`, the timestamp and task-table checks in the merge, the shard-block check, and that
  `merge_and_seal` really calls `verify`. One check was dead code (an `episode_index` comparison that the row filter made always
  true) and was removed rather than tested. The fixed-prompt test (the frozen expert writes `put the X in the Y bin` into every frame;
  the sink replaces it and raises on anything else) was shown to fail with the replacement removed, with the wrapper's guard removed,
  and with the wrapper bypassed.

## The collection

16 containers, one per shard of 34 contiguous slots, each pinned to 3 cores (`--cpuset-cpus`, `LP_NUM_THREADS=3`), 16 GB `/tmp`,
`--ulimit nofile=65536:65536`. First container start to last exit: **46 min 43 s** (19:33:11 to 20:19:54 UTC). Mean per slot under
full load: 54.8 s for the attempt and 27.0 s for the commit (so 81.8 s, 12.4 h of serial work); 544 of 544 admitted on the first attempt of
the planned layout, no rejection, no budget used. Worker memory stayed between 3.6 and 4.6 GiB in the three samples taken during the
run (the cache cap of four models held). The merge (aggregate, verify 544 episodes and 1,088 videos against the shards, validate, seal) took 60 s.

**Independent re-verification in a second process** (read-only mount, `/tmp/p3_verify.py`, not part of the package): the seal's
fingerprint recomputes; the plan and the provenance digests recompute; every frame's `task_index` is 0 and every episode has 452
frames; and from the 544 recorded `episode_*.npz` arrays (not from the plan) no held-out object appears, every bin is the object's
container, the object-by-side and bin-by-side counts are the planned balanced ones, and the recorded order equals the plan's.

## One operational miss

The first `--collect` launch died within seconds in all 16 containers: `collect_shard` reads the free space of the shards directory
before creating it, and only the pilot directory had been created by hand. Nothing had been written (the death was before the shard
directory was made), the dead containers were removed and the launch repeated with the directory created first. The code was not changed
(it is hash-bound once collection starts); the launcher creates the directory. Recorded as FAILURES #59.

## What this does and does not show

- It shows a balanced, sealed training set with one prompt, four trained objects, and two objects never seen, whose expert
  re-test and thread-cap proofs were run before the data. It says nothing about whether a policy learns the mapping.
- **The risk recorded with D4 stands:** bottle and cereal have no demonstration at all, so their grasp reliability under the trained
  policy is unknown. The interpretability floor (30 stable placements out of 80 layouts per object) will decide whether the held-out
  result is readable; it is not changed.
- Only one pilot frame was viewed by eye. Per-episode videos were not reviewed; the expert's 99.3% and the replay gate (arm and
  prop to 1e-3, 544 of 544) are the evidence for their content.

## Open (training and evaluation are separate plans)

1. The training package must be a fork of `xvla_props_p2` with its instruction machinery re-derived for one shared prompt
   (`INSTRUCTIONS`, `planned_instructions`, `task_index_map`, `approved_instruction`); not started.
2. Disk: 71 GB free (92% full). P2 snapshots were 10.6 GB each; a P3 run needs headroom, and any reclaim is the user's decision.
3. The evaluation package (288 reserved layouts, one seed each, bin-choice classifier validated on successes first) is not started.
