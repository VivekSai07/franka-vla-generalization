# Props P2 — instruction sorting with distractors — design

**Date:** 2026-10-13
**Status:** approved by the user in conversation (2026-10-13). Supersedes nothing: the props roadmap
spec ([2026-10-05-props-sorting-design.md](2026-10-05-props-sorting-design.md)) is hash-bound by the
P1 seal and stays unedited; this file is P2's own spec.
**Follows:** P1 ([PROPS_P1_EVALUATION_20261009.md](../../PROPS_P1_EVALUATION_20261009.md)): confirm
20/20, report 60/80 = 75% on checkpoint 126,900, zero wrong-bin outcomes.

## Question

Can the policy find the object the instruction names when one or two other objects are on the
table, and put it in the named bin, when the bins swap sides and some object–bin pairs were never
trained?

## Decisions (user, 2026-10-13)

| question | decision |
|---|---|
| collection speed | parallel CPU-rendered shards (OSMesa, P1's renderer and byte-determinism); GPU (EGL) rendering deferred to a phase boundary with its own side-by-side check (TODO.md) |
| distractors in training | 1 or 2 per episode, half each; 0-distractor layouts only at evaluation |
| carry path vs tall distractors | the expert carries higher; layouts keep a minimum spacing; distractors anywhere in the start region |
| training start | the X-VLA base model with P1's exact recipe (5 epochs), so P1 and P2 differ only in data |
| success | four pre-registered conditions on the report tier, all required (below) |
| held-out pairs, expert gate | as P1 (`milk→blue`, `bread→orange`); P0's gate bars plus a distractor-undisturbed check |

## Measured inputs (2026-10-13, robosuite image, one P1 episode)

- Physics + expert 0.6 s; OSMesa rendering of 2 × 452 frames ≈ 37 s; frame write + AV1 encode +
  finalize 22.9 s. P1 ran all of it in one process: 9 h 38 m for 540 episodes.
- Compiled footprints (bounding-box half-diagonal, m): can 0.035, bread 0.031, lemon 0.039,
  milk 0.028, bottle 0.040, cereal 0.052. Heights (m): can 0.080, bread 0.048, lemon 0.040,
  milk 0.144, cereal 0.150, bottle 0.160.
- A props scene model costs ~390 MB resident and `props_scene` caches every model for the process
  lifetime; after `cache_clear()` a rebuild is bit-identical (all 434 compiled arrays equal for 1-,
  2- and 3-object scenes). P2 workers therefore hold a bounded number of models and rebuild the rest.
- LeRobot 0.4.4 `aggregate_datasets` copies each video file byte for byte when files are kept one
  per episode (P1's setting), rewrites the data parquet with shifted episode/frame/task indices, and
  re-aggregates statistics.

## Task

- One target object and 1–2 distractor objects, all from the six props, all distinct, all in P1's
  start region (x ∈ [0.27, 0.40], y ∈ [−0.12, 0.12], radius ≥ 0.285 m; yaw ±90°, lemon ±75°).
- Instruction `put the <object> in the <colour> bin`; the bins swap slots between episodes.
- Held out of training entirely: `milk → blue` and `bread → orange` (as targets). Every object still
  appears as a distractor.
- Spacing: every pair of objects at least `r_a + r_b + 0.03 m` apart (centre to centre, `r` the
  footprint half-diagonal above, rounded up to the mm); additionally every distractor at least
  `0.065 m + r_d` from the target, so the open fingers (8 cm gap plus pads) clear it. Layouts are
  drawn by rejection; the rejection count is recorded in the plan.

## Expert

`props_scene.expert.PropsExpert`, imported unchanged, with one override in a P2 subclass: during
`lift` and `move` the held object's lowest point is carried at **0.19 m** (tallest prop 0.16 m +
0.03 m) instead of P1's wall + 0.05 = 0.12 m. Release height, grasp, phase table and durations are
unchanged, so episodes stay 452 frames.

## Admission of one episode

P1's gates unchanged (expert success with 30 stable ticks, 452 frames, replay < 1e-3 on the arm
**and every prop**, labels, schema), plus **`distractors`**: every distractor ends within 0.01 m (xy)
of where it started after settling and tilts less than 5°. A rejected attempt retries the same slot;
after 3 attempts the layout is resampled (target and distractors), up to 3 substitutions; rejection
budget 150, as P1.

## Dataset

- 20 trained cells (10 trained pairs × 2 bin arrangements) × 24 = **480 training**, each cell 12 with
  one distractor and 12 with two. **60 validation** (20 cells × 3), 30 with one and 30 with two.
  **540 admitted**.
- Distractor identities from a shuffled deck per (cell, count), so each other object appears about
  equally often; tested to be within ±15% of the mean count per object.
- Schema as every dataset here (`experiments.paired_demonstrations.media`): state float32 (9),
  action float32 (8), two 512 × 512 videos at 30 Hz, `task` the instruction.

## Reserved evaluation layouts (never collected): 120

| tier | layouts | composition |
|---|---|---|
| screen | 20 | one per trained cell, distractor count alternating 1/2 |
| report | 60 | 20 trained × 1 distractor + 20 trained × 2 + 20 held-out (5 per held-out cell, 1/2 alternating) |
| control | 20 | one per trained cell, **0 distractors** |
| confirm | 20 | one per trained cell, 1/2 alternating; untouched until the winner |

All 120 are checked for expert feasibility before any policy rollout.

## Collection: parallel shards

- 12 workers (`--shard k --shards 12`), each owning a contiguous block of 45 plan slots, with its own
  journal and its own LeRobot dataset under `<root>/shards/shard_<k>/`, each resumable on its own.
  Rejections retry inside the worker, so slot order within a shard is plan order.
- `llvmpipe` thread count set per worker (`LP_NUM_THREADS`) so 12 workers do not oversubscribe 48
  cores; its effect on pixels is checked (must be byte-equal to the default) before it is used.
- Merge (`--merge`): `aggregate_datasets` over the shards in shard order, one file per episode.
  Verified against the shards before sealing: episode count and order, frames per episode, task per
  episode, every data column except the shifted index columns, and every video file by sha256.
- Seal over the merged root (dataset, per-episode traces, plan, provenance, journals). Validation as
  P1 plus the distractor balance and count split.
- Estimate: 540 episodes × ~60 s / 12 workers ≈ 45 min plus rejections and merge; budget 1.5 h.

## Gates before the full run

1. **Expert re-test:** 288 scripted attempts (24 cells incl. held-out × counts {1, 2} × 6), run
   without rendering (rendering does not feed back into physics). Bars: ≥ 95% admitted overall, every
   object ≥ 90% as target, both bin slots ≥ 93%, distractor-undisturbed counted inside admission. Plus
   12 rendered attempts re-run in a second process: byte-equal traces and images.
2. **Pilot:** 2 shards × 6 episodes through collect → merge → validate on a scratch root; watched.

## Training and evaluation (separate plans, after the seal)

- Training: fork of `experiments/xvla_props_p1/`, base X-VLA, P1's recipe, 5 epochs (~126,900
  updates if frames are as planned), resumable, `--ulimit nofile=65536:65536`, large `/tmp`.
- Evaluation: fork of `experiments/xvla_props_p1_eval/`; failure classifier written and checked on
  successes before the first rollout; new cause **`wrong_object`**: the first object lifted more than
  2 cm is not the named one. Screen all 6 checkpoints; report on those within 2 successes of the best
  screen; control (0 distractors) and the P1 cross-run (checkpoint 126,900 on the 120 report trials);
  confirm once on the winner. Parallel evaluation workers where GPU memory allows.

## Success (pre-registered; all four required, report tier, 1–2 distractor trials)

1. Overall success **≥ 70%**.
2. Wrong-object rate **≤ 5%**.
3. Held-out pairs within **10 points** of trained pairs.
4. The P2 policy beats P1's checkpoint 126,900 on the same 120 report trials with **non-overlapping
   layout-clustered 95% intervals**.

Reported alongside, not gated: success by distractor count (0 / 1 / 2), by bin arrangement, wrong-bin
rate, failure causes.

## Out of scope

The near-base corner (start region unchanged), GPU rendering, multi-object sequential sorting,
categories (P3), new objects.

## Plans

Collection: [../plans/2026-10-13-props-p2-collection.md](../plans/2026-10-13-props-p2-collection.md).
Training and evaluation get their own plan after the seal.
