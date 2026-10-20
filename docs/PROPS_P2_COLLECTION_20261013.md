# Props P2 collection — instruction sorting with distractors — 2026-10-13

A dated report; do not edit it to reflect later findings. Package `experiments/props_p2_data/` on branch
`feat/props-p2` (commit `ab11849`), robosuite image (`sha256:54d861aa…ccf27e`, robosuite 1.5.2 / MuJoCo
3.4.0), CPU only. Spec: [superpowers/specs/2026-10-13-props-p2-distractors-design.md](superpowers/specs/2026-10-13-props-p2-distractors-design.md);
plan: [superpowers/plans/2026-10-13-props-p2-collection.md](superpowers/plans/2026-10-13-props-p2-collection.md).
Both are now hash-bound by this seal (FAILURES #52) — do not edit them.

## Result

**540 / 540 slots admitted, 0 rejected, 0 substitutions. Sealed** at
`outputs/props_p2_data/props_p2_20261013/`, fingerprint
`b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`.

| | |
|---|---|
| episodes | 480 train + 60 validation; 244,080 frames (540 × 452) |
| balance | 20 trained cells × 12 one-distractor + 12 two-distractor training episodes; validation 3 per cell, 30/30 |
| distractor counts | 270 one-distractor, 270 two-distractor |
| held-out pairs collected | 0 (`milk→blue`, `bread→orange`) |
| distractor touched (any substep, anything but the table) | 0 of 540 |
| reserved evaluation layouts | 120, never executed (screen 20, report 60, control 20, confirm 20) |
| disk | sealed root 3.2 GB; the 12 shard datasets another 3.1 GB (`props_p2_20261013_shards/`, kept) |

**Seal re-verified in a separate process** (admission image): the fingerprint recomputed over the sealed
root equals `seal.json`; the plan digest and the provenance digest equal the seal, and `provenance.json`
equals the current source; `validate_dataset` passes again; 540 per-episode traces present;
`LeRobotDataset` loads 540 episodes / 244,080 frames / 10 tasks and spot-checked frames at episodes 0,
100, 300, 480 and 539 carry their plan slot's instruction.

## Wall-clock: 54 minutes

12 shard containers, 45 contiguous slots each, `LP_NUM_THREADS=4`: first start 20:10:30, last finish
21:04:26 CEST. Serial-equivalent work recorded in the journal: 7.04 h of attempts (median 46.9 s under
full load, 41 s alone) plus 3.75 h of frame write + AV1 encode + finalize (median 24.8 s) — **10.8 h of
work in 54 min**, against P1's 9 h 38 m for the same 540-episode budget. Memory held at ~4 GB per worker
(scene models capped at 4 per worker; P2 meets up to 45 object sets per worker at ~390 MB each). The
merge, its verification against the shards, validation and the seal took 63 s.

## Merge

`lerobot.datasets.aggregate.aggregate_datasets` with file size 0 (one file per episode, as P1).
`merge.verify` compared every merged episode with its shard episode: length, task string,
`observation.state`, `action`, `timestamp`, `frame_index`, contiguous `episode_index`/`index`, and both
video files by sha256 and timestamps — **540 episodes, 1,080 videos byte-identical**. Statistics in
`meta/stats.json` are LeRobot's aggregation of the shards' statistics, not a single pass over all frames.

## Gates before the run

1. **Expert re-test, first run — failed, 261/288 = 90.6%** (bar 95%;
   `outputs/props_p2_data/gate_20261013/`). 20 of 27 rejections: a tall neighbour struck by the hand
   housing during descend while grasping a short target (housing ±0.104 m along the finger axis,
   underside 0.042 m above the grasp point); one by link 7; 7 expert failures. The bar was kept; a
   symmetric hand-clearance rule was added to the layout sampler (FAILURES #55, spec amendment).
2. **Expert re-test, second run — passed** (`gate_20261013b/gate_summary.json`): 284/288 = 98.6%;
   per object 100% except lemon 22/24 = 91.7% (bar 90%); left/right blue slot 98.6% each (bar 93%);
   one/two distractors 97.9% / 99.3%; 0 distractor contacts; failures 3 `grasp`, 1 `lost_in_carry`.
   12 rendered attempts run in two processes (`LP_NUM_THREADS=4` and default) were **byte-equal,
   images included** — determinism holds and the per-worker thread cap does not change pixels.
3. **Pilot:** 2 shards × 6 slots, 12/12 admitted; merged and verified (24 videos byte-identical);
   `LeRobotDataset` loaded it with tasks in plan-slot order.

## Findings

- **The lemon moves on its own.** Untouched lemon distractors drifted up to **18.0 mm and tilted up to
  51.4°** across 123 episodes; every other prop stayed within 0.2 mm and 0.3° (bottle, bread, can,
  cereal, milk). This is why the distractor gate is contact-based, not pose-based (FAILURES #54), and
  it matters for evaluation: a policy's `wrong_object` must be decided by which object is lifted, never
  by a distractor having moved.
- **Inherent imbalance, by design:** milk and bread are targets in half as many trained cells (their
  held-out pairs), so when either is on the table it is a distractor 74% of the time (~55% for the
  others). Distractor appearances per object in training are within ±13% of the mean.
- Layout spacing is tight: median 11 rejection draws per layout, worst 9,915 (limit 100,000).

## What this does not show

No policy result. Robosuite path, so not physics-comparable with the MuJoCo-scene datasets. The near-base
corner is not covered (start radius ≥ 0.285 m, as P1). The 120 reserved layouts were checked by the plan
(region, spacing) but **not yet** for expert feasibility — that runs before any rollout (spec).

## Next

Training (fork of `experiments/xvla_props_p1/`, base X-VLA, P1's recipe, 5 epochs) and evaluation
(fork of `experiments/xvla_props_p1_eval/`, `wrong_object` cause, P1 cross-run, control tier), each with
its own plan, against the success criteria pre-registered in the spec.
