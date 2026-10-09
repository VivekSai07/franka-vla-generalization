# Props P1 collection — 2026-10-07

Branch `feat/props-p0`, nothing pushed or merged. Package `experiments/props_p1_data/` on top of
the accepted, frozen `experiments/props_scene/` (P0). Output:
`outputs/props_p1_data/props_p1_20261006/` (not committed). **Robosuite 1.5.2 / MuJoCo 3.4.0 image**;
numbers are not comparable with the sealed MuJoCo 3.13.0 pipeline. The package is **hash-bound now
that it is sealed** — do not edit `experiments/props_p1_data/*.py` or its README; fork instead.

## P0 re-confirmation (prerequisite, done before any P1 code)

Region amended to radius ≥ 0.285 m (not just x ≥ 0.27) and lemon yaw capped at ±75°. Full 288-attempt
gate re-run on the amended region: **286/288 = 99.3%**, every object ≥ 95.8% (lemon 46/48, all others
48/48), both target slots 143/144, replay exact, determinism 12/12 across all six objects and both
arrangements. Verdict: go. Recorded as a dated note in
[PROPS_P0_RERUN_20261005.md](PROPS_P0_RERUN_20261005.md).

## Pilot

16 slots (every object × both bin arrangements, a lemon at 72.3° — inside the new ±75° cap — and a
start at radius 0.285–0.295 m, the floor the amendment drew): **16/16 = 100% admitted, 0 rejections**.
Frames from three episodes (cereal→orange, bread→blue, can→blue) inspected: object correctly placed
in the instructed colour's bin, both bin arrangements shown. `outputs/props_p1_data/pilot_20261006/`,
never sealed (scratch only, per design).

## Collection result

**540 / 540 slots admitted. Seal verified independently in a second process** (recomputed
`fingerprint` equals `seal.json`'s): `86e3aadb678963eda8cb27d0104d6e22778d4cc25c497829e8fb47937ac53bcf`.

| | planned | collected |
|---|---|---|
| training episodes | 480 (20 trained cells × 24) | 480 |
| validation episodes | 60 (20 trained cells × 3) | 60 |
| held-out pairs collected | 0 | 0 |
| cells off balance | 0 | 0 (every trained cell exactly 24 train / 3 validation) |

Training set, by construction and confirmed from the sealed journal:

| | | |
|---|---|---|
| per object | cereal 96, lemon 96, bottle 96, can 96 (2 trained pairs each) · bread 48, milk 48 (1 trained pair each, the other held out) | |
| per bin colour | blue 240, orange 240 | |
| per bin arrangement | left 240, right 240 | |
| per (object, bin) trained pair | 48 each (24 × 2 arrangements), all 10 pairs | |

`milk → blue` and `bread → orange` do not appear anywhere in the 540 collected episodes (checked
directly against the journal, not just trusted from `validation.json`) — they exist only in the
reserved report-tier evaluation layouts, never executed by the collector.

### Rejections and substitutions

**7 of 547 attempts rejected (1.3%), all `lemon`, all gate `expert_success`** (the P0 success rule,
not reached within 900 ticks) — no `frames`, `replay`, `labels` or `schema` rejection occurred.

| slot | object → bin | attempts rejected | resolution |
|---|---|---|---|
| 74 | lemon → orange | 3 (every draw at that start) | substituted to a fresh start (`r=1`), admitted on its first attempt there |
| 261 | lemon → blue | 3 | substituted to a fresh start (`r=1`), admitted on its first attempt there |
| 515 | lemon → blue | 1 | admitted on the next draw at the same start |

Two substitutions, recorded in `substitutions.json`: `{"74": 1, "261": 1}`. The rejection budget (150)
was never close to being tested. This matches the P0 re-run's finding that lemon is the one
consistently harder object (95.8% there too) — expected, not a new problem, and the collector's
reject-and-resample design handled it exactly as designed.

### Wall-clock

**True elapsed time: 9 h 38 m** (container started 18:54:22, finished 04:32:47, both CEST). The
per-attempt `wall_seconds` recorded in the journal sums to only 6 h 04 m (median 40.2 s, max 43.4 s)
— that figure times only the scripted attempt itself (`execute()`), not the per-episode commit
(writing 452 frames, encoding two AV1 videos single-threaded, `finalize()`), which runs afterward and
is not separately timed. The ~3.6 h difference is this per-episode write-and-encode cost, about
24 s/episode over 540 episodes. **The plan's "~6 hours" estimate was derived from attempt time alone
and undercounts real wall-clock by roughly 1.6×** — corrected in the documentation pass below; a
future collection of this size should budget nearer 1.1 minutes per admitted episode end-to-end, not
40 seconds.

### Disk

Dataset (video + parquet) 2.9 GB, per-episode diagnostic traces (`episodes/*.npz`: qpos, ctrl,
props_pose, initial/final state) 62 MB, `plan.json` 500 KB, `journal.jsonl` 140 KB — **3.0 GB total**.
96 GB was free before the run; 91 GB free after.

### Visual check

Frames from five episodes inspected directly from the sealed dataset (not the pilot): both
substituted slots (74, 261), the one simple re-attempt (515), and two more (cereal→orange, can→orange).
All five: the object ends inside the bin whose colour the instruction names, regardless of which
physical slot that colour is in. Confirms the substitution mechanism produced a normal, correctly
labelled episode and not a near-miss admitted on a technicality.

### What was surprising

- The wall-clock gap above — the only finding worth correcting in the record.
- Nothing else: no gate besides `expert_success` ever rejected an attempt, no object other than lemon
  was ever rejected, and the collector's logs show only the expected one-time robosuite/SVT-AV1
  startup warnings (1,080 "Preset M12 is mapped to M10" lines, 2 per episode, harmless encoder notices)
  — no error, traceback, or unexpected pause.

## What this does not show

- No policy has been trained on this dataset yet.
- The reserved evaluation layouts (80: 20 screen, 40 report, 20 confirm) have not been checked for
  expert feasibility — P0's gate covered the trained-cell region generally, not these specific drawn
  layouts. Screen them with the expert before the first rollout, the same way B3 screened its holdout.
- Lemon's rejection rate here (7/547 attempts, all lemon) is consistent with P0 but is not large enough
  on its own to say more than "lemon is the hardest object" — already known.

Decision: collection complete and sealed; ready for P1 training and evaluation design.
