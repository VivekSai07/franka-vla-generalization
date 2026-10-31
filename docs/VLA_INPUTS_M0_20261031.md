# VLA inputs M0: common evaluation protocol and the control arm A0 — 2026-10-31

**Status: A0 (the P2 X-VLA checkpoint 101,520, language and the two cameras only) has been run on the full
comparison tier: 451 of 573 layouts succeeded, 78.7%.** Branch `feat/vla-inputs`, packages
`experiments/vla_inputs_eval/` (shared core, frozen and hash-bound) and `experiments/vla_inputs_arm_xvla/` (the arm
adapter, bound separately), driver `scripts/vla_inputs_eval.sh`. Spec: [design](superpowers/specs/2026-10-30-vla-inputs-m0-evaluation-design.md);
plan: [plan](superpowers/plans/2026-10-30-vla-inputs-m0-evaluation.md). Upstream: [S0 sidecar](VLA_INPUTS_S0_20261030.md),
[audit](VLA_INPUTS_AUDIT_20261030.md). Raw outputs: `outputs/vla_inputs_eval/m0_20261030/` (not committed).

This is a measurement of one checkpoint on one set of layouts. It is the baseline every later arm (A1–A4) is compared
against; it is not a finding about input signals.

## What was run

| | |
|---|---|
| model | P2 X-VLA, checkpoint 101,520 (the checkpoint P2's report tier scored 90/120), weights sha256 `59d02d0a…75932e` |
| tier | comparison C: 48 cells (6 objects × 2 bins × blue slot × 1–2 distractors) × 12 layouts = 576 planned, **573 feasible**; one trial per layout |
| plan | `plan/plan.json`, sha256 `0184c4d7b3b72c55e10ce1946cbd50b2db9092dd05d137c4196f964f8b831715`; identities `vla_inputs:m0:<tier>:<cell>:<i>`, a namespace no earlier plan used |
| image | robosuite image `sha256:54d861aa…27e` (robosuite 1.5.2, MuJoCo 3.4.0, OSMesa rendering), 8 workers on one GPU, one checkpoint per worker pool |
| bindings | shared core `c4f6c303f2ae`, arm `75104fca7bf7`, S0 seal `7e2e06db6f71`; checkpoint weights verified against their marker; `compare` re-hashes everything deeply |
| primary test (unrun until A1–A4 exist) | exact two-sided McNemar on C, α = .05, paired bootstrap stratified by object (10,000, seed 0), Holm over the fixed family A1–A4 |
| policy output | a non-finite action counts as a failure (0 occurred); a completed trial is never re-run (none was) |

## Feasibility (before any policy trial)

An unrendered expert replay decided which layouts can be solved at all, so no arm is charged for an impossible layout.
**652 of 656 are feasible; 4 are dropped for every arm.** Comparison: 573 of 576 (`bread-blue-right-1` 11/12 — one `grasp`;
`lemon-orange-right-2` 10/12 — one `grasp`, one `lost_in_carry`). Selection tier (trained cells only, used later to choose
checkpoints for new arms): 79 of 80 (`lemon-blue-left-2` 1/2, `grasp`). 573 is therefore the number every later comparison
is paired over.

Power at N = 573 was recomputed before any policy outcome (exact enumeration; Monte Carlo of 20,000 repetitions agrees
within 0.0012). Probability to detect a gain at Holm-corrected α = .0125, for discordant-pair fraction D:

| gain | D = .19 | D = .30 | D = .40 |
|---|---|---|---|
| 10 points | 0.999 | 0.969 | 0.895 |
| 8 points | 0.972 | 0.831 | 0.684 |
| 6 points | 0.771 | 0.524 | 0.387 |

At A0's measured 78.7% the largest possible gain is 21 points, so the design can resolve 8–10 point improvements at
D ≤ .30 and cannot resolve 6.

## Equivalence gate and the one core change

- **Equivalence gate.** The new rollout and X-VLA adapter reproduce the two lowest-ordinal successes and the two
  lowest-ordinal failures of P2's recorded report trials of 101,520 byte for byte (702, 719, 900, 900 ticks), and an
  expert replay through the real loop records the S0 signals byte-equal to the sealed sidecar.
- **Core change before the run.** The first preparation (core `3cc8c6239067`, arm `be3412b23c08`) did not bind the X-VLA
  base bundle (`.cache/xvla_trial/bundle`, 30 files, 3.5 GB: tokenizer files and base config) that the adapter reads at
  rollout. On the user's decision it is now bound through a new `INPUT_DIRS` declaration (files below 64 MB fully hashed,
  larger ones by size and mtime; `compare` re-checks). Only `experiments/vla_inputs_eval/evaluate.py` changed
  (core `3cc8c6239067` → `c4f6c303f2ae`; arm `be3412b23c08` → `75104fca7bf7`); weights and dropped layouts are equal. No
  comparison-tier trial had run before the change. The old smoke (2 trials, a cost check) ran under the old core and is
  archived in `arms/A0/superseded_core_3cc8c6239067/`, not repeated. Record: `core_change.json`.
- A closure test now fails if the adapter reads anything under `.cache` or `/bundle` outside the checkpoint and the
  declared inputs.

## Result: A0 on the 573 comparison layouts

Intervals are 95% bootstrap intervals clustered by layout (one trial per layout, so equal to a plain bootstrap).

| | n | success | 95% interval |
|---|---|---|---|
| **overall** | 573 | **451 (78.7%)** | 75.4 – 82.0 |
| trained pairs | 477 | 384 (80.5%) | 76.9 – 84.1 |
| held-out pairs (milk→blue, bread→orange) | 96 | 67 (69.8%) | 60.4 – 79.2 |

### Per object (both bins)

| object | n | success | 95% interval |
|---|---|---|---|
| can | 96 | 85 (88.5%) | 81.3 – 94.8 |
| bread | 95 | 84 (88.4%) | 82.1 – 94.7 |
| bottle | 96 | 80 (83.3%) | 76.0 – 90.6 |
| cereal | 96 | 78 (81.3%) | 72.9 – 88.5 |
| lemon | 94 | 73 (77.7%) | 69.2 – 86.2 |
| **milk** | 96 | **51 (53.1%)** | 42.7 – 63.5 |

### Object × bin, with trained/held-out marked

| pair | n | success | | pair | n | success |
|---|---|---|---|---|---|---|
| can→blue | 48 | 44 (91.7%) | | lemon→blue | 48 | 33 (68.8%) |
| can→orange | 48 | 41 (85.4%) | | lemon→orange | 46 | 40 (87.0%) |
| **milk→blue (held out)** | 48 | 26 (54.2%) | | bottle→blue | 48 | 40 (83.3%) |
| milk→orange | 48 | 25 (52.1%) | | bottle→orange | 48 | 40 (83.3%) |
| bread→blue | 47 | 43 (91.5%) | | cereal→blue | 48 | 41 (85.4%) |
| **bread→orange (held out)** | 48 | 41 (85.4%) | | cereal→orange | 48 | 37 (77.1%) |

The 11-point gap between trained and held-out pairs is mostly which objects the held-out pairs contain, not the pairing:
milk is half the held-out trials and is weak in **both** bins (54.2% held out, 52.1% trained). Within an object the
held-out minus trained contrast is +2 points for milk and −6 for bread, each with an interval of about ±15 points.

### Other splits

| split | success (n) |
|---|---|
| 1 distractor | 234 of 287 (81.5%, 77.0 – 85.7) |
| 2 distractors | 217 of 286 (75.9%, 70.6 – 80.8) |
| blue bin on the left | 223 of 288 (77.4%, 72.6 – 82.3) |
| blue bin on the right | 228 of 285 (80.0%, 75.4 – 84.6) |

Per cell (12 layouts each; 48 cells, listed in `a0_comparison_summary.json`): 25 of 48 cells have 10 or more successes
and 6 are clean sweeps. The weakest are `milk-blue-left-1` 5/12, `milk-orange-right-2` 5/12 and `milk-orange-left-1` 6/12,
then seven cells at 7/12: `lemon-blue-left-2`, `lemon-blue-right-2` and five milk cells. A cell has 12 trials: read cells
as a map, not as estimates.

### Failure causes (122 failures)

P2's classifier, imported unchanged.

| cause | all | trained | held out |
|---|---|---|---|
| success | 451 | 384 | 67 |
| never lifted | 59 | 49 | 10 |
| timeout while carrying | 36 | 26 | 10 |
| dropped outside the bin | 11 | 7 | 4 |
| wrong bin | 7 | 7 | 0 |
| unstable in the target | 5 | 3 | 2 |
| wrong object | 4 | 1 | 3 |

- Every failure ran to the tick budget (122 timeouts, 0 non-finite outputs). Median ticks 455 overall, 412 for successes.
- **The failures are mostly grasp-and-carry, not instruction following.** 95 of 122 are never-lifted or timeout-carrying;
  wrong bin and wrong object together are 11 (1.9% of all trials; wrong-object 0.7% [0.2, 1.4], all four in two-distractor
  layouts).
- Milk accounts for 22 of the 59 never-lifted and 14 of the 36 carrying timeouts; lemon for 13 never-lifted; bottle's
  failures are mostly timeouts while carrying (8) and wrong bin (5).

### Signals recorded during the rollouts (S0, computed on a scratch copy; descriptive)

Every trial records wrist wrench, finger forces and the target box in both cameras (573 trials, all finite).

- **Finger contact separates failure kinds.** Fraction of ticks with both fingers in contact: success 0.64, timeout while
  carrying 0.48, dropped outside 0.30, never lifted 0.28. Median peak finger force: success 14 N, carrying timeout 28 N,
  dropped outside 28 N, **never lifted 82 N**, wrong object 53 N (median peak wrist force 301 N). Never-lifted trials
  press hard on something for a short time; that is a signature, not a mechanism.
- **The in-frame bit is informative on rollouts, although it was constant on the demonstrations** (S0: exactly 1 in every
  segment). The front camera's target box was in frame on every tick of every trial (minimum 1.0). The gripper camera's
  target box left the frame at some tick in **26 of 122 failures and 0 of 451 successes**, mean in-frame rate 1.00 on
  successes and 0.93 on failures (wrong-object trials 0.56). Whether the box leaves the frame because the hand went the
  wrong way or the other way round is not decided by this measurement.

## Next to P2's report tier (context only)

| | success |
|---|---|
| A0 on C (this report, 573 new layouts, 1 trial each) | 78.7% (451/573) [75.4, 82.0] |
| P2 report tier, same checkpoint (60 layouts × 2 seeds) | 75.0% (90/120) [65.0, 84.2] |
| P2 report tier, trained / held-out pairs | 85.0% (68/80) / 55.0% (22/40) (this report: 80.5% / 69.8%) |
| P2 report tier, wrong-object trials | 7 of 120 (this report: 4 of 573) |
| P2 re-test, fresh layouts, milk → trained / held-out bin | 57.6% / 56.8% (this report: 52.1% / 54.2%) |
| P2 re-test, bread → trained / held-out bin | 80.8% / 76.7% (this report: 91.5% / 85.4%) |

The layouts, plans and trial counts differ and nothing here is a test. What reproduces is the object ordering (milk weak
in both bins, bread strong) and the P2 re-test's reading that the held-out bin is not clearly worse than the trained bin
for the same object. The held-out gap is smaller here (10.7 points against 30 in P2's 40 held-out trials) and, as above,
is mostly object composition; wrong-object failures are rarer here (0.7% against 5.8%).

## Resources and the thermal guard

- One worker process: about 4.15 GB GPU (torch reserved 3.6 GB), host RSS up to 9.3 GB; trial wall time median 61 s,
  90th percentile 115 s, maximum 206 s with 5 pinned cores. Eight workers peaked at 33.6 GB of the 48 GB card and 84 °C.
- The run took 2 h 46 min (11:26 – 14:12, 2026-10-31); outputs 817 MB with videos; disk 108 GB free afterwards. GPU idle
  afterwards (363 MiB).
- **The guard lost about an hour of wall time.** A worker that finds the GPU at 83 °C waits until it is at 75 °C, but with
  the other seven still running the card stayed at 76–81 °C, so three workers sat paused for 4,440 s, 3,810 s and 3,345 s
  (two more paused 150 s and 75 s). They resumed only once the other workers had finished their shares (12:42 – 12:57).
  This changed time, not outcomes: a pause happens before a trial starts, and the three workers' trials are complete and
  identical in kind. See [FAILURES.md](FAILURES.md) #66.

## Caveats

- **One training seed and one checkpoint.** 101,520 is the P2 report checkpoint. Any later arm is a different training run
  with its own seed, so a difference between arms mixes the signal with seed-to-seed variation; the design cannot
  separate them.
- **Conditional on these layouts.** 573 layouts from a fixed plan, one trial each; the intervals describe layout sampling,
  not training variation. The 4 infeasible layouts are dropped for every arm.
- **Not comparable** with MuJoCo-pipeline numbers (B1/B2) or with P2's tier sizes; the P2 rows above are context.
- Held-out pairs are two (milk→blue, bread→orange): "held out" here is two pairs, not a population of unseen pairs.
- A scratch probe (A0 on the 79 selection layouts, 8 workers, 60 of 79 successes) was used for the resource measurements
  only; it is not part of the protocol and its root is `m0_20261030_probe`.

## What is next

- A1–A4 (force, boxes, outputs) each need a trained checkpoint on P2's data with the new input; M1 (pi0.5) is a separate
  arm family. Run them through `scripts/vla_inputs_eval.sh` (`prepare` → `stage` → `select` → `compare`); the 4.0-epoch
  matched-phase checkpoint is declared at `prepare`.
- **Before the next 8-worker run** decide how to handle the guard: fewer workers (the 8-worker probe and this run both
  reach 84 °C), or a cross-worker guard. The second is a core change and would be recorded like the one above; it is not
  made here.
