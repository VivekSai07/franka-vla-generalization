# Props P2 X-VLA evaluation on the reserved layouts — 2026-10-16

A dated report; do not edit it to reflect later findings. Measured, not sealed: the evaluation package
(`experiments/xvla_props_p2_eval/`) has no contract. Robosuite image (`sha256:54d861aa…ccf27e`, robosuite 1.5.2 /
MuJoCo 3.4.0), GPU for rollouts. Training: `outputs/train/xvla_props_p2_v1` (126,900 updates, P1's recipe on the
P2 seal `b1d642b3…f4d898`). Rules, criteria and tie handling were fixed and committed before any rollout:
[plan](superpowers/plans/2026-10-16-props-p2-evaluation.md) (`842e15c`). Evaluation code: `4d38af7`.

## Verdict against the four pre-registered criteria

Winner chosen by the pre-registered rule: **checkpoint 101,520** (90 report successes against 83 for 126,900).
Criteria are evaluated on the report tier, 120 trials (60 layouts × 2 rollout seeds), against P1's 126,900 on the
same trials. Intervals resample layouts, not trials.

| # | criterion | measured | result |
|---|---|---|---|
| 1 | overall success at least 70% | **90/120 = 75.0%**, 95% [65.0, 84.2] | pass |
| 2 | wrong-object at most 5% | **7/120 = 5.8%**, [1.7, 11.7] | **fail** (by one trial) |
| 3 | trained vs held-out gap within 10 points | trained 68/80 = 85.0% [75.0, 93.8], held-out 22/40 = 55.0% [35.0, 75.0]: **gap 30 points** | **fail** |
| 4 | beats P1 with non-overlapping intervals | P1 126,900: **0/120**, [0, 0] against [65.0, 84.2]; discordant pairs 0 vs 90, McNemar p = 1.6e-27 | pass |

**Two of four pass, so P2's pre-registered success is not met.** No threshold was moved after seeing the numbers.
The wrong-object miss is one trial (6/120 = 5.0% would pass), but the interval spans the threshold and the
criterion is stated as written.

The winner was selected on the report tier, so criteria on that tier are optimistic. The confirm tier, touched once
and selected on nothing: **16/20 = 80%** [60, 95], 0 wrong-object (8/10 with one distractor, 8/10 with two). It does
not show an optimism gap (report 75%), though 20 trials cannot rule out a small one.

## Stages

540 rollouts in total. Wall-clock was about 3 h 5 min from the successful launch (19:08 to 22:13) for 12.0 h of
summed trial time, a 3.9× speedup over serial; median trial 82 s (P1's serial median was 67 s: parallel load and
thermal pauses make trials slower).

| stage | trials | what ran |
|---|---|---|
| feasibility | 120 layouts | scripted expert, unrendered, 12 shards: **120/120 feasible, 0 excluded** |
| screen | 6 × 20 | all non-trivial P2 checkpoints |
| report | 120 × 3 | 101,520 and 126,900 (P2), and P1 126,900 on the same trials |
| control | 20 × 2 | trained cells, no distractor: 101,520 and P1 126,900 |
| confirm | 20 | 101,520, once |

### Screen (20 trials: 10 one-distractor, 10 two-distractor)

| checkpoint | 50,760 | 76,140 | 88,830 | **101,520** | 114,210 | **126,900** |
|---|---|---|---|---|---|---|
| successes /20 | 11 | 10 | 16 | **19** | 16 | **18** |

Survivors under the "within 2 of the best" rule: 101,520 and 126,900. The earlier checkpoints are clearly behind
(10–11/20); 88,830–126,900 are within noise of one another at this size.

### Report (120 trials per checkpoint)

| | 101,520 (P2) | 126,900 (P2) | P1 126,900 |
|---|---|---|---|
| overall | **90/120 = 75.0%** | 83/120 = 69.2% | 0/120 |
| trained cells (80) | 68 (85.0%) | 61 (76.3%) | 0 |
| held-out cells (40) | 22 (55.0%) | 22 (55.0%) | 0 |
| one distractor (60) | 48 (80.0%) | — | 0 |
| two distractors (60) | 42 (70.0%) | — | 0 |
| blue bin on left / right (60 each) | 44 (73.3%) / 46 (76.7%) | — | 0 / 0 |
| wrong-object | 7 | 2 | 5 |
| wrong-bin | **0** | 0 | 0 |

101,520 versus 126,900: 15 trials only the former solved, 8 only the latter, exact McNemar **p = 0.21**. They are
not distinguishable on this evidence; the pre-registered rule picked 101,520 and that is the checkpoint reported.
For information only, since it was not the selected checkpoint: 126,900 has fewer wrong-object trials (2) but
misses criterion 1 (69.2%) and criterion 3 (trained 76.3% against held-out 55.0%).

### Failure causes (first cause, mutually exclusive)

| cause | P2 101,520 | P2 126,900 | P1 126,900 |
|---|---|---|---|
| success | 90 | 83 | 0 |
| never_lifted | 12 | 24 | 110 |
| wrong_object | 7 | 2 | 5 |
| timeout_carrying | 7 | 10 | 3 |
| dropped_outside | 3 | 1 | 2 |
| unstable_in_target | 1 | 0 | 0 |
| wrong_bin | 0 | 0 | 0 |

`wrong_object` means the first object lifted 2 cm for 15 ticks was not the target. The classifier was validated
before any rollout: on the 540 sealed expert episodes the target is always the first object lifted and no
distractor ever is (distractors rise at most 0.02 cm, targets 18–21 cm), and an expert made to carry a distractor
produces `wrong_object` in the real rollout loop.

### Control: no distractor, trained cells (20 trials)

| | P2 101,520 | P1 126,900 |
|---|---|---|
| success | 17/20 = 85% [70, 100] | 14/20 = 70% [50, 90] |

Training with distractors did not cost the old task at this resolution (P1's own report-tier rate was 75%); the
17 against 14 difference is inside the noise of 20 trials. This control also validates the pipeline for P1:
the same P1 checkpoint that scores 0/120 here scores 14/20 on its own task in the same worker, loader and scene
restore.

## What the numbers do and do not say

Measured:

- **Training with distractors turned a policy that never completes the task in a cluttered scene (P1: 0/120) into
  one that completes about three in four (75%).** P1 mostly never lifts anything (110/120), so its failure with
  distractors is not a wrong choice but no pick-up at all. P1 never saw a distractor, so this is the expected
  direction; the size (0 to 75%) is the measured part.
- **Choosing the right object is largely solved, not fully.** 7 wrong-object trials in 120. They are in 5 layouts
  (report_026 and report_059 failed this way on both seeds) and fall mostly on two-distractor layouts (6 of 7 against
  1 of 60 one-distractor trials). The nearest-distractor distance is smaller in these trials (median 0.123 m)
  than in successes (0.150 m); with 7 trials this is suggestive, not tested.
- **Zero wrong-bin outcomes in 120** — the bin choice generalises to the held-out cells as well as to trained ones.
  The physical side of the blue bin does not matter (73% / 77%).
- **Held-out cells are weaker: 55% against 85% (criterion 3 fails).** Per cell: `bread→orange` 15/20 (6/10 with the
  bin on the right, 9/10 left), `milk→blue` **7/20** (4/10 left, 3/10 right). The two held-out cells share
  objects with trained cells, so the comparison is partly confounded by object difficulty: milk is the weakest
  object even when trained (`milk→orange` 5/8 for 101,520; milk overall 12/28, 11/28 for 126,900), and bread is
  perfect when trained (8/8 for `bread→blue`). Both held-out cells are still below their trained counterpart
  (bread 75% against 100%, milk 35% against 62%), but the trained references are 8 trials each and cannot
  separate "unseen pair" from "hard object" at this sample size.
- **Failures concentrate in layouts.** Of 60 report layouts, 40 succeeded on both seeds, 10 on one and 10 on neither;
  7 of the 10 never-solved layouts are held-out (5 milk→blue, 2 bread→orange). In 50 of 60 layouts the two seeds
  agree, so most failures are properties of the layout rather than of the rollout seed.

Not measured, so not claimed:

- Why `never_lifted` happens (12 trials; 8 of them held-out). Grasp-approach failure, perception, and instruction
  grounding are not separated.
- Whether the milk gap is "object" or "pair". A follow-up would hold milk out in a different bin as well.
- Whether the 900-tick cap converts any of the 7 `timeout_carrying` trials (as for P1, not changed).
- Whether 101,520 beats 126,900 in general. They are tied on this evidence.

## Operations

- **Machine health.** GPU peaked at 84 °C; trials were paused before starting while at or above 83 °C until 75 °C,
  4.0 worker-hours of pauses in total (summed per worker). No pause cut a trial.
- **First launch failed and was stopped.** Eight workers sharing all 48 cores thrashed (load average 142, no trial
  finished in 6 minutes). Pinning each worker to 5 cores with matching thread limits fixed it (load about 25). The 8
  half-written folders from the aborted launch were moved to `quarantine/`, not deleted.
- **A worker died of CUDA out of memory during the screen, and the resume absorbed it.** The smoke run measured
  about 4.1 GB per worker, but the out-of-memory message listed four processes at 7.3 GiB each and the 47 GB GPU
  full; these are workers that had already switched checkpoint (the cause of the growth was not isolated). The
  driver's built-in single resume completed the missing screen trials (2-3 per checkpoint), which reseed themselves and
  equal uninterrupted ones. No later stage failed. See FAILURES #57.
- Output: `outputs/xvla_props_p2_eval/` (gitignored): `final.json`, `decisions/`, per-stage `result.json`, per-trial
  `cameras.mp4`, `trace.npz` (with every prop's position), `episode.json`, `telemetry.json`, and the driver and
  worker logs.

## Reproducing

```bash
scripts/xvla_props_p2_eval.sh all 8      # screen -> report -> control + confirm -> final.json
```

`evaluate.py prepare` once first (checkpoints, seals and feasibility verified once, `prepared.json`). The pure
tests run in the admission image; the full 45 in the robosuite image (README of the package).
