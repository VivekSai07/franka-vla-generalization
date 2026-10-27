# Props P3 (Food vs Drinks) training and evaluation, 2026-10-22 to 2026-10-27

**Status: failed attempt. The user closed P3 on 2026-10-27 and deleted its dataset, checkpoints, gate scratch and evaluation outputs
(never copied to Windows or published). The figures below are the record; they can no longer be re-checked.**
Dated report; not to be rewritten. Live figures: [RESULTS.md](RESULTS.md).

## What was run

- Dataset: [PROPS_P3_COLLECTION_20261020.md](PROPS_P3_COLLECTION_20261020.md) (544 episodes, seal `6430d34e…d151acf`; trained on can, milk, bread,
  lemon; bottle and cereal never collected; fixed prompt `sort the item`; drinks to the orange bin, food to the blue bin, sides swap).
- Training: `experiments/xvla_props_p3/`, X-VLA, effective batch 8, 126,900 updates = 5.0 epochs, checkpoints 1, 50,760, 76,140, 88,830, 101,520,
  114,210, 126,900. Gates passed 2026-10-22: `--check`, `--preflight` (23 GB scratch), `--resume-proof` (metrics, model and training state equal),
  `--equivalence` (30 updates equal). Three batches (2026-10-22 19:00 to 10-23 06:00, 10-25 19:16 to 10-26 08:00, 10-26 15:01 to 23:05), one resume
  per batch from the last snapshot, finished `complete`. Effective pace about 0.9 s/update (raw 0.52; 1,000+ GPU cool-down pauses at 83 C).
- Evaluation: `experiments/xvla_props_p3_eval/`, 288 reserved layouts, one seed each; expert feasibility 287 of 288 (`report_062`, a bottle layout,
  dropped and counted). Screen 24 layouts over all six checkpoints, report 240 layouts (239 used) over the four survivors, confirm 24 layouts on the winner.
  8 parallel workers (5 pinned cores, about 4.2 GB GPU each); about 6 h of rollouts, no worker failure, no retry. Smoke: 86 s per trial.

## Results

Screen successes (of 24): 50,760: 17; 76,140: 16; 88,830: 20; 101,520: 22; 114,210: 22; 126,900: 22. Survivors (within 2 of the best): the last four.

Report tier (successes; trained objects n = 80, held-out n = 159, milk n = 20, cereal n = 80, bottle n = 79):

| checkpoint | trained | held-out | all | milk | cereal | bottle |
|---|---|---|---|---|---|---|
| 88,830 | 68 | 10 | 78 | 12 | 10 | 0 |
| **101,520** | **68 (85.0%)** | **9 (5.7%)** | **77 (32.2%)** | 10 | 9 | 0 |
| 114,210 | 66 | 8 | 74 | 8 | 8 | 0 |
| 126,900 | 66 | 17 | 83 | 8 | 16 | 1 |

Winner 101,520 by the plan's rule (highest report success on trained-object layouts; 88,830 tied at 68 and the later checkpoint wins). Confirm tier
(24 fresh layouts, selected on nothing): **22/24 = 91.7%** (Wilson 74.2-97.7%; can, bread, lemon 6/6, milk 4/6).
Per trained object at 101,520: can 19/20, bread 20/20, lemon 19/20, milk 10/20. By blue-bin side: left 41/119, right 36/120 (not different).

Pre-registered criteria at 101,520:

1. Trained wrong-bin at most 5% of stable placements (floor 60): **pass**, 68 stable, 68 correct, 0 wrong (Wilson upper bound of the wrong rate 5.3%).
   Same 0 wrong at the other three survivors (68/68, 66/66, 66/66).
2. Held-out generalisation (Wilson lower bound above 0.5 for both objects, floor 30 stable per object): **not interpretable**. Bottle has 0 stable
   placements (floor not met). Cereal has 31 stable placements, 9 correct, 22 wrong (accuracy 29%, Wilson 16-47%).

Failure causes at 101,520 (162 non-successes of 239): never lifted 120 (bottle 77, cereal 36, milk 6, lemon 1), wrong bin 22 (all cereal), timeout
while carrying 12, dropped outside 9. Across the four report checkpoints the causes are similar (never lifted 108-120, wrong bin 18-24, timeout while
carrying 12-23).

## What the traces show (descriptive, not a verified cause)

From `trace.npz` (object position, finger contact, gripper). The measures were first checked on the successes: every success has finger contact and a
lift of more than 2 cm (median 11.6-12.1 cm for can, milk, bread, lemon), so the lift measure registers a carry.

- **Bottle (held out):** about 300 of 320 trials across four checkpoints are never lifted. Finger contact occurs in 91% of them but covers a median 8%
  of the ticks; the gripper reads closed for 89% of the time; the bottle is displaced a median 7.6 cm (98% move more than 2 cm). Reaching and pushing,
  no grasp. Layout is not the factor (uniform failure), and the expert was feasible on 79 of 80 bottle layouts.
- **Cereal (held out):** 120 of about 320 never lifted; of the placements that came to rest in a bin, 81 wrong-bin against 43 correct pooled over four
  checkpoints (65% wrong). Pooled counts reuse the same layouts, so they are not independent. Blue-bin side does not explain it (wrong-bin 45 left, 36 right).
- **Milk (trained):** 12, 10, 8, 8 of 20 across the four report checkpoints and never lifted 3, 6, 7, 11. In a never-lifted milk trial the finger contact
  covers a median 89% of the ticks and the carton moves a median 1.6 cm: the gripper closes on the carton and does not lift it. The failures are scattered:
  per layout successes over the four checkpoints are 0 to 4, three layouts fail at every checkpoint, and the side effect flips between checkpoints
  (6/10 vs 6/10, 7/10 vs 3/10, 1/10 vs 7/10, 2/10 vs 6/10). The decline 12 to 8 over 20 layouts is not significant.
- Can, bread, lemon: 19/20, 20/20, 19/20.

## What was not done

No expert replay on the failing layouts, no interference measure, no video review, no hand-frame grasp geometry, no paired comparison of checkpoints
(see [diagnosing-policy-failures](../.github/skills/diagnosing-policy-failures/SKILL.md)). The causes above are descriptions; no cause is claimed.

## Limitations and caveats

- One seed per layout; the four report checkpoints share the same layouts, so their counts are not independent.
- The checkpoint-selection rule (trained-object layouts only) is the author's, written in the plan; the spec did not state one. On all layouts 126,900
  would win (83 against 77, held-out 17 against 9). Trained-object counts 68, 68, 66, 66 are a tie in substance. The trained-object criterion at the
  winner is optimistic because the winner was selected on those layouts; the confirm tier (91.7%) is the unbiased figure.
- Bottle and cereal differ from the trained objects in several respects at once (shape, size, category); this design cannot separate them.

## Hypotheses (not tested)

Held-out failures are grasp acquisition for an unfamiliar shape (bottle) and a bin choice driven by appearance rather than category (cereal, which
resembles the trained milk carton). Milk may be limited by its width relative to the gripper or by too few distinct grasps in 480 episodes. A next dataset
would vary objects within each category far more than four objects can.

## Process notes

FAILURES #59 (collection launch died), #60 (NaN interval), #61 (disk planning), #62 (selection rule). Disk: the run needed about 110 GB beyond the
dataset because `--train` re-reads the preflight and resume-proof checkpoints; the machine's free space fell from 63 to 30 GiB overnight when
`/var/log/syslog` (root-owned) reached 134 GB, and was only restored by deleting data and later by the syslog being truncated.
