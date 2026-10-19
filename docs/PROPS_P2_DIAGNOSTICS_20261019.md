# Props P2 failure diagnostics — 2026-10-19

Follows [PROPS_P2_EVALUATION_20261016.md](PROPS_P2_EVALUATION_20261016.md). Design:
[spec](superpowers/specs/2026-10-19-props-p2-diagnostics-design.md) (with its dated amendment) and
[plan](superpowers/plans/2026-10-19-props-p2-diagnostics.md). Code: `experiments/xvla_props_p2_eval/diagnostics/`
(not hash-bound). No new rollouts, no GPU: every number is a replay of a recorded trial. Branch `feat/props-p2-followup`.

## Headline

- **The pre-registered failure taxonomy was not produced.** Replay itself is sound (Gate 0: 440 of 440 trials
  reproduced exactly), but the validation Gate 1 failed twice, and the spec's stop rule forbids a third definition. What
  survives is a set of envelope-free measurements, below. None of them is a pre-registered result.
- **Of the 12 report trials of 101,520 that never lifted anything, none left the gripper open: all 12 closed it, 9
  held the target without lifting it, and 9 of 12 hands rose more than 2 cm after closing.** The policy tried to lift and
  the object stayed down. 126,900 differs: 24 never-lifted trials, 7 of 24 hands rose more than 2 cm (median rise 3 mm).
  This difference was found after looking at the data and was not tested.
- **Milk is the largest single contributor: 7 of the 12 report never-lifted trials and 9 of 16 across tiers.** Milk
  failures are not approach failures: the hand gets as close to the carton in failures (0.105–0.109 m) as in successes
  (0.108 m), and the hand-frame grasp position at closure is indistinguishable (median z 0.172 m in successes, 0.173 m in
  failures). Half of the 20 milk failures lifted the carton and then failed (carried, dropped, unstable).
- **Never-lifted trials are not concentrated in particular layouts for 101,520** (permutation p = 0.45), but are for
  126,900 (p = 0.0001) and P1 (p = 0.0001).
- **The wrong-object association is not supported at layout level** (p = 0.17; nearest target-distractor distance 0.126 m
  in the 5 positive layouts against 0.152 m in the other 55). Five layouts, seven trials: exploratory and underpowered.

## Gate 0: replay reproduces the recorded trials

Each recorded `applied_action` is fed through the rollout's own physics calls, unrendered. The replay must reproduce the
recorded `state` and `props_xyz` exactly at every tick. **440 of 440** (101,520: 180 trials, 126,900: 140, P1 126,900:
120). A second pass after adding one measure reproduced every previously stored field in all 440 trials identically.

## Gate 1: failed twice; the stop rule applies

Gate 1 asks whether a *measure* of a good grasp agrees with the successes before any failure is read through it. It was
defined as the fraction of successful grasps whose hand-frame position and yaw at closure lie inside an envelope.

| definition | result | threshold |
|---|---|---|
| expert envelope (p1–p99 over the sealed expert episodes) | 0.225 (32 of 142 successes of 101,520) | 0.80 |
| success envelope of the *other* checkpoint, both directions (user decision, spec amendment) | forward 0.486, backward 0.634 | 0.80 each |

**Why the first failed.** The expert is much more precise than a success needs to be. Policy successes grasp box objects
with a finger-axis yaw error the expert never makes (median −15° bread, −7° lemon, −2° cereal; expert p1–p99 within about
±0.06–3°) and a few millimetres shallower than the expert for tall objects. "Outside the expert envelope" therefore did
not mean "a bad grasp". Per object, successes inside all four components: bottle 0.31 (n=26), bread 0.00 (27), can 0.79
(28), cereal 0.04 (24), lemon 0.04 (23), milk 0.00 (14).

**Why the second failed, and what that really shows.** The 0.80 minimum was a threshold I chose without first computing
what the sample sizes permit. I then measured it: when two *identical* policies are simulated by pooling both
checkpoints' successes and resampling to the same sizes, the same statistic is only **forward 0.605, backward 0.696**. The
threshold was unattainable by construction (FAILURES #58). Observed 0.486 / 0.634 is below that null but not by much,
except for milk:

| object | forward / backward observed | null (identical policies) |
|---|---|---|
| bottle | 0.69 / 0.59 | 0.57 / 0.67 |
| bread | 0.52 / 0.73 | 0.74 / 0.75 |
| can | 0.43 / 0.50 | 0.62 / 0.71 |
| cereal | 0.29 / 0.58 | 0.51 / 0.72 |
| lemon | 0.70 / 0.59 | 0.65 / 0.72 |
| milk | **0.14** / 0.82 | 0.47 / 0.53 |

Reading: the two checkpoints' successful grasps differ detectably for several objects, so "inside the other
checkpoint's successes" is not a clean definition of a good grasp either. I did not look for a third definition, and the
stop rule would not have allowed it: no failure was read through an envelope at any point.

## What was measured without an envelope

All from replay; `analysis_descriptive.json` (gitignored, under `outputs/xvla_props_p2_eval/diagnostics/`).

### What the gripper did on never-lifted trials

| run | never lifted | never closed | closed, never held | held, not lifted |
|---|---|---|---|---|
| 101,520 report | 12 | 0 | 3 | 9 |
| 126,900 report | 24 | 0 | 6 | 18 |
| P1 126,900 report | 110 | 9 | 80 | 21 |
| 101,520, all four tiers | 16 | 1 | 4 | 11 |

"Closed" is a run of at least 5 ticks with the gripper command below half. "Held" is both fingers in contact with the
target for at least 10 consecutive ticks after the first closure.

### Did the hand rise after closing

Peak hand height after the first closure minus hand height at that closure, `never_lifted` trials where the gripper closed:

| run | n | median | rose more than 2 cm (the evaluation's lift height) |
|---|---|---|---|
| 101,520 report | 12 | 0.084 m | 9 |
| 126,900 report | 24 | 0.003 m | 7 |
| P1 126,900 report | 101 | 0.011 m | 35 |

Caveat: the peak is over *any* time after the first closure, including after the gripper re-opened, so a rising hand does
not prove the object was carried. It separates "the policy never raised the hand" from "it did and the object did not
come with it". This table and the next were chosen after reading the pooled results; they are exploratory and untested.

### Closest approach (finger midpoint to target), per object

The first pooled figure (101,520 never-lifted median 0.106 m against successes 0.066 m) looked like an approach failure. It
is mostly object mix: milk and bottle cartons are held about 0.105–0.12 m from the finger midpoint even when they succeed.
Within an object (101,520 report; medians, n):

| object | success | never lifted | other failure |
|---|---|---|---|
| bottle | 0.121 (15) | none | 0.120 (1) |
| bread | 0.054 (23) | 0.106 (2) | 0.131 (3) |
| can | 0.065 (16) | none | none |
| cereal | 0.101 (12) | 0.123 (3) | 0.142 (1) |
| lemon | 0.055 (12) | none | 0.122 (4) |
| milk | 0.108 (12) | 0.105 (7) | 0.109 (9) |

Bread and cereal never-lifted trials closed the gripper farther from the target than their successes did (n = 2–3 each: an
observation, not a finding). Milk shows no difference. For 126,900 bread's never-lifted median is 0.168 m (n=4) against
0.054 m.

### Layout concentration of never-lifted trials

Sum of squared per-layout failure counts against 10,000 permutations, the two seeds of a layout kept together:
101,520 report 12 trials in 11 of 60 layouts, **p = 0.454**; 126,900 report 24 trials in 14 layouts, **p = 0.0001**; P1 110
trials in 56 layouts, p = 0.0001. No layout was left unsolved by both 101,520 and 126,900.

### Wrong object

Pre-registered layout-level test (60 report layouts; one test): 5 layouts have a wrong-object trial of 101,520, 55 do not;
median nearest target-distractor distance 0.126 m against 0.152 m, exact p = 0.174. The trial-level figures quoted before
(0.123 m, 0.150 m) are not the pre-registered comparison. The seven events: bottle lifted for cereal (0.136 m apart) and
for bread (0.181 m); bread lifted for lemon, twice, same layout (0.111 m); cereal lifted for milk (0.126 m); lemon lifted
for bread, twice, same layout (0.123 m). Two layouts account for four of the seven.

### Milk

34 trials, 20 layouts: `milk→orange` (trained) 7 of 14 succeeded, `milk→blue` (held out) 7 of 20. All 20 `milk→blue`
trials are report-tier, so cell and tier are one factor. Failures by cause: never lifted 9 (6 blue, 3 orange), timed out
carrying 6, dropped outside 3, unstable in the bin 1, wrong object 1. Grasp at closure, medians: successes z 0.172 m,
failures 0.173 m; |x| 0.004 m both; |y| 0.006 m against 0.008 m. Finger-axis yaw error: |yaw| median 10.9° in successes
(n=14) and 20.7° in the never-lifted failures (n=8); not tested, 8 trials. No expert comparison is claimed, because the
expert envelope was not validated.

## Video check

Seven trials rendered as strips (closure −15, closure, +30, +90, last tick; and for two trials ticks 150–800), under
`outputs/xvla_props_p2_eval/diagnostics/sheets/` (gitignored). Viewed: bread `report_050`, milk `report_042`, milk
`screen_011`, bread `report_055`. Not viewed: lemon `confirm_008`, milk `report_030`, cereal `report_018`.

- **Reading corrected.** My first reading of the closing frames was that the bread and milk were being carried off the
  table. The trace says otherwise: the target never rose more than 1.2 cm and shifted 3.8–5.7 cm along the table. The
  front camera's perspective misled me; the trace decides. Bread `report_050`: gripper closed at tick 104 and stayed
  closed for 796 ticks, hand rise 0.0 m, bread pushed 5.7 cm. Milk `report_042`: hand rose 0.07 m, carton rose 0.5 cm
  and tilted, gripper re-opened at tick 515.
- `report_055` (target bread): the gripper closes on the green bottle at tick 60 and carries it toward the orange bin; the
  bread, 0.22 m away, never moves. A clear wrong-object pick, not a near miss.
- `screen_011` (target milk): the gripper never closes for a sustained run; the hand ends hovering over a red can.
- No class definition exists to be contradicted by the video, since no taxonomy was produced.

## Checks that did not hold up

1. The expert envelope as a success region (Gate 1, 0.225).
2. The 0.80 threshold under a two-way cross-checkpoint definition (null 0.605 / 0.696).
3. The pooled closest-approach gap, which is an object-mix effect.
4. My reading of the contact sheets ("carried off the table").
5. In the analysis code itself: Gate 1 judged trials against an envelope that did not exist when an object only appeared
   as a distractor; found only because the passing-gate branch, which real data never reached, was given its own test.
   Several of my first tests were vacuous (compared a value with itself, or took the expected value from the code under
   test) and were replaced; mutation checks (break the code, confirm a test fails) became routine, and several surviving mutations exposed gaps that were then closed.

## Limitations

- Counts are small: milk 34 trials, wrong-object 7 trials in 5 layouts, per-object never-lifted cells of 2–7.
- Layout is the unit; 101,520's 12 never-lifted trials sit in 11 layouts, so concentration is low-powered.
- Tests actually run: layout concentration (3, one per run) and the wrong-object layout test (1), plus the Gate 1 null
  (a calibration). The three milk comparisons the spec allowed were not run. No multiplicity correction is claimed.
- The hand-rise table and the per-object closest-approach table were added after the pooled table was seen.
- 126,900 is a contrast, not the subject, and no 101,520-against-126,900 test was run.

## Hypotheses (not results)

| if this holds | the next step it would suggest | test needed |
|---|---|---|
| milk fails at hold, not approach (grasp and approach match successes) | a milk-specific grasp the demonstrations do not teach (yaw alignment, grip width); compare against how the expert grasps milk | a fresh-layout re-test of 101,520 that holds milk out in the other bin separates "unseen pair" from "hard object" |
| 101,520 tries to lift and the object does not come (9 of 12) while 126,900 mostly never raises the hand | training longer changed the failure from slip to no lift attempt | replay both checkpoints on the same fresh layouts |
| bread and cereal never-lifted close far from the target | approach accuracy near distractors | fresh layouts with the distractor placement varied on purpose |

## Method notes worth carrying forward

- Compute what the sample sizes permit before fixing a validation threshold: simulate two identical policies and read the
  same statistic.
- An envelope built from the scripted expert describes how precisely the expert grasps, not what is enough to succeed.
- Pool across objects only after checking that the objects share a baseline.
- Where a picture and a trace disagree, the trace decides; where a test can only fail when the code is wrong, it must be
  seen to fail (mutation).
