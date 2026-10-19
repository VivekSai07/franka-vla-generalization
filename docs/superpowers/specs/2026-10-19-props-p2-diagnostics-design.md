# Props P2 failure diagnostics — design

**Date:** 2026-10-19
**Status:** design approved by the user in conversation (2026-10-19). Follows
[PROPS_P2_EVALUATION_20261016.md](../../PROPS_P2_EVALUATION_20261016.md). No new rollouts, no GPU, no new data.
**Method guide:** [diagnosing-policy-failures](../../../.github/skills/diagnosing-policy-failures/SKILL.md).

## Questions

The P2 evaluation met two of four criteria and left three things unexplained. All three are asked of checkpoint
101,520 (the pre-registered winner), with 126,900 and the P1 checkpoint as contrasts.

1. **Never lifted.** 12 of the 120 report trials never lift anything (16 of 180 over all four tiers). What did the
   policy do instead: never close the gripper, close it away from the object, close it on a distractor, or close it
   correctly and fail to hold?
2. **Wrong object.** 7 report trials lift a distractor first. Their nearest distractor is closer to the target
   (median 0.123 m) than in successes (0.150 m). Is that a real association, and which distractors get lifted?
3. **Milk.** `milk→blue` (held out) scored 7/20 and trained `milk→orange` 5/8. For each failed milk trial, did the
   approach or the grasp fail?

## Method

**Replay, not new rollouts.** A recorded trial is reproduced by restoring its reserved layout (`scenes.restore`) and
feeding the recorded `applied_action` tick by tick through the same physics calls the rollout uses
(`synchronize_observation`, `apply_action`, `step_control`), unrendered. The stored traces hold only joint state and
prop positions; replay recovers what they lack: hand pose, finger positions, and every contact (finger–target,
finger–distractor, distractor–anything-but-table) at every tick.

**Gate 0, positive control.** A replay must reproduce the trial's recorded `state` and `props_xyz` arrays exactly (all
ticks) on every one of the 180 trials of 101,520. If any trial does not, replay is not used and the cause is found
first; no analysis proceeds on a method that does not reproduce its own data. (The bundle smoke re-run already
reproduced two trials bit for bit; this checks all of them.)

**Expert reference.** The 540 sealed expert episodes (`props_p2_data/episodes/*.npz`: `qpos`, `props_pose`) give the
reference distribution of every grasp quantity below, by forward kinematics only (set `qpos`, `mj_forward`; no
stepping). Quantities are defined once, from the expert, **before** any policy failure is looked at.

**Every measure is validated on the successes first.** A measure that does not separate the policy's own successes
from its failures, and does not agree with the expert on successes, is noise and is dropped.

## Definitions (fixed before any failure is examined)

- **Hand frame.** The robot hand body `robot0_right_hand`, axes as MuJoCo reports them. Target pose is taken in this
  frame. Finger axis yaw is the yaw of the vector between `left_finger` and `right_finger`.
- **First closure.** The first tick whose gripper command (`applied_action[7]`) is 0. 255 is open
  (`props_scene.constants`, `GRIPPER_OPEN_CTRL`).
- **Expert envelope, per object.** At the expert's first closure, the target centre in the hand frame, per axis, from
  the 1st to the 99th percentile over that object's expert episodes; plus the finger-axis-to-object yaw error modulo
  90 degrees, same percentiles.
- **Lifted.** As in the evaluation: the object rises 2 cm above its start and stays up 15 ticks
  (`analysis.lift_tick`). **Held.** Both fingers in contact with the object on at least 10 consecutive ticks after the
  first closure.

## Taxonomy of a failed trial (first cause, mutually exclusive, in this order)

| class | condition |
|---|---|
| `never_closed` | the gripper command is never 0 |
| `closed_on_distractor` | at first closure a distractor is inside its own envelope and the target is not |
| `closed_off_target` | at first closure the target is outside its envelope on at least one axis (approach error) |
| `closed_not_held` | the target is inside its envelope at first closure and is never held (grasp execution) |
| `held_not_lifted` | held, but never lifted (lift or carry failure) |
| `lifted_then_failed` | lifted; the failure is then the evaluation's own cause (`timeout_carrying`, `dropped_outside`, `unstable_in_target`, `wrong_bin`) |
| `wrong_object` | the evaluation's own label: first lifted object is not the target; kept as is |

`never_lifted` trials fall into the first five rows. For every `never_lifted` trial the hand's closest approach
(minimum fingertip-midpoint-to-target-centre distance) is also reported, to separate hovering from wandering.

## Analyses

1. **Never lifted (question 1).** The taxonomy table for all 16 never-lifted trials of 101,520 (report tier reported
   separately: 12), and for 126,900 (24 report trials) as a contrast. Layout concentration: the sum of squared
   per-layout failure counts against 10,000 permutations of the failure labels over trials with the two seeds of a
   layout kept together (the B2 diagnosis procedure). Which layouts fail at both checkpoints.
2. **Wrong object (question 2).** Layout is the unit (60 report layouts): a layout is positive if any of its trials is
   `wrong_object`. One test: Mann–Whitney on the layout's minimum target–distractor distance, positive against
   negative. Descriptive: which distractor was lifted (object and its distance to the target and to the hand's
   starting pose), 126,900 and P1 alongside. Seven events: **exploratory**.
3. **Milk (question 3).** All 34 milk trials of 101,520 (20 failures, 14 successes) through the taxonomy, split by cell,
   with the hand-frame target position at first closure and the yaw error against the milk expert envelope. Expert
   milk episodes are the comparison. The 20 `milk→blue` trials are all report-tier, so cell and tier are one factor.
4. **Video check.** Before the taxonomy is written up, the cases that land in each class are checked on
   `cameras.mp4`; at least the two oddest per class. Wrong labels are fixed in the definition, not in the table.

## Counting tests

At most six tests, stated here: the layout-concentration permutation (1), the layout-level Mann–Whitney (1), the
milk comparisons (at most 3: held-out against trained failure class, grasp height against outcome, yaw error against
outcome), and Gate 0 (a check, not a test). No correction is claimed; anything not backed by a physical mechanism or a
replay measurement is labelled exploratory in the report.

## What would change the next step

- `closed_off_target` dominant: approach accuracy limits the policy, so more or more varied approach data
  (positions near distractors) is the lever.
- `never_closed` dominant: the policy stalls before grasping; an observation or action-chunking question, not a data one.
- `closed_not_held` on milk with the expert's envelope respected: grasp height or yaw, i.e. a milk-specific grasp
  strategy the demonstrations did not teach.
- Wrong-object concentrated on small target–distractor distances: spacing in the data distribution
  (`props_p2_data` spacing rule) is the lever.

## Outputs

- `docs/PROPS_P2_DIAGNOSTICS_<date>.md`: headline, taxonomy tables, what was ruled out, checks that did not hold up,
  limitations, directions labelled as hypotheses.
- Scripts under `experiments/xvla_props_p2_eval/diagnostics/` (not hash-bound: the evaluation package has no contract).
- Live docs updated in the same pass (`RESULTS.md`, `BENCHMARK_STATUS.md`, `TODO.md`, `FAILURES.md` if a mistake
  recurs).

## Out of scope

New rollouts, any GPU use, new layouts (that is the fresh-layout re-test, its own spec), any change to the sealed P2
dataset or to hash-bound files, and any claim about why the model's weights behave as they do.
