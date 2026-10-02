# Sim2sim: the MuJoCo-trained B2 policy on the robosuite scene — 2026-10-02

**Headline: 3/20 on the robosuite scene against 19/20 on the MuJoCo scene, same policy, same
starts, same seeds, same loop.** The policy trained on the MuJoCo scene does not transfer
zero-shot to the robosuite scene. It usually still picks the cube up (14 of 20 lifts) and then
does not take it to the target.

Not a sealed result: `experiments/robosuite_sim2sim/` has no contract and is not hash-bound. Ten
layouts and two seeds each; read the numbers as counts, not as rates.

## What was run

| | |
|---|---|
| policy | X-VLA, `xvla_franka_stacking_b2_distractors_v1_20260928`, step 50,000 (B2's reported 91.9%) |
| trials | B2's own report trials, **control condition** (no distractors), first 10 report layouts × 2 seeds = 20, with B2's rollout seeds |
| starts | re-derived and proven against B2's sealed bank in the admission image (MuJoCo 3.13.0), exported as qpos / qvel / ctrl |
| arm "mujoco" | the scene the policy was trained on (`smolvla_scene.xml`) |
| arm "robosuite" | the robosuite bring-up scene, same arm pose and cube poses, settled 0.2 s on its table |
| shared by both arms | the rollout loop (mirrors B2's `_rollout`), the processors, the 1,200-tick cap, the 30-tick stable-stack success rule, and the image: robosuite image, **MuJoCo 3.4.0**, one GPU |
| output | `outputs/robosuite_sim2sim/b2_step050000_20261002/` (`result.json`, per-trial `episode.json`, `trace.npz`, `cameras.mp4`) |

The only thing that differs between the arms is the scene. The distractor condition was not run:
the robosuite scene has only the red and green cubes.

## Result

| arm | successes |
|---|---|
| mujoco (control) | **19 / 20** |
| robosuite | **3 / 20** |

Paired by trial: both succeed 3, MuJoCo only 16, robosuite only 0, neither 1. Exact McNemar on the
16 discordant pairs: p = 3 × 10⁻⁵. The three robosuite successes come from two layouts
(`evaluation_026` 1 of 2, `evaluation_028` 2 of 2); the other eight layouts are 0 of 2.

**The control validates the harness.** B2's own report scored 19/20 on these same 20 trials in the
admission image. The MuJoCo arm here also scored 19/20, agreeing trial-for-trial on 18: one trial
flipped each way (`t017` failed here and passed in B2; `t038` passed here and failed in B2). On
the first trial the success came at tick 365 here and 362 in B2. So neither the re-implemented
loop nor MuJoCo 3.4.0 explains the robosuite drop.

## What the 17 robosuite failures are

Classified by first cause from the traces (`experiments/robosuite_sim2sim/analysis/classify.py`):
lifted means the pick cube rose more than 3 cm; released means the gripper reopened after the lift.

| class | robosuite | mujoco |
|---|---|---|
| success | 3 | 19 |
| lifted the cube, never took it to the target, never released | **11** | 1 |
| never lifted the cube | **6** | 0 |
| released off target | 0 | 0 |

- **Carry is what breaks most.** In the 11 carry failures the cube is grasped and lifted 12–19 cm
  at about the same time as on MuJoCo (lift at tick 145–186, against 135–167), then the arm drifts
  10–26 cm, in 9 of the 11 *away* from the place cube, and holds the cube a few centimetres above
  the table until the cap. The closest any of them got to the target was 51 cm. Not one released.
- **Six never lifted.** The gripper closes (in three of them it opens and closes 10–13 times) and
  the cube stays on the table.
- The same classifier on the MuJoCo arm gives 19 clean pick-carry-release sequences, so it is not
  labelling normal behaviour as failure. The one MuJoCo failure is also a carry failure.
- By pick colour and arrangement, robosuite: red 1/12, green 2/8; familiar 0/8, swapped 3/12. With
  ten layouts these splits are not evidence of anything.

## What this does and does not show

Shown: on these 20 trials the scene change alone removes almost all of the policy's success, and
the loss is mostly after the grasp.

Not shown: *which* difference between the scenes is responsible. They differ in several ways at
once, none isolated here:

- the wrist camera sees a featureless white table instead of a blue checker floor, and shorter
  fingers at the bottom of the frame;
- the front camera sees a table, wood floor, walls and a mount ring;
- the robot meshes are robosuite's (greyer, different shading); lighting differs;
- contact and finger-pad properties are robosuite's, which could matter for the six non-lifts.

Hypothesis, not a finding: the carry depends on visual cues the white table does not give. The
cheapest test changes one thing — give the robosuite table the MuJoCo floor's blue checker texture
and rerun these 20 trials.

Limits: one checkpoint, ten layouts, two seeds per layout (the repo's rule is at least eight
seeds for any per-layout claim, so no per-layout claims are made); control condition only; the
robosuite scene's fingertip-pad contract fix is what the grasp detection rests on.

## What it means for the next project

A policy trained on the MuJoCo cube scene cannot be reused on a robosuite scene as it is; the
props project needs its own training data collected in robosuite. It also weakens one argument for
keeping the joint-target action space — "old checkpoints can still be rolled out" — because they
roll out but mostly fail. That argument should carry little weight in the controller decision
(TODO.md).

## Reproduce

```bash
# stage A, admission image, CPU: export the sealed starts
python -m experiments.robosuite_sim2sim.starts --out /out/<run>
# stage B, robosuite image, GPU: both arms
python -m experiments.robosuite_sim2sim.rollout --run /out/<run>
# classification
python experiments/robosuite_sim2sim/analysis/classify.py /out/<run>
```

Container flags and mounts: `experiments/robosuite_sim2sim/README.md`.
