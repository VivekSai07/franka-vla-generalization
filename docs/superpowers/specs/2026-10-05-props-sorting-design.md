# Props sorting in robosuite — design

**Date:** 2026-10-05
**Status:** roadmap and P0 agreed with the user in conversation (2026-10-05); P1–P3 are designed here
at roadmap level and each gets its own detailed section or spec revision after the previous phase
reports.
**Follows:** B3, the last cube experiment ([B3_XVLA_EVALUATION_20261005.md](../../B3_XVLA_EVALUATION_20261005.md)).

## What the user wants

Move from stacking cubes to sorting real-life objects into bins, in the robosuite scene, and learn
how far an X-VLA policy can go: first following explicit instructions, then sorting by category
(Food vs Drinks). Progress in phases, each one a working result before the next starts.

## Decisions (user, 2026-10-05)

| question | decision |
|---|---|
| order | P0 scene + expert → P1 instruction, one object → P2 instruction + distractors → P3 Food vs Drinks |
| bins in P1/P2 | two **visually distinct** bins that swap sides between episodes; the instruction names the bin by its look, so the policy must recognise which bin is which, not where it usually is |
| controller | the repo's joint-target IK expert; robosuite's OSC only if ours cannot pass the P0 gate |
| P3 containers | two visually different containers (one for food, one for drinks) |

## Carried over from B3 (apply in every phase)

- The +y side near the robot base was the weak corner in B2 and B3: keep a layout stratum there and
  check expert and policy for left/right asymmetry.
- Log every object's full pose (position and orientation) in every trace.
- Write a detailed failure-mode classifier before the first policy rollout of a phase, validated on
  successes first.
- Evaluate several checkpoints spread through training; keep a set of layouts that no selection
  touches and confirm the winner on it.
- Training resumable from the start; `--ulimit nofile=65536:65536` and a large `/tmp` (16–32 GB).
- One new variable per phase where possible; when not, the evaluation must separate them.

---

## P0 — two-bin scene and expert bring-up (no policy)

### Scene

New package `experiments/props_scene/`, built like `experiments/robosuite_scene/`: robosuite 1.5.2
composes, plain MuJoCo drives, `env.step` never called, robosuite image (MuJoCo 3.4.0).
`experiments/robosuite_scene/*.py` is sealed by B3 and is only imported, never edited.

- Robot, mount, table (top at z = 0), cameras `front_cam` and `gripper_cam` at 512 × 512: as the
  robosuite scene. The robot part of its model contract (actuators, finger joints, joint dynamics,
  pad re-parenting, render groups) is reproduced in the new package and tested against the current
  MuJoCo scene the same way.
- **Two bins**, static, open boxes: inner 0.22 × 0.22 m, walls 0.07 m high and 0.01 m thick, on the
  table. **Bin slots** centred at (0.32, −0.33) and (0.32, +0.33). Each bin has a fixed look:
  `blue` (rgba 0.15 0.35 0.85) and `orange` (rgba 0.95 0.55 0.10). Either bin can occupy either slot.
- **Objects:** robosuite's `CanObject`, `BreadObject`, `LemonObject`, `MilkObject`, `BottleObject`,
  `CerealObject`, with masses set explicitly: can 0.15, bread 0.05, lemon 0.08, milk 0.20,
  bottle 0.20, cereal 0.15 kg. Only the objects in an episode exist in its model (one cached model
  per object set and bin arrangement).
- **Start region** for objects: x ∈ [0.27, 0.40], y ∈ [−0.12, 0.12], yaw uniform in [−90°, 90°).
  *(Amended 2026-10-05 after the first P0 run: the original lower bound of 0.22 m put starts inside the
  arm's working envelope — below about 0.285 m radius joint 4 saturates on the straight-line approach
  and the hand closes on air, 59 of 300 attempts. B3 used radius ≥ 0.28 m. There is no near-base
  stratum in props as a result; the B2/B3 weak corner is not tested here.)*

### Expert

A new scripted expert (not a fork of the cube expert, whose phase table assumes a cube). Same
building blocks: pinocchio IK, bounded joint targets, `GraspController`, `step_control` at 30 Hz
with 20 substeps. Per object, from its model geometry:

- **Grasp yaw:** fingers close across the object's narrowest horizontal extent (all six are ≤ 6 cm
  across it; the gripper opens 8 cm). The hand yaw follows the object's current yaw.
- **Grasp height:** the grasp point sits `min(0.03, height/2)` below the object's top; the hand
  target is that point plus the cube expert's 0.105 m hand offset.
- **Carry:** lift until the object's bottom clears the bin walls by 0.05 m, move over the target bin's
  centre, lower until the object's bottom is 0.02 m above the wall top, open, retract, go home.
- Jitter as in B3: home pose ±1°, target ±5 mm (seeded).

### Success

The object's **lowest point** is below the target bin's wall top, its centre is inside the bin's inner
box in x and y, and the gripper is open and not touching it, for 30 consecutive ticks, within 900
ticks.

*(Amended 2026-10-05 after the first P0 run. The original rule required the object's **centre** below
the wall top, which a 14–16 cm object standing in a 7 cm bin can never satisfy: milk's centre sits at
7.2 cm, cereal's at 7.5 cm, bottle's at 8.0 cm. Measured consequence: 77 of 136 failures were objects
resting correctly in the bin, and all 35 tall-object "successes" were objects that had fallen over
(tilt 52–57°) — the rule rewarded toppling. The amended rule is what `metrics.placement` already
computes; re-scoring the existing 300 attempts under it gives 241/300, and 205/206 once the start
region is also fixed.)*

### P0 gate (go / no-go)

- **Expert success ≥ 95%** over 300 scripted attempts: 6 objects × 2 target bins × 2 bin arrangements
  × ~12 seeds, random start pose and yaw. *(Re-scoring the first run under the two amendments above
  predicts 99.5%, every object ≥ 97%, both slots ≥ 99% — so the re-run is expected to pass, and a
  result far from that means something else changed.)*
- Every object reaches ≥ 90% on its own (no object carried by the others).
- Both bin slots ≥ 93% (the left/right asymmetry check).
- Replay of recorded actions reproduces states (< 1e-3); two processes give byte-equal traces.
- Every trace logs every object's pose each tick.
- Frames from both cameras show both bins and the object; checked by eye on a contact sheet.
- If the gate fails: tune within a bounded list (grasp depth, lift clearance, release height, durations,
  per-object friction), record every attempt; if still red, try OSC on the failing objects only and
  report. Never loosen the success definition.

---

## P1 — instruction sorting, one object, no distractors (roadmap level)

- Instruction: `put the <object> in the <colour> bin`, colour ∈ {blue, orange}.
- Cells: 6 objects × 2 target bins × 2 arrangements (which slot the blue bin is in) = 24; **2
  object × bin pairs held out** of training (chosen in P1's design, each object still trained into
  both bins overall).
- About 300 training episodes (12–13 per cell), validation episodes, reserved screen / report /
  confirmation layouts.
- **Shortcut check:** success on the arrangement the instruction's bin is in the "other" slot must not
  collapse; a policy that ignores the colour and uses position scores ~50%.

## P2 — instruction sorting with 1–2 distractor objects (roadmap level)

- Same instructions; 1–2 other objects in the start region, untouched by a correct episode.
- About 500 training episodes; a stratum of start positions toward the +y near-base side.
- Read-outs: held-out pairs, distractor count, wrong-object rate (first object lifted), bin errors.

## P3 — Food vs Drinks (roadmap level)

- Two visually different containers (one food, one drinks). Drinks: milk, bottle, can. Food: bread,
  lemon, cereal.
- First one object per episode with a fixed prompt (e.g. `sort the item`); the container sides swap.
- Stretch test, read as exploratory: an object held out of training entirely.
- Multi-object sequential sorting only after the single-object version works.

## Out of scope

OSC unless P0 requires it; hardware; SmolVLA; objects other than the six above until P3 asks for more.

## Plans

One plan per phase, written after the previous phase reports:
`docs/superpowers/plans/2026-10-05-props-p0-scene-expert.md` first.
