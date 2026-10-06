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
- **Start region** for objects: x ∈ [0.27, 0.40], y ∈ [−0.12, 0.12] **and radius √(x²+y²) ≥ 0.285 m**,
  yaw uniform in [−90°, 90°) — except the **lemon, whose yaw is capped at ±75°**.
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

## P1 — instruction sorting, one object, no distractors

**Status:** designed 2026-10-06, after P0 passed (283/288 = 98.3%). Plan:
`docs/superpowers/plans/2026-10-06-props-p1-collection.md`.

The first phase with a policy. It asks one question: **can the policy put the named object in the
named bin, when the bins swap sides and it has never seen that object–bin pair?**

### The task

- One object on the table, two bins, nothing else.
- Instruction: `put the <object> in the <colour> bin`, colour ∈ {blue, orange}.
- The bins **swap slots** between episodes, so the colour word is the only thing that says where to
  put it. A policy that learned "always the left bin" scores about 50%.

### Cells and the held-out pairs

6 objects × 2 target colours = **12 object–bin pairs**; each appears in both bin arrangements, so
**24 cells in total — of which 20 are trained** (the other 4 belong to the two held-out pairs and are
never collected).

*(Corrected 2026-10-06: the first draft said "24 training cells", conflating all 24 cells with the 20
that are actually trained. The totals below are unchanged; the per-cell counts are.)*

**Held out of training entirely: `milk → blue` and `bread → orange`.** Every object is still trained
into both colours through its other pairs, and every colour is still a target for five objects, so a
held-out pair can only be solved by combining the object word with the colour word.

### Size

- **480 training episodes** — **20 trained cells × 24 episodes**. (24 per cell is close to B3's
  proven 25 per cell; 20 per cell would give 400, which is thinner than anything that has worked here.)
- **60 validation episodes** (training-loss monitoring only) — 20 trained cells × 3, no held-out pair.
- **540 admitted episodes** in total.
- Collection at P0's measured rate (about 34 s per attempt with rendering) is roughly **6 hours**
  for 540 admitted episodes including rejections.

### Reserved evaluation layouts (never collected)

**80 layouts**, drawn in the same plan and never executed by the collector:

| tier | layouts | composition |
|---|---|---|
| screen | 20 | one per trained cell — choosing a checkpoint |
| report | 40 | **20 trained** (one per trained cell) + **20 held-out** (5 per held-out cell × 4) |
| confirm | 20 | one per trained cell — untouched, used once on the winner only (the B3 lesson) |

**80 reserved layouts.** The report tier is deliberately balanced 20 trained against 20 held-out, so
the grounding comparison is like-for-like.

### Admission of one episode

The P0 success rule (object's lowest point below the wall top, centre inside in x and y, gripper open
and clear, 30 consecutive ticks within 900), plus: frame count as recorded, the replayed actions
reproduce the trajectory (< 1e-3), and the recorded instruction matches the plan slot. A rejected
attempt retries the same slot with a fresh draw; a layout that fails three times is resampled
deterministically, as in B3.

### Read-outs, fixed in advance

- overall success, with layout-clustered intervals;
- **held-out pairs vs trained pairs** — the grounding result;
- **the shortcut check:** success split by bin arrangement. A policy using position rather than
  colour collapses on the arrangement where the named bin is in the other slot. The two arrangements
  must be within a few points of each other;
- **wrong-bin rate:** how often the object is placed, correctly and stably, in the *other* bin. This
  is the clean failure signature for a colour that was not read;
- failure causes from the classifier, written before the first rollout.

### What P1 does not test

Distractors (P2), categories (P3), the near-base corner (outside the start region), multi-object
scenes. No claim about an individual layout: seeds per layout are few.

## P1 training and evaluation

**Status:** designed 2026-10-07 from the sealed dataset's measured size. Plan:
`docs/superpowers/plans/2026-10-07-props-p1-training-eval.md`.

### The dataset, as sealed

540 episodes, **452 frames each** (not 528 as in B3), 244,080 frames, 10 task strings, fingerprint
`86e3aadb…`. Training is episodes in the 480 train slots; 60 validation episodes are for the loss
curve only. With chunk 30, the training split gives **203,040 complete chunk starts**, so at the
repo's effective batch of 8, **one epoch = 25,380 updates**.

### Training

A new package `experiments/xvla_props_p1/`, forked from `experiments/xvla_b3_robosuite/`: same
recipe (effective batch 8, chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and
cosine decay to 1e-5, FP32 weights with bf16 autocast, from the base checkpoint, no resume of a prior
run), resumable from snapshots from the start, and the same `--check` / `--preflight` / `--train`
gates in the admission image.

**Horizon: 5 epochs = 126,900 updates**, about **30 hours** at B3's measured 0.848 s per update.

**Checkpoint schedule, chosen from this repo's own evidence rather than round numbers.** Where the
best checkpoint has fallen, as a fraction of each run's horizon: B1 at 3.35 epochs (its end), B2 at
5.0 (its end), **B3 at 4.0, with 5.0 its worst by 22 points**. The peak is somewhere in 3–5 epochs
and B3 shows it can be passed, so checkpoints are dense from epoch 3:

| update | epoch | why |
|---|---|---|
| 1 | — | checkpoint machinery sanity |
| 50,760 | 2.0 | early reference |
| 76,140 | 3.0 | B1's peak lies here |
| 88,830 | 3.5 | between |
| 101,520 | 4.0 | B3's peak |
| 114,210 | 4.5 | between |
| 126,900 | 5.0 | B2's peak, and B3's collapse |

Seven checkpoints: six at ~3.3 GB plus a final ~9.9 GB with optimizer state, about 30 GB, plus two
rolling resume snapshots of ~10.5 GB. Budget about **51 GB** against the 91 GB free on 2026-10-07.

### Evaluation

A new package `experiments/xvla_props_p1_eval/`, robosuite image, GPU. Rollout cap **900 ticks**
(the expert needs 452), success exactly the P0 rule.

1. **Feasibility first, on CPU, while training runs.** The scripted expert runs all 80 reserved
   layouts. B3's reserved layouts were never expert-verified (FAILURES #14); this costs nothing
   because it uses a core the GPU is not using. A layout the expert cannot solve is excluded and
   reported before any policy number is read.
2. **Screen:** the six non-trivial checkpoints on the 20 screen layouts, one seed: 120 rollouts.
   A tie is a tie — every candidate within 2 successes of the best goes to the report.
3. **Report:** each carried candidate on the 40 report layouts (20 trained pairs, 20 held-out), two
   seeds: 80 rollouts per candidate.
4. **Confirm:** the winner only, once, on the 20 untouched confirm layouts. This is the number that
   is quoted, because the report both selected and measured (the B3 lesson).

Budget at about 1.5 minutes per rollout: screen ≈ 3 h, report ≈ 2 h per candidate, confirm ≈ 0.5 h.
**About 5–6 hours of GPU** after training, plus about 1 hour of CPU for feasibility during it.

### Read-outs (fixed before any rollout, as the P1 section already states)

Overall success with layout-clustered intervals; **held-out pairs vs trained pairs**; the
**arrangement shortcut check** (the two bin arrangements must be within a few points — a policy using
position rather than colour collapses on one of them); the **wrong-bin rate** (placed correctly and
stably in the *other* bin, the clean signature of a colour that was not read); and failure causes
from a classifier written before the first rollout.

**What a result will mean**, fixed now: held-out within 10 points of trained, wrong-bin under 5% and
the two arrangements within 5 points of each other reads as *the policy follows the colour word*. A
large arrangement gap reads as *a position shortcut* regardless of the headline. A held-out drop with
a low wrong-bin rate reads as *grounding holds, execution on unseen combinations is weaker*.

## P2 — instruction sorting with 1–2 distractor objects (roadmap level)

- Same instructions; 1–2 other objects in the start region, untouched by a correct episode.
- About 500 training episodes.
- Read-outs: held-out pairs, distractor count, wrong-object rate (first object lifted), bin errors.
- **The near-base stratum is dropped** (decision 2026-10-06). The start region now excludes radii
  below 0.285 m because the expert's straight-line approach cannot reach there, so props tests no
  position that B2 and B3 found weak. Rebuilding the approach (descend first, then move in) is a
  motion-planning project of its own, not a P2 prerequisite; the gap is recorded here deliberately.

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
