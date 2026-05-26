# Dataset Design: franka_libero_stacking_v2

## What went wrong with v1 — and why

After fine-tuning SmolVLA on `franka_libero_stacking_v1` and running inference,
the robot showed qualitatively correct behaviour — it approached the right area
of the table, attempted a grasp, and tried to place — but achieved **0% task
success**. The motion was better than OpenVLA but never completed the stack.

Watching the inference videos, the failure mode was consistent:
> The robot reaches **around** the specified cube, not exactly on it.
> It places **around** the target, not on top of it.

This document explains what caused each failure and what v2 changes to fix it.

---

## Failure 1 — Single fixed instruction (root cause)

### What we did
Every one of the 50 episodes used the exact same language instruction:

```python
INSTRUCTION = "pick up the red cube and place it perfectly on top of the blue cube"
```

### Why it failed
SmolVLA is a Vision-Language-Action model. Its VLM backbone is supposed to
**ground language tokens to visual objects** — the word "red" should activate
attention on the red cube in the image, and "blue" on the blue cube.

With only one instruction across all 50 episodes, the model never needed to
distinguish "red" from "green" or "blue" from "yellow". The language token
"red" was always correlated with the same pick action regardless of which
cube was actually red in that frame. The model learned a visuomotor reflex
(go to the cube in the pick-zone) rather than genuine language grounding.

At inference, when cubes were randomised into new positions, the model had no
precise colour-to-location binding. It went to the approximate region where
the red cube tended to appear during training, not where it actually was.

### What v2 does
Generate all 12 valid pick-place colour combinations and sample one randomly
per episode:

```
red→green, red→blue, red→yellow
green→red, green→blue, green→yellow
blue→red, blue→green, blue→yellow
yellow→red, yellow→green, yellow→blue
```

Each colour now appears as both a pick target and a place target across the
dataset. The model is forced to learn what each colour word means visually,
not just where cubes tend to be.

The success verification also switches from hardcoded red/blue checking to
verifying the instructed pair.

---

## Failure 2 — Too few episodes for the task complexity

### What we did
50 total episodes, all of the same single instruction.

### Why it failed
50 episodes is very sparse even for a single-instruction fine-tuning task.
For a 12-instruction task it is completely insufficient — each colour
combination would appear only ~4 times on average, which is nowhere near
enough for the model to generalise.

### What v2 does
Target **300 episodes** (~25 per colour combination). This gives enough
demonstrations for the model to see each pick-place pair in diverse cube
arrangements, building a robust policy per instruction variant.

---

## Failure 3 — Cubes too clustered, insufficient spatial diversity

### What we did
- Spawn range: `x=[0.25, 0.55]`, `y=[-0.35, 0.35]`
- Minimum inter-cube clearance: 12 cm

### Why it failed
A 12 cm clearance on a ~30 cm × 70 cm table means cubes often spawn close
together. The gripper camera could not reliably distinguish the target from an
adjacent distractor. The model learned a spatial prior ("the pick cube is
usually near the centre-left") rather than a colour-grounded search strategy.

### What v2 does
- Widen spawn range to `x=[0.20, 0.60]`, `y=[-0.40, 0.40]`
- Increase minimum clearance to 15 cm

This forces cubes to be clearly separated in the camera frame, making
colour-based object selection a necessary skill rather than a spatial shortcut.

---

## Failure 4 — Always 4 cubes present (fixed distractor count)

### What we did
All 4 cubes (red, green, blue, yellow) were present in every episode.

### Why it failed
The model always saw the same 4-object scene. It could not learn to handle
scenes with fewer objects, and more importantly it was never tested on its
ability to identify objects without the full colour palette present. The fixed
visual context became a cue itself.

### What v2 does
Randomly vary the number of cubes per scene:
- 2 cubes: pick cube + place cube only (no distractors)
- 3 cubes: pick + place + one distractor
- 4 cubes: pick + place + two distractors

The model must identify objects by colour, not by the number or arrangement
of objects in the scene.

---

## Failure 5 — Deterministic approach trajectories

### What we did
The IK controller always followed the same cubic trajectory to exactly the
same offsets above each cube. Every episode's approach path was identical
modulo the cube's spawn position.

### Why it failed
The model overfit to a single canonical approach style. At inference, if the
gripper approached from a slightly different angle (due to a different arm
starting posture or small physics differences), the policy could not adapt.
This contributed to the "reaches around but not onto" effect — the policy
expected to arrive at the cube from one fixed direction and the gripper
ended up a few centimetres off.

### What v2 does
Add Gaussian noise to approach offsets:

```python
approach_noise = np.random.normal(0, 0.005, 3)  # ±5 mm in x, y, z
```

Applied only during the approach and descent phases, not during the grasp
or placement itself. This exposes the model to natural variation in how the
gripper arrives above the target, making the learned policy more robust.

---

## Failure 6 — Success threshold too lenient for a precision task

### What we did
Accept an episode as successful if `xy_dist < 0.03 m` (3 cm XY alignment
between stacked cubes).

### Why it failed
3 cm is loose. A cube is roughly 4–5 cm on each side, so a 3 cm XY offset
means the top cube is nearly hanging off the edge of the bottom cube. The
model learned from demonstrations that were, by the training data's own
definition, imprecise. The ceiling for inference quality is set by
demonstration quality.

### What v2 does
Tighten the success threshold to `xy_dist < 0.02 m` (2 cm). Only tight,
centred stacks enter the dataset. The model learns from genuinely successful
demonstrations and has a correspondingly higher ceiling at inference.

---

## Summary table

| Dimension | v1 (broken) | v2 (fixed) | Why it matters |
|---|---|---|---|
| Instructions | 1 (red→blue always) | 12 (all colour pairs) | Forces genuine language grounding |
| Episodes | 50 | 300 | Enough coverage per instruction variant |
| Success threshold | xy < 3 cm | xy < 2 cm | Demonstration quality sets inference ceiling |
| Cube count per scene | Always 4 | Random 2, 3, or 4 | No spatial/count shortcuts |
| Cube min clearance | 12 cm | 15 cm | Colour must drive selection, not position |
| Spawn range | x=[0.25,0.55] y=[-0.35,0.35] | x=[0.20,0.60] y=[-0.40,0.40] | More spatial diversity |
| Approach trajectory | Deterministic | ±5 mm Gaussian jitter | Robustness to approach variation |
| Success verification | Always checks red/blue | Checks instructed pair | Data integrity |
