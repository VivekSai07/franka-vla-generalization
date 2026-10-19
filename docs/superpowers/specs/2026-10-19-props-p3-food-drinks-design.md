# Props P3 — Food vs Drinks — design

**Date:** 2026-10-19. **Status: DRAFT, nothing in "Decisions" has been confirmed by the user.** Written at the end of a session
that finished the P2 failure diagnostics and the EGL check, to be confirmed and executed in a new session
([handoff](../handoffs/2026-10-19-props-p3-handoff.md)). The roadmap level text is in
[2026-10-05-props-sorting-design.md](2026-10-05-props-sorting-design.md) ("P3 — Food vs Drinks"); this document replaces it
with a designed phase. Follows [P2](../../PROPS_P2_EVALUATION_20261016.md) and its
[diagnostics](../../PROPS_P2_DIAGNOSTICS_20261019.md). No code or data exists yet.

## Question

With a **fixed prompt** that names nothing, can the policy put each object in the container that matches its category (drinks
in one, food in the other), when the containers swap sides, and does that carry to an object it never saw?

## What is established (so the design does not repeat it)

| fact | source |
|---|---|
| P1: the policy reads a named bin colour: **0 wrong-bin outcomes in 480 report trials**, held-out pairs not below trained | RESULTS (P1) |
| P2: with 1–2 distractors 75.0% [65.0, 84.2] report; trained 85.0%, held-out 55.0%; wrong-bin 0 of 120; wrong-object 7 of 120 | RESULTS (P2) |
| P2 per target object, all tiers, 101,520: can 28/28, bottle 26/28, cereal 24/28, lemon 23/28, bread 27/34, **milk 14/34** | computed from the P2 evaluation results |
| milk is weak **even in its trained pair** (`milk→orange` 7/14) and fails at the hold, not the approach | P2 diagnostics |
| P1's confirm winner had an arrangement split: blue-left 85% against blue-right 65% | RESULTS (P1) |
| the expert passes its gate on all six objects: P0 283/288 (98.3%), P2 re-test 284/288 (98.6%) | P0, P2 collection reports |
| 540 P2 episodes collect in 54 min with 12 parallel OSMesa shards; training ran ~0.52 s/update (projected ~18.5 h for 126,900) | P2 collection, training notes |

Consequences used below: (a) a held-out object must be one the policy handles (≥ 0.85 in P2), or a category failure cannot be
told from a grasp failure; (b) milk will drag the drinks category and must be read per object; (c) an arrangement gate would
trip on noise or on a real position asymmetry, so the arrangement split is reported, not gated.

## Decisions (proposed; to be confirmed with the user before any code)

| id | decision | reason |
|---|---|---|
| D1 | **Reuse P0's two bins** (blue, orange) as the containers; **drinks → orange, food → blue**. Different container shapes are deferred. | the scene package is frozen and hashed through three seals; new geometry means a scene fork, a second new variable. With this mapping every object–bin pair already exists in P1/P2 (`milk→orange`, `bottle→orange`, `can→orange`, `bread→blue`, `lemon→blue`, `cereal→blue`) |
| D2 | **Fixed prompt `sort the item`** in every frame of every episode. | the roadmap's wording; the prompt carries no information, so the object's identity is the only cue to the container |
| D3 | **One object per episode, no distractors.** | a fixed prompt cannot name a target among several |
| D4 | **Trained objects:** can, milk (drinks); bread, lemon (food). **Held out entirely:** bottle (drink), cereal (food). | 2 against 2 in training; each held-out object is ≥ 0.85 in P2. Bottle resembles the trained can (cylinder, easy generalisation); cereal is a tall box like the trained milk carton, so **a shape cue points to the wrong container**, which is the informative case |
| D5 | **Renderer: OSMesa, as P1/P2. EGL is not adopted.** | one new variable per phase; collection is already 54 min in parallel; the P2 checkpoints and any comparison with P1/P2 need the same pixels ([EGL check](../../EGL_RENDERING_CHECK_20261019.md)). Evaluation speed is the only real gain and does not justify mixing |
| D6 | Training: base X-VLA, P1/P2 recipe, **5 epochs** (exposure held constant, not updates). | the repo rule; the update count follows from the dataset size |
| D7 | Size: **8 trained cells × 60 = 480 train, 8 × 8 = 64 validation, 544 admitted.** Cells are object × bin arrangement (the bin follows from the object). | ≈ P1/P2's 540; 120 demonstrations per object against P1's 72–96 |

## Task

- One object from the six, two bins in P0's two slots (the bin colours swap sides between episodes, so position carries
  nothing), start region as P1/P2 (x ∈ [0.27, 0.40], y ∈ [−0.12, 0.12], radius ≥ 0.285 m, yaw ±90°, lemon ±75°).
- Success: P0's rule, unchanged — lowest point below the bin's wall top, centre inside in x and y, gripper open and clear, 30
  ticks within 900.
- The correct container is the one for the object's category: drinks (can, milk, bottle) orange; food (bread, lemon, cereal)
  blue.

## Cells and held-out objects

Trained cells: {can, milk} × orange and {bread, lemon} × blue, each in both arrangements: **8 cells**. Held-out cells (never
collected): bottle × orange and cereal × blue, both arrangements: **4 cells**.

## Dataset

480 training (60 per cell), 64 validation (8 per cell, monitoring only), 544 admitted. Schema as every dataset here
(`experiments.paired_demonstrations.media`). Seeds `sha256("props_p3_v1:<date>:<identity>")[:4]`. A fork of
`experiments/props_p2_data/` that **removes** the distractor machinery rather than carrying it: no spacing rule, no hand
clearance, no distractor gate, the P0/P1 expert unchanged. The recorded `task` is the fixed prompt: the expert's own text
(`props_scene.expert.task_text`, "put the X in the Y bin") must never reach the dataset, so it is replaced in the episode
sink and a test must fail if it ever appears.

## Reserved evaluation layouts (never collected): 128

| tier | layouts | composition |
|---|---|---|
| screen | 24 | 3 per trained cell |
| report | 80 | 40 trained (5 per trained cell) + 40 held-out (10 per held-out cell), balanced like P1 |
| confirm | 24 | 3 per trained cell, untouched until the winner |

Two seeds per layout, as P2. All 128 checked for expert feasibility (including the held-out objects) before any policy rollout.

## Gates before the full run

1. **Expert re-test:** 288 scripted attempts without rendering, all six objects × both arrangements, bars as P0/P2: ≥ 95%
   overall, every object ≥ 90%, both slots ≥ 93%. Plus rendered attempts re-run in a second process: byte-equal traces and
   images.
2. **Pilot:** 2 shards × 6 episodes through collect → merge → validate on a scratch root; watched.
3. Everything the P2 collection did (parallel shards, merge verified against the shards, seal re-verified in a second process).

## Training and evaluation (separate plans after the seal)

Fork `xvla_props_p2` and `xvla_props_p2_eval`. The evaluation classifier is written and checked on successes before the first
rollout; its causes are P2's (`never_lifted`, `timeout_carrying`, `dropped_outside`, `unstable_in_target`, `wrong_bin`) minus
`wrong_object`. Screen all checkpoints; report on those within 2 successes of the best screen; confirm once on the winner;
checkpoint selection never uses offline metrics. Parallel workers **pinned to cores**, sized by peak memory (FAILURES #57).

## Success (proposed pre-registration; gates only on trained objects)

1. Trained objects, report tier: overall success **≥ 75%** (P1 trained cells 72.5%, P2 85.0%).
2. Trained objects: **wrong-bin ≤ 5%** (an object that comes to rest stably in the other container).
3. Reported, **not gated**: success by bin arrangement (with its interval; P1 showed 85% against 65%), by object, failure
   causes, and the P2-diagnostics measures (hand rise after closure, closest approach per object).

**Held-out objects: exploratory, read as such.** Primary read-out is **bin-choice accuracy**: among trials in which the object
came to rest stably in either container, the fraction in the container of its category, per object, with a layout-clustered
interval, against the chance level 0.5. "Evidence of generalisation" is declared in advance to mean the lower bound exceeds
0.5 **for both** held-out objects. Success on held-out objects is reported next to it, never in its place.

## What P3 cannot show

- With six objects, "category" cannot be separated from visual similarity to a trained object. Bottle resembles can; cereal
  resembles milk in shape but is in the other category, which is the only discrimination this design offers. A pass is
  evidence about the object-to-container mapping on these objects, not about semantic food versus drinks.
- Milk's weak grasp lowers the drinks category's success and must not be read as a category failure; per-object tables are
  required, and bin choice is conditioned on a stable placement for this reason.
- No distractors, no multiple objects, no new container shapes.

## Out of scope

EGL, new objects, new container geometry, a milk-specific grasp (a second variable; it would have to be its own phase),
multi-object sequential sorting, the near-base corner.

## Plans

[Collection plan](../plans/2026-10-19-props-p3-collection.md). Training and evaluation plans follow the seal.
