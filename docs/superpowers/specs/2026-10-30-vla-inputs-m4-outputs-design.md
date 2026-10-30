# M4 — the output side: an auxiliary box head (M4a), force-carrying outputs (M4b, scoped only) — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). M4a depends on M3's result; M4b is a scoping study and
trains nothing. No code exists.

## M4a — learn the box, do not feed it

### Question

If the policy is trained to **predict** the target's box from its own encoding (an auxiliary loss), with the state left at
P2's 9 values, does it recover some of what A3 gains from being **given** the box, without needing the box at test time?

### What is established

| fact | source |
|---|---|
| X-VLA's VLM stage (`XVLAModel.forward_vlm`) returns `vlm_features` (the Florence-2 encoder output over image and instruction tokens), which the action transformer conditions on; the training loss is `action_space.compute_loss(pred_action, action)` | `lerobot/policies/xvla/modeling_xvla.py` lines 157–229 |
| S0 records `target_box` (2 cameras × 5) for every training frame | S0 |
| the flow-matching noise is drawn from the global torch generator (`torch.randn`, line 252), so any extra random draw changes every later sample | `modeling_xvla.py` |

### Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | **Run M4a only if M3 shows the box helps:** A3 beats A0 on M0's primary test, or A3's `wrong_object` count is lower than A0's at p < 0.05 (paired, exact). Otherwise M4a is not run, and the user may override with the reason recorded | if the true box does not help as an input, a predicted one is unlikely to |
| D2 | Head: mean-pool `vlm_features` over tokens → 2-layer MLP (hidden 256, GELU) → 10 outputs; loss = L1 on the four coordinates of each camera where `visible` = 1, plus binary cross-entropy on the two `visible` flags | small, attached where the action transformer reads, no change to the action path |
| D3 | Total loss = action loss + λ · box loss, with **λ fixed before training**: on the first training batch at initialisation (a probe run before `--check`), λ = 0.1 × (action loss / box loss), then frozen and recorded. No tuning on rollouts or on offline metrics | one value chosen a priori; tuning would need a selection set this programme does not have |
| D4 | Head parameters initialised from a **separate, seeded generator** so the global torch random stream is unchanged | needed for the equivalence gate below |
| D5 | Inputs at training and test time: P2's 9-value state, sealed images, P2 instructions (as A0). The head is discarded at test time | deployable without a box |
| D6 | Recipe unchanged (126,900 updates, P2's schedule, sample order); the head's parameters use the action transformer's learning rate | one variable: the loss |
| D7 | Package `experiments/xvla_p2_auxbox/`, forked from `experiments/xvla_props_p2/`, with a local subclass of `XVLAPolicy` that adds the head and the loss term; LeRobot's installed modules are not edited | fork, do not edit |
| D8 | **Zero-equivalence gate:** with λ = 0, the new runner reproduces P2's runner byte for byte over 30 updates for the metrics, the base model weights and their Adam moments (the head's own parameters are excluded from the comparison) | the auxiliary loss is then the only difference |

### Primary result

M0's primary test of A4 against A0. **Secondary (exploratory):** A4 against A3 on C (how much of the given-box effect is
recovered), and `wrong_object`. The head's box error on validation frames is reported as a description, never used to choose a
checkpoint.

### Prediction (2026-10-30)

Smaller than A3's effect, if any. Auxiliary-loss gains are usually modest, and the head can learn the box without the action
transformer using it.

### What M4a cannot show

Whether another head placement, loss weight or target (3D position, all objects) would do better. Each is a separate run.

## M4b — outputs that carry force (scoping study only)

### Why it is not a training phase yet

The action today is 7 joint position targets plus a gripper command that the expert only ever sets to one of two values
(`gripper = 0 if name in HOLDING else 255`, `props_scene/expert.py`); the expert never varies grip effort or arm compliance.
A policy cannot learn a force output that its demonstrations do not contain. Outputs that carry force need, together: (i) a
controller that executes a force or stiffness target (robosuite ships operational-space control with variable impedance; P0
kept the joint-target IK expert because it reached its gate, and OSC only as evidence), (ii) an expert that sets those targets
meaningfully, (iii) a new dataset with its own seal. That is a dataset phase, not an input or loss change, and it is decided
after M2 says whether force information helps at all.

### What the scoping study produces (no policy training)

1. From S0's sidecar: how much grip force and wrist load vary across objects and phases in P2's demonstrations.
2. From M0's recorded signals: grip force and wrist load in A0's and A2's failing milk trials against their successes.
3. A short written options note: gripper effort as a continuous output; impedance targets for the arm; what each needs from the
   controller, the expert and the scene; the frozen files each would touch (all of them would need forks).
4. A recommendation to the user: pursue as a new phase (with its own spec), or stop.

## Plan

[../plans/2026-10-30-vla-inputs-m4-outputs.md](../plans/2026-10-30-vla-inputs-m4-outputs.md).
