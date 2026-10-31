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
| X-VLA's VLM stage (`XVLAModel.forward_vlm`) returns `vlm_features` (the Florence-2 output over the first image and instruction tokens), plus `aux_visual_inputs` for other views, which the action transformer conditions on; the training loss is `action_space.compute_loss(pred_action, action)` | `lerobot/policies/xvla/modeling_xvla.py` lines 157–229 |
| S0 records `target_box` (2 cameras × 5) for every training frame | S0 |
| the flow-matching noise is drawn from the global torch generator (`torch.randn`, line 252), so any extra random draw changes every later sample | `modeling_xvla.py` |

### Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | **Run M4a only if M3 shows the box helps:** A3 beats A0 on M0's primary test, or A3's `wrong_object` count is lower than A0's at p < 0.05 (paired, exact). Otherwise M4a is not run, and the user may override with the reason recorded | budget heuristic only; input conditioning and auxiliary supervision are distinct hypotheses, and a null M3 does not rule out M4a |
| D2 | Head: concatenate separately mean-pooled `vlm_features` (front view + language) and `aux_visual_inputs` (wrist view) in fixed order → 2-layer MLP (hidden 256, GELU) → 8 outputs; loss = L1 on the four coordinates of each camera. **No `in_frame` term:** S0 found it 1 on every training frame, so a classification target would be constant | small, attached where the action transformer reads, no change to the action path |
| D3 | Total loss = action loss + λ · box loss, with **λ fixed before training**: on the first training batch at initialisation (a probe run before `--check`), λ = 0.1 × (action loss / box loss), then frozen and recorded. No tuning on rollouts or on offline metrics | one value chosen a priori; tuning would need a selection set this programme does not have |
| D4 | Head parameters initialised from a **separate, seeded generator** so the global torch random stream is unchanged | needed for the equivalence gate below |
| D5 | Inputs at training and test time: P2's 9-value state, sealed images, P2 instructions (as A0). The head is discarded at test time | deployable without a box |
| D6 | Recipe unchanged (126,900 updates, P2's schedule, sample order); the head's parameters use the action transformer's learning rate | one variable: the loss |
| D7 | Package `experiments/xvla_p2_auxbox/`, forked from `experiments/xvla_props_p2/`, with a local subclass of `XVLAPolicy` that adds the head and the loss term; LeRobot's installed modules are not edited | fork, do not edit |
| D8 | **Zero-equivalence gate:** with λ = 0, the new runner reproduces P2's runner byte for byte over 30 updates for the metrics, the base model weights and their Adam moments (the head's own parameters are excluded from the comparison) | the auxiliary loss is then the only difference |

### Review requirements before training

[Audit R2](../../VLA_INPUTS_AUDIT_20261030.md): the inspected X-VLA source sends only the first view through
`vlm_features`; the second view is separate. **Confirmed on the installed source (2026-10-30, `modeling_xvla.py` sha256
`d8fc952243f764a33157be2518fddd38ebc071d53fa66a5f9dea8d775f787dbf`, lines 157–192):** every view goes through the Florence-2
image encoder, but only view 0 (`image_features[:, 0]`) is merged with the instruction tokens and run through the language
encoder (`vlm_features`); views 1+ are returned as image-encoder features only (`aux_visual_inputs`). Keep both features in the head, with
camera order and token masks explicit. Assert each view can affect the output, auxiliary gradients reach shared trainable
visual parameters, and all head parameters are included exactly once in the optimizer with the intended LR. Preserve the
base model's random stream and avoid a second VLM forward that changes dropout/noise draws.

Define loss reductions: per-camera coordinate L1 mean over its four coordinates and rows; mean the two camera terms
(S0: `in_frame` = 1 on every training frame, so no mask and no BCE term). Compute reductions in FP32.
For D3 record batch identity, RNG seed, loss reductions, finite/nonzero denominator and shared-parameter gradient norms;
matching scalar loss magnitudes does not guarantee a 10% gradient contribution. No outcome-driven lambda adjustment.
Verify save/load through the ordinary inference adapter after dropping only the declared head keys, never broad
`strict=False` acceptance of missing base weights.

A4 tests representation regularization; it does not feed predicted boxes to A3. D1's original start condition remains a
budget heuristic, not a scientific necessity. Its secondary unadjusted p-value is exploratory and is not programme-wide
proof. Recipes remain frozen before C; redesign informed by C requires fresh confirmation. Follow M0's secondary
matched-phase comparison and single-seed limits.

### Primary result

M0's primary test of A4 against A0. **Secondary (exploratory):** A4 against A3 on C (how much of the given-box effect is
recovered), and `wrong_object`. The head's box error on validation frames is reported as a description, never used to choose a
checkpoint.

### Prediction (2026-10-30)

Hypothesis: smaller than A3's effect, if any. No general bound is justified here; the head can learn the box without
the action transformer making useful use of the shared features.

### What M4a cannot show

Whether another head placement, loss weight or target (3D position, all objects) would do better. Each is a separate run.

## M4b — outputs that carry force (scoping study only)

### Why it is not a training phase yet

The action today is 7 joint position targets plus a gripper command that the expert only ever sets to one of two values
(`gripper = 0 if name in HOLDING else 255`, `props_scene/expert.py`); the expert never varies grip effort or arm compliance.
A policy cannot learn a force output that its demonstrations do not contain. Outputs that carry force need, together: (i) a
controller that executes a force or stiffness target (robosuite ships operational-space control with variable impedance; P0
kept the joint-target IK expert because it reached its gate, and OSC only as evidence), (ii) an expert that sets those targets
meaningfully, (iii) a new dataset with its own seal. That is a dataset phase, not an input or loss change, and it is scoped after M2, whose instantaneous-input result cannot rule out other force representations or controllers.

### What the scoping study produces (no policy training)

1. From S0's sidecar: how much grip force and wrist load vary across objects and phases in P2's demonstrations.
2. From M0's recorded signals: grip force and wrist load in A0's and A2's failing milk trials against their successes.
3. A short written options note: gripper effort as a continuous output; impedance targets for the arm; what each needs from the
   controller, the expert and the scene; the frozen files each would touch (all of them would need forks).
4. A recommendation to the user: pursue as a new phase (with its own spec), or stop.

## Plan

[../plans/2026-10-30-vla-inputs-m4-outputs.md](../plans/2026-10-30-vla-inputs-m4-outputs.md).
