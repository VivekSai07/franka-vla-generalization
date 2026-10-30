# M3 — the target's bounding box in the robot state — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). Depends on S0, M0, and M2's package (forked after M2's
run has finished, so M2's bound files are never edited). No code exists.

## Question

With everything else identical to P2's X-VLA run, does giving the policy the target object's 2D box in both cameras change
success on M0's comparison tier, and does it remove wrong-object choices?

## What is established

| fact | source |
|---|---|
| A0 wrong-object 7 of 120 report trials (5.8%) in 5 layouts; whether those layouts have a closer distractor was not supported at layout level (p = 0.17) | PROPS_P2_EVALUATION_20261016, PROPS_P2_DIAGNOSTICS_20261019 |
| most A0 failures are not about which object: 12 of 30 never lifted, milk at the hold | PROPS_P2_DIAGNOSTICS_20261019 |
| X-VLA has 20 proprioception slots | `check_model_inputs` |

## Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | State = 9 P2 values + `target_box` = front camera `(u_min, v_min, u_max, v_max, in_frame)` then gripper camera, **19** values, from the S0 sidecar for training and S0 `signals.py` at rollout | approved: target only, both cameras, normalised to [0, 1] |
| D2 | No clipping (bounded by construction); the unchanged mean/std normaliser over 19 values, training-row statistics | — |
| D3 | Recipe unchanged, as M2 D3 | one variable |
| D4 | Package `experiments/xvla_p2_box/`, forked from `experiments/xvla_p2_force/` **after A2's evaluation**; only the signal selection, `OBS` (19) and the local input check change | M2's files stay bound to M2's run |
| D5 | Zero-equivalence gate and train/rollout parity gate exactly as M2 D5–D6 | same proofs |

## Review requirements

[Audit R7](../../VLA_INPUTS_AUDIT_20261030.md): `in_frame` is geometric projected-box validity, not visible pixels;
boxes are amodal and ignore occlusion. S0 must test near-plane clipping, geom transforms and tightness as well as containment.
No claim about detector robustness follows from an oracle box. A zero-treatment run checks plumbing, not whether useful
spatial information must work through this encoding.

Put the state builder/adapter in `experiments/xvla_p2_box/eval_adapter.py`, independently bound by M0. Check normalized/padded
input parity, including invalid boxes and constant validity flags. Follow M0's matched-phase secondary comparison.

## Primary result

M0's primary test of A3 against A0. **Secondary (exploratory):** the `wrong_object` count (paired), success on held-out pairs.
Descriptive: A3 against A2 (both on C), by object.

## Prediction (2026-10-30, before any run)

`wrong_object` should fall toward zero: the box names the target. A gain around the historical wrong-object rate (about 6 points) is one hypothesis, **not an upper bound**:
box location and size could also affect approach, alignment or carrying. M0's power at 6 points (C = 576) is 0.60–0.90 unadjusted, 0.39–0.77 at the first Holm threshold;
interpret the interval rather than equating non-significance with no effect. Little effect on milk's hold is a prediction,
not a mechanism established by the existing failure labels.

## Caveats

- The box gives away which object to pick. A wrong-object reduction is partly by construction; it measures whether the policy
  uses the information, not whether it could ground the instruction.
- Privileged at test time: an upper bound. A detector-fed policy needs its own training/inference parity and robustness evaluation. M4a tests auxiliary
  supervision with unchanged inputs; it is not a replacement detector for A3.

## Costs

As M2: about 32 h of training, 65 GiB disk guard, ~2.5 h of M0 evaluation.

## What M3 cannot show

- Whether boxes drawn on the image, boxes of all objects, or a detector's boxes would do the same.
- Whether the gain survives detector noise.

## Out of scope

Drawn-on-image boxes (a later variant, S0 already records `props_box`), boxes of distractors, a detector.

## Plan

[../plans/2026-10-30-vla-inputs-m3-box.md](../plans/2026-10-30-vla-inputs-m3-box.md).
