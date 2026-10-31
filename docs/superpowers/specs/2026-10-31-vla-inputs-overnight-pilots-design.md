# Overnight pilots — screen input and loss changes before full budgets — design

**Date:** 2026-10-31. **Status: approved in outline by the user (2026-10-31: "I agree with your pilot designs"; launch in the
evening, judge in the morning). Details below are proposed and open to review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). Nothing is built.

## Why

A full X-VLA arm costs about 32 h of training plus about 5–6 h of M0 evaluation, and the GPU runs one job at a time. Ideas
beyond M2–M4 (force history, drawn boxes, all-object boxes, an auxiliary head, …) cannot each get that. A pilot answers one
question per idea, **promote to a full run or drop**, inside one night (launch about 19:00, judge about 08:00, ~13 h).

## Why not small data and a short run from scratch (the evidence)

| observation | source |
|---|---|
| X-VLA from base, 5,000-update run: 0.0% at 3,000, 20.8% at 4,000, 68.1% at 5,000; matched-exposure run: 70.8% at 12,000, 86.1% at 16,700 | RESULTS (broad workspace, A1–A3) |
| one run read 25% → 0% → 86% across checkpoints while offline metrics improved monotonically | FAILURES #21–#23 |
| A0's failures are at the hold and carry (95 of 122 `never_lifted` or `timeout_carrying`), the regime M2/M3 target | VLA_INPUTS_M0_20261031 |

A short run from scratch leaves both arms near the floor, where an input change cannot show. Pilots therefore start from a
policy that already works.

## Design A — continuation pilots (input and loss changes)

- **Start:** A0's weights (P2 X-VLA checkpoint 101,520 from the bundle, hash-verified), fresh AdamW state (the bundle keeps
  weights only), P2's optimizer settings (betas, eps, weight decay, clip, bf16 autocast, FP32 parameters).
- **Schedule:** 200-update linear warm-up to **1e-5** (the floor of A0's own schedule), then constant; **10,000 updates** at
  effective batch 8 (80,000 samples, 0.39 epoch of P2's 203,040 windows). Same sample order (fixed seed) for every arm.
- **Arms:** a **shared control** (continuation with every extra input exactly zero, i.e. A0 continued) trained once per
  recipe version and reused by every pilot of that version; each **treatment** differs from the control only by its variant
  (extra state values from the sealed S0 sidecar, or an extra loss term). X-VLA's proprioception slots 9–19 were fed zeros
  through all of P2's training, so a new input starts silent and must be learned.
- **Variants are data, not code:** a variant is a small JSON file (`variants/<name>.json`: state builder name, transform
  constants source, loss term) outside `SOURCE_FILES` and hashed into each run's record, so adding a variant never edits the
  bound package. New state builders or loss terms are code and need a package version.
- **Equality gate per variant:** with the variant's extras forced to zero (and any extra loss weight 0), 30 updates are byte-equal
  to the control's first 30 updates (metrics, weights, Adam moments). Then the night's treatment differs only by the variant.

## Design B — small-task pilots (output changes that need new demonstrations)

Only for changes the existing demonstrations cannot carry (M4b force outputs). About 120 episodes of a deliberately easy subset
(two objects, no distractors; P2 collected 540 in 54 min on 12 shards), trained from base to competence (~30,000 updates;
B1 reached 94% at 33,400 on 160 episodes) — about one night per arm, so a control and a treatment take two nights. Not
planned in detail until M4b's scoping note recommends a phase.

## Evaluation: a separate pilot tier

- **192 fresh layouts** (48 cells × 4, P2's samplers, namespace `vla_inputs:pilot`), disjoint from every P2 layout, the P2
  re-test and both M0 tiers; expert feasibility first, infeasible dropped for all pilot arms.
- Run with M0's frozen core in its own campaign root (`m0_pilot_<date>`): the core reads any plan recorded in a campaign root,
  so no core edit (verified in plan Task 1 before anything else; if the core cannot, stop and propose a versioned wrapper).
- One rollout seed per layout; 6 workers, one checkpoint per worker (FAILURES #57, #66).
- **A0 itself is evaluated once on the pilot tier** (context: what continuation alone does).

## Decision rule (pre-registered; a pilot is a screen, never a reported result)

Δ = treatment − control success on the paired pilot layouts. **Promote if Δ ≥ +5 points**, otherwise drop. Exact operating
characteristics at 192 layouts (D = fraction of discordant layouts, 0.19 / 0.30 / 0.40):

| true gain | P(promote) |
|---|---|
| 0 | 0.06 / 0.11 / 0.14 |
| +5 points | 0.51 / 0.51 / 0.51 |
| +10 points | 0.95 / 0.90 / 0.87 |
| +15 points | 1.00 / 1.00 / 0.99 |

(144 layouts would give 0.08–0.16 false promotions and 0.82–0.91 at +10; 192 was chosen for the lower false-promotion rate.)
Mechanism counts (`never_lifted`, `wrong_object`, gripper-camera target loss) are reported beside Δ and never decide.

## Calibration (before a pilot may drop an idea)

The force variant is piloted **and** trained in full (M2). If the pilot's Δ and the full arm's effect on C disagree in sign, or
the full arm is significant while the pilot dropped it, pilots may only promote, not drop, until the regime is revised.

## Night budget (measured rates; X-VLA 0.524 s/update raw, about 0.9 s with thermal pauses; ~67 s per trial)

| step | estimate |
|---|---|
| one continuation (10,000 updates) | 1.5–2.5 h |
| one pilot-tier evaluation (192 trials, 6 workers) | 0.6–0.8 h |
| night 1: control + A0 on pilot tier + 2 treatments | ~9–11 h |
| later nights: up to 3 treatments | ~7–10 h |

Training and evaluation are sequential (X-VLA training ~28 GiB plus 6 evaluation workers ~25 GiB do not fit together). A
morning summary (`summary.json` + a short `SUMMARY.md` in the night's root) lists each pilot's Δ, the decision, the mechanism
counts and any failure, so the judgement takes minutes.

## Revised programme order (proposed)

1. M1 training (running) → one or two nights of A1 evaluation on M0.
2. **Pilot night 1:** control, A0 on the pilot tier, force (M2 variant), target box (M3 variant).
3. Full M2 (calibration, regardless of its pilot) over nights or a weekend → M0 evaluation night.
4. Full M3 only if its pilot promotes; M4a as a continuation pilot (auxiliary loss variant) before any full run.
5. Further ideas from the audit's research list enter as pilots, a few per night, each needing only a variant file or, for new
   code, a package version.

## What pilots cannot show

- Effects of training from scratch: a continuation pilot asks whether information helps a policy that already works.
- Effects below about 10 points (promotion probability ~0.5 at +5).
- Anything reportable: a promoted idea earns a full run on M0's comparison tier; only that is a result.

## Plan

[../plans/2026-10-31-vla-inputs-overnight-pilots.md](../plans/2026-10-31-vla-inputs-overnight-pilots.md).
