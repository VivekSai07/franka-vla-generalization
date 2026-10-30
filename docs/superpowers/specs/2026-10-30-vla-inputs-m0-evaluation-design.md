# M0 — common evaluation protocol and the control arm — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). No code exists. Everything here is fixed before the first
rollout and not moved afterwards.

## Question

On one set of fresh P2-style layouts, how does each new arm (A1–A4) compare with the P2 control (A0, X-VLA checkpoint
101,520), paired by layout?

## What is established

| fact | source |
|---|---|
| A0 report 90/120 = 75.0% [65.0, 84.2] on P2's reserved layouts; 126,900 83/120 on the same trials | PROPS_P2_EVALUATION_20261016 |
| on those 120 identical trials the two checkpoints disagree on 23 (15 + 8): discordance 0.19 between two checkpoints of one run | computed 2026-10-30 from `outputs/xvla_props_p2_eval/report_p2_step_*` |
| milk 57.6% (trained bin) / 56.8% (held-out bin), bread 80.8% / 76.7%, 238 paired fresh layouts | PROPS_P2_RETEST_20261020 |
| two seeds of one layout correlate (ICC 0.56): one seed on more layouts is the better use of rollouts | P2 re-test design |
| the bundle re-evaluation reproduced recorded trials bit for bit in a clean container: rollouts are deterministic | TODO (P2 bundle smoke) |
| median trial 82 s; 8 pinned workers; ~7.3 GiB GPU per X-VLA worker after a checkpoint switch | P2 evaluation, FAILURES #57 |

## Design

### Cells

6 objects × 2 bins × 2 blue-bin sides × 2 distractor counts (1, 2) = **48 cells**. Of the 12 object–bin pairs, 10 were
trained in P2 and 2 were held out (`milk→blue`, `bread→orange`), so 8 cells are held-out cells.

### Tiers (one rollout seed per layout; the seed belongs to the layout and is the same for every arm)

| tier | layouts | composition | who runs it |
|---|---|---|---|
| smoke | 2 | from the selection tier | every arm, cost check only |
| selection **S** | **80** | trained pairs only: 40 trained cells × 2 | new arms only, every candidate checkpoint |
| comparison **C** | **384** | all 48 cells × 8 (320 trained-pair, 64 held-out-pair layouts) | every arm, **one** checkpoint, once |

- Layouts are drawn with P2's own samplers (start region, yaw caps, symmetric spacing, hand clearance, distractor decks) under
  identities `vla_inputs:m0:<tier>:<cell>:<i>`, a namespace no earlier plan used. Tests assert no seed, start pose or
  distractor pose coincides with any P2 training, validation or reserved layout, or any P2 re-test layout.
- **Expert feasibility** of every layout (unrendered) before any policy rollout. An infeasible layout is dropped for **all**
  arms and counted, never replaced.
- **Checkpoint selection (new arms):** candidates are the six checkpoints after update 1 (for X-VLA: 50,760, 76,140, 88,830,
  101,520, 114,210, 126,900; for pi0.5 the same fractions of its run). Winner = most successes on S; a tie goes to the later
  checkpoint (P2/P3 rule). S counts only trained pairs, so held-out cells never select anything. 80 layouts cannot rank close
  checkpoints; S exists to avoid a bad one, not to find the best.
- **A0 is not re-selected:** 101,520 was chosen on P2's report tier, which C does not share, so on C it is unselected like every
  other arm's winner.
- A completed trial is never re-run. A crashed stage resumes the missing trials only.

### Evaluator

A new package `experiments/vla_inputs_eval/` (robosuite image + GPU; pi0.5 in its own image, see M1), **not hash-bound**, with:

- a **policy adapter** interface (`reset()`, `act(observation) → 8 floats`), one adapter per model family; the observation dict
  carries both images, the 9-value state, the instruction, and the S0 signals;
- the rollout loop forked from `experiments/xvla_props_p2_eval/rollout.py` with one change: after `synchronize_observation`, it
  computes the S0 signals (`experiments/vla_inputs_signals/`) on a **scratch copy** of the simulator data and records them in
  `trace.npz` for every arm, A0 included (diagnostic data on failures at no extra rollout);
- P2's success rule, lift rule and failure classifier imported unchanged from `experiments/xvla_props_p2_eval/analysis.py`;
  the evaluator records the sha256 of every imported evaluation module in each stage's record and refuses to run an arm if
  they differ from the record of A0's run.

**Equivalence gate (before any M0 rollout is counted):** with the X-VLA adapter and checkpoint 101,520 from the bundle, the
new evaluator reproduces **four recorded P2 report trials** (two successes, two failures, chosen before running) with
`trace.npz` arrays and `episode.json` outcomes byte-equal, except the added signal arrays. This proves the fork, the adapter
and the signal tap change nothing about the rollout.

## Primary result (per arm, against A0, on C)

- **Paired McNemar**, exact, two-sided, α = 0.05, on layouts where both arms have a trial.
- **Effect:** difference in success rate with a 95% percentile interval from a paired bootstrap over layouts stratified by
  object (10,000 resamples, seed 0).
- Each arm has **one** primary test (this one). Each arm's spec names at most two secondary measures; they are labelled
  exploratory and uncorrected. Across arms the raw p-values are reported with Holm-adjusted values alongside.

### Power (`2026-10-30-vla-inputs-m0-power.py`, 4,000 repetitions, seed 0)

D = discordance (fraction of layouts where exactly one arm succeeds). Two checkpoints of one run gave 0.19; different models
or inputs should disagree more, so 0.30–0.40 is the planning range.

| true improvement | D = 0.19 | D = 0.30 | D = 0.40 |
|---|---|---|---|
| 6 points, 384 layouts | 0.74 | 0.54 | 0.43 |
| 8 points, 384 layouts | 0.95 | 0.79 | 0.68 |
| 10 points, 384 layouts | 1.00 | 0.95 | 0.85 |
| 15 points, 384 layouts | 1.00 | 1.00 | 1.00 |
| 10 points, 240 layouts | 0.95 | 0.80 | 0.66 |

**Reading:** C reliably detects a 10-point change; a 6-point change (the size M3 is predicted to produce) is close to a coin
flip. A non-significant result below 10 points is reported as **inconclusive**, not as "no effect". 384 = 48 × 8 (balanced
cells); 480 would add little at 10 points and 25% more rollouts per arm.

## Reported, not gated

Per object (milk read separately), trained-pair against held-out-pair, by distractor count, by blue-bin side, failure
causes per arm (`never_lifted`, `wrong_object`, `wrong_bin`, `timeout_carrying`, `dropped_outside`, `unstable_in_target`),
the discordant layouts, and A0's C-tier rate next to its P2 report (context only: different layouts).

## Costs (estimated from P2's measured 82 s median, 8 workers)

| run | rollouts | wall-clock |
|---|---|---|
| A0 on C | 384 | ~1.1 h |
| one new X-VLA arm (S × 6 checkpoints + C) | 480 + 384 = 864 | ~2.5 h |
| pi0.5 arm | 864 | unknown: GPU memory per worker and inference time measured in its smoke |

## What M0 cannot show

- Effects smaller than about 8 points with useful power.
- Per-object differences with any power (64 C layouts per object); per-object tables are descriptive.
- Anything about A0's earlier P2 numbers: different layouts, so A0's C rate is a fresh estimate, not a correction.

## Out of scope

EGL, a second seed per layout, a 0-distractor control tier, new objects or scenes.

## Plan

[../plans/2026-10-30-vla-inputs-m0-evaluation.md](../plans/2026-10-30-vla-inputs-m0-evaluation.md).
