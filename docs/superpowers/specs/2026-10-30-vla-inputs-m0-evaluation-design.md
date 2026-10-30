# M0 — common evaluation protocol and the control arm — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). No code exists. Everything here is fixed before the first
rollout and not moved afterwards.

**Review revision:** [audit R1/R4/R5](../../VLA_INPUTS_AUDIT_20261030.md). This is a selected-system,
single-training-seed pilot. C estimates performance of these fixed policies on the declared P2-style layout distribution,
not variability across training runs. Promising method claims require replicated treatment/control training seeds.

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
| comparison **C** | **384** | all 48 cells × 8 (320 trained-pair, 64 held-out-pair layouts) | A0 fixed at 101,520; each new arm one S-selected checkpoint plus the predeclared 4.0-epoch secondary checkpoint (deduplicate if identical) |

- Layouts are drawn with P2's own samplers (start region, yaw caps, symmetric spacing, hand clearance, distractor decks) under
  identities `vla_inputs:m0:<tier>:<cell>:<i>`, a namespace no earlier plan used. Tests assert no seed, start pose or
  distractor pose coincides with any P2 training, validation or reserved layout, or any P2 re-test layout.
- **Expert feasibility** of every layout (unrendered) before any policy rollout. An infeasible layout is dropped for **all**
  arms and counted, never replaced.
- **Checkpoint selection (new arms):** candidates are the six checkpoints after update 1 (for X-VLA: 50,760, 76,140, 88,830,
  101,520, 114,210, 126,900; for pi0.5 the same fractions of its run). Winner = most successes on S; a tie goes to the later
  checkpoint (P2/P3 rule). S counts only trained pairs, so held-out cells never select anything. 80 layouts cannot rank close
  checkpoints; S exists to avoid a bad one, not to find the best.
- **A0 is not re-selected:** 101,520 was chosen on P2's report tier, which C does not share, so C is independent of that historical selection. Its selection procedure differs from the new arms;
  the matched-phase secondary contrast below makes this limitation explicit.
- A completed trial is never re-run. A crashed stage resumes the missing trials only.

### Evaluator

A new package `experiments/vla_inputs_eval/` (robosuite image + GPU; pi0.5 in its own image, see M1), with a **frozen, hash-bound shared core** and separately bound arm adapters:

- a **policy adapter** interface (`reset()`, `act(observation) → 8 floats`), one adapter per model family; the observation dict
  carries both images, the 9-value state, the instruction, and the S0 signals;
- the rollout loop forked from `experiments/xvla_props_p2_eval/rollout.py` with one change: after `synchronize_observation`, it
  computes the S0 signals (`experiments/vla_inputs_signals/`) on a **scratch copy** of the simulator data and records them in
  `trace.npz` for every arm, A0 included (diagnostic data on failures at no extra rollout);
- P2's success rule, lift rule and failure classifier imported unchanged from `experiments/xvla_props_p2_eval/analysis.py`;
  the evaluator records the source closure of the shared core (rollout, scene restoration, labels, statistics, signal tap,
  layout and feasibility manifests) and requires equality with A0. Each arm adapter/state builder lives outside that core,
  is loaded lazily by an explicit module identifier, and is separately bound with its weights, processors, statistics,
  runtime digest and dependencies. Different declared adapters are expected; undeclared changes are rejected.
  Freeze this boundary before A0. A shared-core correction creates a new protocol/campaign with a new control; it is never
  resolved by ignoring a hash mismatch. Later arm work must not modify the shared `adapters.py` interface.

**Equivalence gate (before any M0 rollout is counted):** with the X-VLA adapter and checkpoint 101,520 from the bundle, the
new evaluator reproduces **four recorded P2 report trials** (two successes, two failures, chosen before running) with
`trace.npz` arrays and deterministic `episode.json` outcome fields byte-equal, except the added signal arrays (wall-clock
duration and output paths are excluded). This tests that the fork, the adapter
and the signal tap change nothing about the rollout.

## Primary result (per arm, against A0, on C)

- **Paired McNemar**, exact, two-sided, α = 0.05, on layouts where both arms have a trial.
- **Effect:** difference in success rate with a 95% percentile interval from a paired bootstrap over layouts stratified by
  object (10,000 resamples, seed 0).
- Each arm has **one** primary test (this one). Each arm's spec names exploratory secondary measures, in addition to the matched-phase contrast; they are labelled
  exploratory and uncorrected. Freeze the family A1–A4 before C; report raw p-values and Holm-adjusted values for all four
  hypotheses (unrun arms conservatively entered as p=1). A programme-wide superiority claim requires the adjusted test.
  The original power table below uses unadjusted alpha .05; it is not familywise power.

**Additional predeclared comparison:** evaluate the 4.0-epoch checkpoint of each new arm against A0 (also 4.0 epochs),
as a secondary matched-phase contrast, even if S selects a different winner. Do not choose between the two using C.
Deduplicate if the checkpoints coincide. Budget up to 384 additional trials per arm. This controls selection-phase
differences but does not turn M1 into an architecture-only comparison. Selection remains trained-pair-only;
the primary C population includes held-out pairs and is explicitly a different target population.

**Holdout lifecycle:** freeze all recipes and analysis decisions before viewing C. Later decisions informed by C need a
new confirmation set, even if each checkpoint ran on C only once. Resume infrastructure failures; count policy failures
(including nonfinite actions) as failures. Do not publish a complete comparison until all planned feasible paired trials
exist. Report feasibility exclusions by cell and recompute power at the fixed feasible N before inspecting policy outcomes.

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

**Reading:** under the unadjusted planning assumptions, C has the tabulated power for a 10-point change; a 6-point change is close to a coin
flip. A non-significant result below 10 points is reported as **inconclusive**, not as "no effect". 384 = 48 × 8 (balanced cells). Cost/power choices must also consider multiplicity and feasibility losses;
do not infer that 480 adds little from the unadjusted table alone.

Audit exact enumeration at alpha .0125 (conservative first threshold for four Holm tests), N=384:
for D=.19/.30/.40, power is .540/.330/.238 at +6 points and .980/.851/.708 at +10 points.
Formula and full comparison with .05: [audit R4](../../VLA_INPUTS_AUDIT_20261030.md).
These are marginal sensitivities, not exact joint Holm power.

## Reported, not gated

Per object (milk read separately), trained-pair against held-out-pair, by distractor count, by blue-bin side, failure
causes per arm (`never_lifted`, `wrong_object`, `wrong_bin`, `timeout_carrying`, `dropped_outside`, `unstable_in_target`),
the discordant layouts, and A0's C-tier rate next to its P2 report (context only: different layouts).

## Costs (throughput arithmetic from P2's measured 82 s median, conditional on 8 workers)

Eight workers at the reported 7.3 GiB peak require 58.4 GiB and cannot fit on the 48 GiB GPU. The table is not a feasible
resource reservation. Choose workers from measured switch/load peaks and available memory with headroom, then recompute
wall time including the matched-phase contrast. Only layout generation/feasibility are CPU-only; GPU rollouts must be
scheduled separately from M1 training unless their measured combined peaks fit. Include videos in disk planning.

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
