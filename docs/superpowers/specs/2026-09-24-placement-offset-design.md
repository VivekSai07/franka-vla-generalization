# Closing the 4.5 cm placement gap — design

**Date:** 2026-09-24
**Status:** approved (Tier 2 → Tier 1, before B1 collection design)
**Context:** [docs/RESULTS.md](../../RESULTS.md) · [docs/FAILURES.md](../../FAILURES.md) #23 · [TODO.md](../../../TODO.md) A4

## The problem

X-VLA at matched exposure scores 62/72 = 86.1% on `broad_holdout`, missing the 65/72 gate by three
trials. **All ten failures are one failure**: the cube is grasped, lifted ~29 cm, carried up to
0.80 m across the workspace, then set down beside the target instead of on top of it.

| | n | tangential offset | radial offset |
|---|---|---|---|
| successes | 62 | +0.06 cm (sd 0.68) | −0.17 cm (sd 0.57) |
| failures | 10 | **+3.73 cm (sd 0.41)** | −1.81 cm (sd 1.73) |

The systematic component is **tangential** — a bearing error of ~5° at 0.4 m radius, one cube width,
consistently counter-clockwise. The radial component is noise. The two populations are separated by
a **1.11 cm gap containing zero trials**, so this is a discrete mode, not a precision gradient.

## Cause: established by elimination

| hypothesis | test | verdict |
|---|---|---|
| imitating an imprecise expert | all 200 sealed demos: mean 0.679 cm, max 1.981 cm, 0 above 2 cm, signed bias 0.1 mm, correct stack height 200/200 | **refuted** |
| unreachable far layouts | failing layouts mean reach 0.424 m vs solved 0.472 m | **refuted, inverted** |
| sparse training coverage | 4-D nearest-neighbour distance identical (0.0348 m both groups), rank correlation τ = 0.000 | **refuted** |
| **action-sampling multimodality** | 6 of 8 failing layouts are split decisions — same scene, same weights, different sampling seed, opposite outcome; failures split 6/4 across seeds | **supported** |

Mechanism confirmed in source: `modeling_xvla.py:252` draws a fresh `torch.randn` per chunk and
integrates it over `num_denoising_steps = 10`. `predict_action_chunk(batch, noise=None)` accepts
injected noise, which is the hook for every Tier 1 intervention below.

**Consequence: this gap is not closable by data collection.** More episodes, denser coverage and
better layouts do not touch a sampling property. This is the primary input to B1 planning.

**Confidence:** the conclusion rests on 10 failures and 6 split decisions. It is the best-supported
explanation and survived three falsification attempts, but it is inferred, not measured. Tier 2
exists to measure it.

---

## Tier 2 — measurement (do first)

### T2a. Mode-probability measurement — **COMPLETE 2026-09-24**

128 trials, rc=0, 223 min. Frozen seeds reproduced the original results bit-exactly.

| group | result |
|---|---|
| control (8 layouts) | **64/64 = 100%** |
| failing (8 layouts) | 32/64 = 50% — three layouts at 1/8, four at 5–7/8, one at 4/8 |

**Answers to the questions this was built to decide:**
- *Is p(bad mode) > 0 on layouts that currently score 2/2?* **No — zero in 64 control trials.** The
  bad mode is absent from reliable layouts, not lurking at low probability.
- *What N does best-of-N need?* **Best-of-N does not solve this.** Majority selection drives each
  layout to its dominant mode: it fixes the four good-dominant layouts and pins the three
  bad-dominant ones at ~0%. Net ≈ 62 → 65 of 72, for N× inference cost forever.
- *Is the offset a continuum or discrete?* **Discrete in outcome** — successes +0.150 (sd 0.715),
  failures +3.893 (sd 0.723), empty band between. But see the intent caveat below.

**Unplanned findings:**
- There is **no global bias** — successes sit at +0.15 cm once controls are included.
- Failure is **layout-specific and itself bimodal**; no middle group.
- **Two seeds per layout are worthless for per-layout attribution** (024 and 029 both read 1/2;
  actual 87.5% and 12.5%). Aggregate capability from 2 seeds was fine.
- **No geometric feature explains the three hard layouts** (dx, dy, distance, Δradius, Δbearing,
  target radius; max separation 0.51 pooled sd).

**Caveat:** the measurement is the cube's final resting position, not the aim point. A marginal
placement tips and slides, so part of the empty band may be physics. Bimodality of *intent* is
untested and needs predicted action chunks rather than rollout outcomes.

### T2b. Evaluation seed budget — **now established, not proposed**
At the observed p, the same policy passes the 65/72 gate by luck 20% of the time, and T2a showed
2 seeds cannot distinguish 12.5% from 87.5% per layout. **≥8 seeds per layout is required for any
per-layout claim.** Aggregate rates survive at 2 seeds; per-layout ones do not.

Fold this into the protocol **before** B1 — retrofitting it invalidates cross-phase comparisons.

---

## Tier 1 — inference-time interventions (no retraining, no new data)

**Downgraded by T2a.** Best-of-N is now projected to move the 72-trial protocol from 62 to ~65 —
marginal, and it cannot help the three layouts where the wrong mode is dominant. Worth doing only
if the gate matters more than the inference cost. Not a blocker for B1.

Run after T2a says which are worth it. Each changes inference only, so each needs its own package
and must be reported as a **separate protocol** — never compared directly to the sealed 86.1%
unless layouts and seeds are held identical.

### T1a. Raise `num_denoising_steps` (cheapest)
10 → 30. A config change at inference. May sharpen mode separation at ~3× inference cost.
Test on the 8 failing layouts × 2 seeds = 16 trials (~40 min) before committing to a full protocol.

### T1b. Best-of-N with mode selection
Draw N action chunks via the `noise` argument, aggregate by median or largest cluster.
Cost: N× inference. Most likely to work if T2a shows the bad mode is a minority draw.
Open design question: median across chunks may interpolate *between* two modes and land in the
1.11 cm gap — cluster-then-pick-largest is safer than averaging.

### T1c. Temporal ensembling
`n_action_steps = 10` inside `chunk_size = 30` means each timestep already has ~3 overlapping
predictions that are currently discarded. Averaging them suppresses one-off bad draws at **zero**
extra inference cost. Same interpolation caveat as T1b.

---

## Tier 3 — data design for B1 (low expected value for this gap)

Carry into B1 planning, but do not expect them to close this gap:

- **Approach-direction consistency in the expert.** If demonstrations reach targets from varied
  bearings, a multimodal action distribution is being taught by construction. The only data-side
  lever with a plausible mechanism. Auditable against the sealed dataset — cheap, CPU-only.
- **Terminal-phase supervision density** — more frames through the final descent. Hypothesis only.
- **Explicitly not worth doing:** more episodes, denser layout coverage. τ = 0.000.

---

## Success criteria

- **T2a**: per-layout p(bad mode) with CIs for both failing and control groups; a stated N for
  best-of-N, or evidence that best-of-N cannot work.
- **T1**: any intervention that raises `broad_holdout` above 86.1% **on the identical roster and
  seeds**, with the tangential-offset distribution shown to have lost its +3.73 cm cluster.
- Whole effort: a defensible answer to "is 65/72 the right gate", argued on the protocol's own
  terms rather than fitted to whatever number we got.

## Non-goals

- Retraining. Exposure is already paid off; nothing here needs new weights.
- New data collection. Ruled out by the elimination table.
- Lowering the gate to match the result.
