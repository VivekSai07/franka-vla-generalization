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

### T2a. Mode-probability measurement
Re-run selected layouts at many sampling seeds to convert "sometimes fails" into a per-layout
probability, and to measure the tangential offset distribution directly.

- Roster: the **8 layouts with ≥1 failure** plus **8 fully-solved control layouts**, matched on
  reach. The control is essential — without it we only measure the tail and cannot tell whether the
  bad mode exists everywhere at low probability.
- **8 sampling seeds per layout** → 128 trials, ~4 GPU-hours.
- Record per trial: success, tangential and radial offset, stage timings.
- Fixed checkpoint 16700; protocol otherwise identical to `xvla_broad_workspace_long_eval`.

Decides:
- Is p(bad mode) > 0 on layouts that currently score 2/2? → does best-of-N help globally or only locally
- What N does best-of-N need? If p(bad) ≈ 0.3, N = 3 median-of-3 removes most of it
- Is the offset always +3.73 cm tangential, or a continuum sampled coarsely

### T2b. Evaluation seed budget
At the observed p = 0.861, the **same policy passes the 65/72 gate by luck 20% of the time**.
Going 2 → 4 seeds per layout cuts the estimate's sd from 4.08 pp to 2.88 pp; 8 seeds gives 2.04 pp.

The gate is currently decided partly by dice. Fold the seed increase into the protocol **before**
B1, because retrofitting it afterwards invalidates cross-phase comparisons.

---

## Tier 1 — inference-time interventions (no retraining, no new data)

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
