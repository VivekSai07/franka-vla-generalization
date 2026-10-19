# Props P2 fresh-layout re-test (milk and bread, paired) — design

**Date:** 2026-10-19. **Subject:** X-VLA checkpoint 101,520 (the P2 winner, from the archive bundle). **Follows**
[PROPS_P2_EVALUATION_20261016.md](../../PROPS_P2_EVALUATION_20261016.md) and
[PROPS_P2_DIAGNOSTICS_20261019.md](../../PROPS_P2_DIAGNOSTICS_20261019.md). No training, no new data collection.
**Everything below is fixed before any rollout and is not moved afterwards.**

## Question

P2's trained-versus-held-out gap was 30 points and criterion 3 (≤ 10 points) failed. That comparison mixed objects: the
held-out cells are `milk→blue` and `bread→orange`, while the trained cells include easy objects (can 28/28). On the *same*
objects the penalty looked smaller (post-hoc, 14 to 20 trials per cell in 10 layouts each): milk 0.50 trained against 0.35
held-out, bread 0.86 against 0.75. **Is there an unseen-pair penalty on the same objects, and is it small?** The
earlier proposal to compare milk's penalty with bread's was dropped (below).

## Design

- **Paired layouts.** 120 fresh layouts per object (milk, bread). Each layout is run **twice**, once per bin instruction,
  with the same arrangement (which slot holds the blue bin), start pose, distractors, home jitter. Milk: trained
  `milk→orange`, held-out `milk→blue`. Bread: trained `bread→blue`, held-out `bread→orange`. One rollout per trial.
  **240 layouts × 2 instructions = 480 rollouts.**
- **Balance** (per object): distractor count 1 and 2 on 60 layouts each; blue bin left and right on 60 each; every
  (count, side) combination on 30; distractor identities from shuffled decks (every option cycled before any repeats).
  Averaged over layouts, the target bin's side is the same in the trained and the held-out arm, because the arrangement is
  balanced; side is not constant within a pair (the other bin is the target), and is reported.
- **Fresh.** Layouts are drawn with P2's own samplers (`props_p2_data.plan`: start region, yaw caps, symmetric spacing,
  hand clearance) under identities `retest:<object>:<i>`, a namespace P2 never used. Checked in tests: no seed or pose of
  the re-test coincides with any P2 layout (training, validation or reserved).
- **Expert feasibility** of every layout under both bin instructions, before any policy rollout. A pair for which the expert
  fails under either instruction is dropped **as a pair**, and reported.
- **Same everything else as P2:** same evaluator, rollout loop, 900-tick cap, success rule and failure classifier
  (`experiments/xvla_props_p2_eval/`, unchanged), OSMesa rendering (not EGL: the checkpoint was trained on OSMesa pixels),
  the same image, bundle-verified weights.

## Primary result

For every pair, `d = success(trained) − success(held-out)` ∈ {−1, 0, +1}. **Pooled penalty** = mean of `d` over all
surviving pairs, **bootstrapped over pairs, stratified by object** (10,000 resamples, seed 0), 95% percentile interval.
Layouts are the independent unit.

**Pre-registered reading** (as chosen by the user 2026-10-19):

| result | reading |
|---|---|
| lower bound of the interval > 0 | **an unseen-pair penalty exists** |
| upper bound < +10 points | **the penalty is small**: most of P2's 30-point gap is object difficulty and not the unseen pair |
| both hold | a small but real penalty |
| neither (interval contains 0 and reaches ≥ 10) | **inconclusive**; no claim either way |

## Why this design (power, computed before any rollout)

Inputs: the four P2 cell rates above (small samples: assumptions, not estimates) and the measured within-layout
correlation of two seeds (intraclass correlation **0.56**, 60 report layouts; both seeds succeed in 67% of layouts against
56% expected under independence). Simulated with a Gaussian copula that reproduces the marginals exactly (a first attempt
mis-calibrated them and was discarded):

| design | pooled penalty SE | power for a true 13-point penalty | milk-vs-bread penalty difference: power for 8 points |
|---|---|---|---|
| 60 layouts × 2 seeds per object (480 rollouts) | 4.3 points | 0.84 | 0.14 |
| **120 layouts × 1 seed** (480 rollouts) | **3.4 points** | **0.96** | 0.16 |
| 200 × 1 (800) | 2.8 | 1.00 | 0.27 |
| 400 × 1 (1,600) | 2.0 | 1.00 | 0.43 |

One seed on more layouts beats two seeds on fewer because layouts carry the information. **The earlier idea of gating on
"milk's penalty against bread's" cannot be resolved at any feasible size** (43% power for an 8-point difference at 1,600
rollouts), so that comparison is **descriptive only**.

## Reported, not gated

Per object: trained and held-out success, the pair table (both succeed / only trained / only held-out / neither), the
penalty with its own interval. By bin side and by distractor count. Failure causes per arm, including `wrong_bin`. The
discordant pairs. These carry no decision.

## What this re-test cannot show

- Whether any penalty is "unseen pair" versus "milk and bread are hard in the held-out bin for another reason" (bin
  side, bin colour). The balance cancels side and colour on average; it does not isolate them.
- Anything about the other four objects or the other eight cells.
- A claim about checkpoint 126,900 or any other checkpoint.
- A causal reading of the P2 diagnostics (milk fails at the hold): this test does not touch that.

## Out of scope

Training, new checkpoints, EGL, the P2 sealed plan (not edited), the full-protocol re-run.

## Plans

[../plans/2026-10-19-props-p2-retest.md](../plans/2026-10-19-props-p2-retest.md).
