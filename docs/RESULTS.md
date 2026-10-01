# Results

All benchmark numbers. Protocols differ in difficulty and are **not** comparable across
rows unless stated. Chronological narrative lives in [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md);
failures and lessons in [FAILURES.md](FAILURES.md).

## Current status (2026-09-30)

**Headline — B2: distractor cubes had no detectable effect on success; more training helped a lot;
most failures are release precision, not perception.** X-VLA trained 50,000 updates on
`b2_distractor_20260928` (200 episodes: 40 control, 160 with 1–2 distractor cubes). Report
protocol: 160 trials on 40 held-out layouts, each run in both conditions, 2 seeds. Full analysis:
[B2_DISTRACTOR_EVALUATION_20260930.md](B2_DISTRACTOR_EVALUATION_20260930.md).

| checkpoint | overall | layout-cluster 95% CI | control | distractor-present |
|---|---|---|---|---|
| 33,400 | 132/160 = 82.5% | [73.1, 90.6] | 67/80 = 83.8% | 65/80 = 81.2% |
| **50,000** | **147/160 = 91.9%** | [83.8, 98.1] | 74/80 = 92.5% | 73/80 = 91.2% |

- **Distractor contrast (within-layout, control − distractor):** +2.5 pts at 33,400 (CI [−7.5,
  +12.5], p = 0.82) and +1.25 pts at 50,000 (CI [−5.0, +8.8], p = 1.0). Not detectable; the
  interval excludes a drop larger than about 9–12 points. **No distractor was ever touched**
  (max displacement 0.064 mm over 160 trials), so B2 tested *visual* distraction only.
- **50,000 vs 33,400, paired on identical layout/condition/seed:** 20 trials only 50,000 solved,
  5 only 33,400 (McNemar p = 0.004). Screening tied them at 17/20; the chain's first-maximum
  tie-break picked 33,400, the worse one (FAILURES #37). The 91.9% is the larger of two reported
  results, so a fresh holdout is the clean confirmation.
- **Control arm vs B1's 94.4%:** same range (92.5% at 50,000), but different layouts, selection and
  LR phase. A sanity check, not a delta.

**The 41 failures (of 320 trials) are layout-specific and have identifiable causes.** Five layouts
hold all 13 failures at 50,000 (permutation p = 0.0002). The scripted expert solved 23/23 attempts,
including every failing layout, so none is infeasible. First-cause mechanisms: **release precision
21 (51%)** — success falls from 98% to 22% to 0% as the cube is released 1.5 → 2.0–2.5 → > 2.5 cm
off-centre, matching the 2.1 cm tipping limit of a 4.2 cm cube; **acquisition failures 11 (27%)**,
mostly two layouts with a mis-positioned grasp; **hesitation/slowness 7 (17%)**; **wrong cube 2**.
The policy's release offset has a fatter tail than the expert's (8.7% / 2.6% of releases > 2 cm at
33,400 / 50,000 vs 0.5% for the expert).

Open and **unexplained**: failures concentrate in the *familiar* arrangement (11/22 layouts vs
1/18 swapped at 33,400). Exploratory, from roughly thirty uncorrected tests; replicate before
relying on it.

## Prior status (2026-09-26)

**Headline — B1 shows the trained policy reads the instruction, not the cube's position.** The
2×2 factorial dataset (instruction × colour/region arrangement) was designed to break the exact
shortcut discovered in #27, where colour was perfectly predictable from `sign(y)`. On the report
protocol — 160 trials on 40 **held-out** layouts never used for checkpoint selection, all four
cells (pick colour × arrangement) — the trained policy scored **151/160 = 94.4%**, 95% CI
[89.7, 97.0].

| contrast | rate | 95% CI | Fisher p vs. other level |
|---|---|---|---|
| pick = red (C1+C2) | 77/80 = 96.2% | [89.5, 98.7] | 0.495 |
| pick = green (C3+C4) | 74/80 = 92.5% | [84.6, 96.5] | (ref) |
| arrangement = familiar (C1+C3) | 75/80 = 93.8% | [86.2, 97.3] | 1.000 |
| arrangement = swapped (C2+C4) | 76/80 = 95.0% | [87.8, 98.0] | (ref) |

**Neither contrast is significant, and swapped is not worse than familiar.** That second point is
the load-bearing one: a policy using the old positional shortcut (ignore colour, always reach for
the cube in a fixed region) would succeed on *familiar* layouts by coincidence and fail on
*swapped* ones, since swapping relocates the correct-coloured cube to the region the shortcut
would ignore. Familiar and swapped scored within one trial of each other (p = 1.0). This is the
first B1 evidence that the policy is actually conditioning on the instruction/colour rather than
on a positional prior.

| cell | pick → place | arrangement | success |
|---|---|---|---|
| C1 | red → green | familiar | 38/40 = 95.0% |
| C2 | red → green | swapped | 39/40 = 97.5% |
| C3 | green → red | familiar | 37/40 = 92.5% |
| C4 | green → red | swapped | 37/40 = 92.5% |

Per-layout clustering is mild: of the 40 report layouts, 34 scored 4/4, 3 scored 3/4, 3 scored
2/4; none scored 0/4 or 1/4. No held-out layout is categorically unsolvable, unlike the T2a
placement-mode result where per-layout rates ranged 12.5%–87.5%.

**Checkpoint selection (screening, cell C1 only, 40 trials each, on a *disjoint* 40 layouts):**

| checkpoint | epochs | screen success | vs. winner (Fisher p) |
|---|---|---|---|
| 16,700 (matched exposure) | 1.67 | 22/40 = 55.0% | 1e-6 |
| 25,000 | 2.50 | 34/40 = 85.0% | 0.026 |
| **33,400 (winner)** | **3.35** | **40/40 = 100%** | — |

Monotonic and significant at every step — B1 needed the doubled horizon. Matched exposure
(16,700, the same budget as the 86.1% broad-workspace run) would have been badly undertrained
here: B1 splits the same 160 training episodes across two instructions and four cells, so each
task saw half the exposure that run gave its single task.

`unassisted_success` (zero joint/gripper-limit interventions) is 0/160, matching 0/72 on the
86.1% broad-workspace run — expected, not a regression; every rollout across this repo clips at
the actuator limits at some point during the grasp.

`b1_language_20260925` is **collected and sealed**: 200 episodes, cells C1–C4 balanced 50/50/50/50,
100 red-pick/100 green-pick, seal `b152cba2e035653d7a800d3a5e5112005e85080300e0616483d9fe74837fdf22`.
Trained to 33,400 updates (3.35 epochs); see [FAILURES.md](FAILURES.md) #28–#31 for the fork
issues resolved along the way (five single-task assumptions, the red-on-green cube binding, an
infeasible layout, and two inherited resource guards).

## Prior status (2026-09-24)

**Headline — matched exposure took X-VLA from 68.1% to 86.1% on the broad holdout.** The 0.50-epoch
run was undertrained, as suspected. Retraining on the identical sealed data at 1.67 epochs
(16,700 updates) scored **62/72**, a significant gain (Fisher exact p = 0.0165). It still fails the
65/72 gate, by three trials.

| X-VLA on `broad_workspace` | updates | epochs | broad_holdout | 95% CI |
|---|---|---|---|---|
| short run | 5,000 | 0.50 | 49/72 = 68.1% | [56.6, 77.7] |
| **matched-exposure run** | **16,700** | **1.67** | **62/72 = 86.1%** | **[76.3, 92.3]** |

`fixed_parity` stayed at 20/20. The gate has **not** been lowered to match the outcome.

> **What this result does not cover.** `plan.py` draws the cubes from disjoint regions — red always
> at y<0, green always at y>0, verified across 200/200 training episodes and 80/80 reserved layouts,
> separated by a 40 cm gap. "Pick the cube at −y and place it on the cube at +y" therefore solves
> 100% of the data, and every result above is consistent with a policy that never perceives colour
> and never reads the instruction. **These numbers measure spatial capability only; the language
> pathway is untested.** See [FAILURES.md](FAILURES.md) #27 and the B1 design at
> [superpowers/specs/2026-09-24-b1-language-grounding-design.md](superpowers/specs/2026-09-24-b1-language-grounding-design.md).
>
> **Vertical capability is untested for the same reason.** Initial pick and place cube heights each
> take exactly one value across all 200 episodes (spread 0.00e+00) and the stack height spreads
> 1.5e-05 m, because the table is flat and the cubes identical. Approach altitude *is* varied
> (±3 cm hover jitter, 200 distinct values), but the terminal target is not, so an absolute-Z policy
> satisfies every episode. The spatial result is a **horizontal** result. See TODO B1b.

`broad_workspace_data_v1` is **collected and sealed**: 200 episodes, 105,600 frames, 160 train / 40 validation,
seal `0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85`. Both policies were retrained on it
and evaluated on the 36 reserved holdout layouts that collection never executed.

**Broad-workspace results** — 36 reserved F7 layouts × 2 policy seeds, plus 20 fixed-parity seeds:

| Metric | SmolVLA | X-VLA | Δ |
|---|---|---|---|
| broad_holdout success rate | 31.9% (23/72) | **68.1% (49/72)** | **+36.2 pp** |
| fixed_parity success rate | (pending) | 100% (20/20) | — |
| offline train arm MAE (rad) | 0.01904 | 0.01655 | −13% |
| offline val arm MAE (rad) | 0.02157 | 0.01748 | −19% |
| layouts failing both seeds | 19 / 36 | 8 / 36 | — |

**Both policies fail the 65/72 holdout gate.** All failures are timeouts, none nonfinite. The gate was set for
the easier local-varied protocol and has deliberately not been lowered to match the outcome.

The 8 layouts X-VLA fails on both seeds were screened with the scripted expert: **all 8 are solvable**. The
holdout is sound, so 68.1% is a real policy limit rather than an artefact of unverified layouts.

**Did the 200-episode collection actually buy generalisation? Yes — decisively.** The same two protocols run
against the *local-varied* checkpoints on the identical 72-trial roster:

| Checkpoint trained on | X-VLA | SmolVLA |
|---|---|---|
| local-varied (48 eps, ±1 cm) | 2.8% (2/72) — 95% CI [0.8, 9.6] | 2.8% (2/72) — [0.8, 9.6] |
| broad-workspace (160 eps, F7) | **68.1%** (49/72) — [56.6, 77.7] | **31.9%** (23/72) — [22.3, 43.4] |
| gain | **+65.3 pp** | **+29.2 pp** |

Confidence intervals do not overlap. Both local-varied models succeeded on exactly one layout —
`evaluation_031`, on both seeds — which is the **closest of all 80** holdout layouts to their training base
(distance 0.0297 vs 0.2209 max). So the 2.8% is not noise: it is the single holdout layout that happens to
fall inside the ±1 cm neighbourhood they were trained in. Outside it, they have no competence at all, despite
scoring 100% on `fixed_parity`.

Two caveats on that claim:
- The comparison confounds **breadth with volume** (48 vs 160 episodes). The defensible statement is that the
  broad-workspace *collection* produced the capability, not that envelope breadth specifically did.
- The SmolVLA cross-eval runs under a data-level binding (`metadata_binding_reproducible: False`): that run
  predates the `local_varied_data` re-seal, so its metadata binding is unreproducible. Training data identity
  is instead proven by recomputed statistics, and checkpoint identity by weight hash.

**Resolved: 68.1% was undertraining, not a ceiling.** X-VLA saw only **0.50 epochs** on the broad
dataset (40,000 chunks over 79,840 starts) versus 1.67 on local-varied — updates were held at 5000
while the data tripled. Across the whole programme, exposure fell as data grew, so data volume and
exposure were perfectly anti-correlated:

| dataset | episodes | X-VLA epochs | SmolVLA epochs |
|---|---|---|---|
| `single_task` | 9 | **8.91** | — |
| `fixed_layout` | 20 | 4.01 | 16.70 |
| `local_varied` | 48 | 1.67 | 6.96 |
| `broad_workspace` | 160 | 0.50 | 2.09 |

Rerunning at matched exposure (16,700 updates = 1.67 epochs) resolved it: **68.1% → 86.1%**.

### The checkpoint curve is non-monotonic — and offline metrics do not track capability

Same policy, same holdout, four checkpoints of the matched-exposure run:

| update | lr | epochs | broad_holdout | grasped | lifted >5 cm | median final xy | train MAE | val MAE |
|---|---|---|---|---|---|---|---|---|
| 4,000 | 9.00e-5 | 0.40 | 18/72 = 25.0% | 72 | 56 | 5.53 cm | 0.03390 | 0.03475 |
| 8,000 | 6.02e-5 | 0.80 | **0/72 = 0.0%** | 72 | **17** | **62.72 cm** | 0.02621 | 0.02637 |
| 12,000 | 2.74e-5 | 1.20 | 51/72 = 70.8% | 72 | 69 | 1.24 cm | 0.01406 | 0.01509 |
| 16,700 | 1.00e-5 | 1.67 | **62/72 = 86.1%** | 72 | 72 | 0.85 cm | 0.01022 | 0.01258 |

Success goes 25% → **0%** → 71% → 86%. At update 8,000 the policy still grasps on every trial but
cannot lift or transport — the cube ends a median 63 cm from target. Meanwhile **offline action MAE
improves monotonically throughout**, falling 23% across the very interval where success collapses to
zero.

**The mid-training collapse is reproducible, not a fluke.** The short run also passed through
**0/72** at update 3,000 (60% of its schedule) before reaching 68.1% at its endpoint. Two
independent runs both collapse to exactly zero near mid-schedule and both recover. Mid-training
checkpoints are therefore not a cheap proxy for the endpoint, in either direction.

Consequence: **arm MAE, validation loss and gripper accuracy are not valid proxies for task success.**
Never select a checkpoint, trigger early stopping, or claim progress from them. Only rollouts count.
This also means the val/train ratio should not be read as an overfitting signal on its own.

### Exposure, not annealing — resolved

The two effects can be separated by comparing checkpoints at matched **schedule phase** rather than
matched step count:

| comparison | lr | epochs | broad_holdout | |
|---|---|---|---|---|
| short @ 4,000 | 2.05e-5 (80% annealed) | 0.40 | 15/72 = 20.8% | |
| long @ 4,000 | 9.00e-5 (24% annealed) | 0.40 | 18/72 = 25.0% | **p = 0.69** |
| short @ 4,000 | 2.05e-5 (80% annealed) | 0.40 | 15/72 = 20.8% | |
| long @ 12,000 | 2.74e-5 (72% annealed) | 1.20 | 51/72 = 70.8% | **p = 2.2e-9** |

At matched **epochs**, a 4.4× difference in learning rate changes nothing (p = 0.69). At matched
**learning rate**, 3× the exposure is worth +50 points (p = 2.2e-9). Exposure is the driver; the
anneal is not. The earlier caveat that the curve could not separate the two is now resolved.

One qualification: capability is not a monotone function of epochs either — 0.80 epochs scores 0%
while 0.40 scores 25%. The mid-schedule collapse sits on top of the exposure trend.

### Anatomy of the 10 remaining failures — a systematic 4.5 cm placement offset

All ten failures at 16,700 are the same failure, and it is not a spatial-understanding failure. A
representative trial grasps at tick 102, lifts **29 cm**, carries the cube **58 cm** across the
workspace, releases — and never forms a stable stack.

| | successes (n=62) | failures (n=10) |
|---|---|---|
| final xy error | mean 0.80 cm | mean 4.51 cm |

The cube is set down *beside* the target, touching it, instead of on top — roughly one cube width.

> **Superseded in part.** The spread of those 10 failures was originally reported as sd 0.20 cm
> ("2 mm across ten trials"). That was a **between-layout** statistic — the 10 failures came from 8
> different layouts, one or two trials each — presented as if it described within-layout sampling
> variability. The 128-trial T2a measurement below gives the correct figure: failures sit at
> +3.893 cm with sd **0.723**. The directional claim that failures land "toward the robot base" was
> also wrong; the offset is tangential, and there is no global radial bias.

The obvious explanation — unreachable far layouts — is **refuted and inverted**: failing layouts
average 0.424 m reach, fully-solved ones 0.472 m. The far layouts are solved. This is not kinematics.

Decomposed about the robot base, the systematic component is **tangential** — a bearing error, not a
reach error. Authoritative values are from T2a (128 trials, 16 layouts × 8 seeds):

| | n | tangential |
|---|---|---|
| successes | 96 | +0.150 cm (sd 0.715) |
| failures | 32 | **+3.893 cm (sd 0.723)** |

+3.9 cm at ~0.4 m radius is a ~5° azimuth error. The two populations are separated by an empty band,
so this is a discrete mode in **outcome** rather than a precision gradient — though see the
intent caveat in the T2a section below.

**Cause: action-sampling multimodality, layout-specific.** `modeling_xvla.py:252` draws a fresh
`torch.randn` per chunk and integrates over `num_denoising_steps = 10`; the action distribution at
placement has two modes and the rollout draws one. T2a showed which mode dominates is a property of
the *layout*: three layouts draw the wrong one 7 times in 8, control layouts never draw it at all.

Coverage density was tested as an alternative and **refuted**: 4-D nearest-neighbour distance to the
training layouts is identical for both groups (0.0348 m), with rank correlation τ = 0.000 between
sparsity and failure. The two sparsest holdout layouts are both 2/2 solved.

**Therefore this gap is not closable by data collection.** Remedies are inference-time; see the
design spec at
[docs/superpowers/specs/2026-09-24-placement-offset-design.md](superpowers/specs/2026-09-24-placement-offset-design.md).

### T2a mode-probability measurement (128 trials, complete)

16 layouts × 8 sampling seeds at checkpoint 16,700 — 8 layouts that had produced ≥1 failure, plus
8 reach-matched controls that had scored 2/2. The two frozen seeds reproduced the original results
bit-exactly, validating the protocol.

| group | result |
|---|---|
| **control** (8 layouts) | **64/64 = 100%** — every layout 8/8 |
| failing (8 layouts) | 32/64 = 50% |

| layout | p(good mode) | what 2 seeds had said |
|---|---|---|
| evaluation_024 | 7/8 | 1/2 |
| evaluation_030 | 7/8 | 1/2 |
| evaluation_019 | 6/8 | 1/2 |
| evaluation_022 | 5/8 | 1/2 |
| evaluation_010 | 4/8 | 1/2 |
| evaluation_008 | **1/8** | 0/2 |
| evaluation_017 | **1/8** | 0/2 |
| evaluation_029 | **1/8** | 1/2 |

Findings:

- **The outcome distribution is bimodal.** Successes cluster at +0.150 cm (sd 0.715, n=96),
  failures at +3.893 cm (sd 0.723, n=32), with an empty band between — and the cluster boundary is
  exactly the success boundary.
- **There is no global bias.** With controls included, successful placements sit at +0.15 cm. The
  +0.579 cm seen in the failing group alone is layout-specific, not a systematic model offset.
- **The bad mode is absent from control layouts entirely** — 0 occurrences in 64 trials.
- **Failure is layout-specific and itself bimodal**: three layouts draw the wrong mode 7 times in 8;
  the rest draw it rarely or never. There is no middle group.
- **Two seeds per layout cannot distinguish 12.5% from 87.5%.** `evaluation_024` and
  `evaluation_029` both looked like 1/2 and are actually 87.5% and 12.5%. Aggregate capability from
  2 seeds was roughly right; **per-layout attribution from 2 seeds was worthless**.
- **No geometric feature separates the three hard layouts.** Inter-cube dx, dy, distance, radius
  difference, bearing difference and target radius all fail (max separation 0.51 pooled sd); control
  `evaluation_031` has a smaller dx than any hard layout and scores 8/8.

Projected true capability at 8 seeds × 36 layouts ≈ **88.9%** (assuming solved layouts behave like
the controls), against 86.1% measured at 2 seeds. The 65/72 gate (90.3%) still fails.

Consequence for the remedies: best-of-N with majority selection drives each layout to its *dominant*
mode, which fixes the four good-dominant layouts and pins the three bad-dominant ones at ~0%. Net
movement on the 72-trial protocol is roughly 62 → 65, i.e. marginal, in exchange for N× inference
cost on every trial. For `evaluation_008`, `_017` and `_029` the model's *dominant belief* about
where the cube goes is wrong — a model error, not sampling noise, and no resampling scheme fixes it.

Caveat: the measured quantity is the cube's final resting position, not the policy's aim point. A
marginal placement tips and slides, so part of the empty band may be physics rather than intent.
Testing bimodality of *intent* requires the predicted action chunks, not rollout outcomes.

One consequence for the gate: at the observed p = 0.861, the *same policy* passes 65/72 by sampling
luck **20%** of the time. With 2 seeds per layout the gate is partly measuring dice.

Longer training fixed 17 of the 23 prior failures and eliminated every gross transport error
(`evaluation_004` and `_005`, previously missing by 0.72–0.88 m, are now 2/2). It broke 4; 6 persist
in both runs.

### The expert demonstrations are clean — the offset is introduced by the policy

Hypothesis tested: the policy is faithfully imitating a teacher that itself places imprecisely.
**Refuted.** Across all 200 sealed expert episodes (`validation.json` → `diagnostics.final_delta_xyz`):

| expert final placement | value |
|---|---|
| mean xy error | 0.679 cm |
| median / sd | 0.635 cm / 0.377 cm |
| max | 1.981 cm |
| episodes > 2 cm | **0** |
| mean signed offset | dx +0.1 mm, dy −0.3 mm |
| dz within [3.8, 4.6] cm | **200 / 200** |

Every demonstration ends in a clean stack with no directional bias. So the 4.5 cm offset is a
policy-side artefact, not a data-quality defect — and **more episodes of the same kind should not be
expected to fix it**. Note the policy's *successes* (0.80 cm) essentially match expert precision
(0.68 cm); the failure is a discrete targeting mode, not degraded precision.

**Earlier local-varied results** (narrow ±1 cm protocol, not comparable to the above — different task difficulty):

| Metric | SmolVLA | X-VLA |
|---|---|---|
| varied_holdout success rate | 88.9% (64/72) | **98.6% (71/72)** |
| fixed_parity success rate | 100% (20/20) | 100% (20/20) |
| offline val arm MAE (rad) | 0.01133 | 0.00996 |

The cross-evaluation that gives 68.1% its reference point is complete and reported above: the
local-varied checkpoints score 2.8% on the identical broad holdout.

---

## Datasets & training runs

| Package | Purpose | Episodes | Status |
|---|---|---|---|
| `experiments/fixed_layout/` | Single-geometry pilot (source reset ep 7) | 24 | sealed |
| `experiments/local_varied_data/` | 60 layouts, ±1° pose jitter | 60 (48 train / 12 val) | sealed |
| `experiments/broad_workspace_data/` | F7 envelope, retries permitted | 200 (160 train / 40 val) | **sealed** |
| `experiments/{smolvla,xvla}_local_varied/` | Local-varied finetunes | — | trained + evaluated |
| `experiments/{smolvla,xvla}_broad_workspace/` | Broad-workspace finetunes | — | trained + evaluated |
| `experiments/xvla_broad_workspace_long/` | Matched-exposure X-VLA (16,700 updates) | — | trained + evaluated |
| `experiments/xvla_broad_workspace_long_eval/` | Eval bound to the matched-exposure run | — | complete |
| `experiments/xvla_localvaried_broad_eval/` | Cross-eval: old checkpoint, new holdout | — | complete |

Training reports: `outputs/train/{smolvla,xvla}_franka_stacking_{localvaried_v1_20260919,broadworkspace_v1_20260921}/result.json`.
Evaluation reports: `.../evaluation/{varied_holdout,broad_holdout,fixed_parity}/result.json`, per-trial `episode.json`, `trace.npz`, `cameras.mp4`.

---

## Full experiment progression

Every trained policy to date, oldest first. Protocols differ in difficulty and are **not**
comparable across rows — each is noted.

| # | Training data | Protocol | SmolVLA | X-VLA |
|---|---|---|---|---|
| 1 | `single_task` (9 episodes, IDs 0–8) | 9 seen + 4 held-out, gate 8/9 and 3/4 | seen **1/9**, held-out **0/4** — failed | seen **4/9** (44.4%), held-out **0/4** — failed |
| 2 | `fixed_layout` (24 episodes, one geometry) | 20 seeds on the source reset | **18/20** (90%) | **20/20** (100%) |
| 3 | `local_varied` (60 eps, ±1 cm) | `varied_holdout` 24 layouts × 3 seeds | **64/72** (88.9%) | **71/72** (98.6%) |
| 3 | `local_varied` | `fixed_parity` 20 seeds | 20/20 (100%) | 20/20 (100%) |
| 4 | `broad_workspace` (200 eps, F7) | `broad_holdout` 36 layouts × 2 seeds | **23/72** (31.9%) | **49/72** (68.1%) |
| 4 | `broad_workspace` | `fixed_parity` 20 seeds | 20/20 (100%) | 20/20 (100%) |
| 5 | `broad_workspace`, **matched exposure** (1.67 ep) | `broad_holdout` 36 layouts × 2 seeds | not run | **62/72** (86.1%) |
| 5 | `broad_workspace`, matched exposure | `fixed_parity` 20 seeds | not run | 20/20 (100%) |
| — | `local_varied` checkpoints | `broad_holdout` (cross-eval) | 2/72 (2.8%) | 2/72 (2.8%) |

Reading across: both architectures failed the 9-episode single-task gate outright, reached
ceiling on one fixed geometry at 24 episodes, and generalised within a ±1 cm neighbourhood at 60.
Only the 200-episode F7 dataset produced any wide-workspace competence at all — the cross-eval row
shows the local-varied policies at 2.8% on the same layouts. X-VLA leads SmolVLA at every stage,
and the margin widens as the task gets harder (+1.4 pp fixed → +9.7 pp local-varied → +36.2 pp broad).

Experiment reports: `docs/SINGLE_TASK_20260917.md`, `docs/{XVLA,SMOLVLA}_FIXED_LAYOUT_20260918.md`,
`docs/LOCAL_VARIED_COLLECTION_20260919.md`, `docs/BENCHMARK_STATUS.md`.

---

## Variations tried

- **Datasets**: fixed pilot → local-varied (60 layouts × ±1° jitter) → broad-workspace (F7 envelope × ±1° × ±3 cm).
- **Policies**: SmolVLA (`chunk_size=50, n_action_steps=10`) vs X-VLA (`chunk_size=30, n_action_steps=10`).
- **Failure handling**: no-retry (fixed_layout) → retry-with-rejection-budget (broad_workspace, 100 rejections allowed per 200 admits).
- **Rollout precision**: SmolVLA uses FP32; X-VLA uses FP32 params + BF16 autocast (matches training loss path exactly).

---

