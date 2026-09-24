# Results

All benchmark numbers. Protocols differ in difficulty and are **not** comparable across
rows unless stated. Chronological narrative lives in [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md);
failures and lessons in [FAILURES.md](FAILURES.md).

## Current status (2026-09-24)

**Headline — matched exposure took X-VLA from 68.1% to 86.1% on the broad holdout.** The 0.50-epoch
run was undertrained, as suspected. Retraining on the identical sealed data at 1.67 epochs
(16,700 updates) scored **62/72**, a significant gain (Fisher exact p = 0.0165). It still fails the
65/72 gate, by three trials.

| X-VLA on `broad_workspace` | updates | epochs | broad_holdout | 95% CI |
|---|---|---|---|---|
| short run | 5,000 | 0.50 | 49/72 = 68.1% | [56.6, 77.7] |
| **matched-exposure run** | **16,700** | **1.67** | **62/72 = 86.1%** | **[76.3, 92.3]** |

`fixed_parity` stayed at 20/20. The gate has **not** been lowered to match the outcome.

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
| final xy error | mean 0.80 cm, max 1.96 cm | mean **4.51 cm**, sd **0.20 cm** |

That standard deviation is 2 mm across ten independent trials, and 4.5 cm is one cube width. The cube
is set down *beside* the target, touching it, instead of on top. In 9 of 10 it lands at lower x —
toward the robot base, i.e. stopping one cube-width short. The distribution is bimodal: under 2 cm,
or exactly 4.5 cm, with nothing between.

The obvious explanation — unreachable far layouts — is **refuted and inverted**: failing layouts
average 0.424 m reach, fully-solved ones 0.472 m. The far layouts are solved. This is not kinematics.

Decomposed about the robot base, the systematic component is **tangential** — a bearing error, not a
reach error:

| | n | tangential | radial |
|---|---|---|---|
| successes | 62 | +0.06 cm (sd 0.68) | −0.17 cm (sd 0.57) |
| failures | 10 | **+3.73 cm (sd 0.41)** | −1.81 cm (sd 1.73) |

+3.73 cm at ~0.4 m radius is a ~5° azimuth error, consistently counter-clockwise. The radial
component is noise. The two populations are separated by a **1.11 cm gap containing zero trials**,
so this is a discrete mode, not a precision gradient.

**Cause: action-sampling multimodality.** 6 of the 8 failing layouts are *split decisions* — same
scene, same weights, different sampling seed, opposite outcome — and failures divide 6/4 across the
two seeds, so there is no bad seed. `modeling_xvla.py:252` draws a fresh `torch.randn` per chunk and
integrates over `num_denoising_steps = 10`; the action distribution at placement has two modes and
the rollout draws one of them.

Coverage density was tested as an alternative and **refuted**: 4-D nearest-neighbour distance to the
training layouts is identical for both groups (0.0348 m), with rank correlation τ = 0.000 between
sparsity and failure. The two sparsest holdout layouts are both 2/2 solved.

**Therefore this gap is not closable by data collection.** Remedies are inference-time; see the
design spec at
[docs/superpowers/specs/2026-09-24-placement-offset-design.md](superpowers/specs/2026-09-24-placement-offset-design.md).

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

