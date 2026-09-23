# Results

All benchmark numbers. Protocols differ in difficulty and are **not** comparable across
rows unless stated. Chronological narrative lives in [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md);
failures and lessons in [FAILURES.md](FAILURES.md).

## Current status (2026-09-22)

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

**Open: 68.1% is likely undertrained, not a ceiling.** X-VLA saw only **0.50 epochs** on the broad dataset
(40,000 chunks over 79,840 starts) versus 1.67 epochs on local-varied — updates were held at 5000 while the
data tripled. The checkpoint curve on the identical holdout confirms training was still improving when it
stopped:

| update | broad_holdout | train / val arm MAE | val/train |
|---|---|---|---|
| 3000 | 0.0% (0/72) | 0.05826 / 0.06060 | 1.040 |
| 4000 | 20.8% (15/72) | 0.02100 / 0.02122 | 1.010 |
| 5000 | **68.1%** (49/72) | 0.01655 / 0.01748 | 1.056 |

Success accelerates while MAE improvement decelerates — stacking is a threshold task, so late precision
gains convert into large success jumps. No overfitting at any checkpoint (val/train 1.01–1.06). Caveat: part
of the 4000→5000 jump is the cosine anneal to its 1e-5 floor, so this curve cannot separate "needs more
exposure" from "needed the anneal", and 68% should not be linearly extrapolated. Next step is a matched-exposure
run at ~16,700 updates; see [TODO](TODO.md) Phase A.

**Earlier local-varied results** (narrow ±1 cm protocol, not comparable to the above — different task difficulty):

| Metric | SmolVLA | X-VLA |
|---|---|---|
| varied_holdout success rate | 88.9% (64/72) | **98.6% (71/72)** |
| fixed_parity success rate | 100% (20/20) | 100% (20/20) |
| offline val arm MAE (rad) | 0.01133 | 0.00996 |

Open question: whether the 200-episode collection actually improved generalisation. A cross-evaluation of the
**local-varied** X-VLA on the identical broad holdout is built and queued; without it 68.1% has no reference point.

---

## Datasets & training runs

| Package | Purpose | Episodes | Status |
|---|---|---|---|
| `experiments/fixed_layout/` | Single-geometry pilot (source reset ep 7) | 24 | sealed |
| `experiments/local_varied_data/` | 60 layouts, ±1° pose jitter | 60 (48 train / 12 val) | sealed |
| `experiments/broad_workspace_data/` | F7 envelope, retries permitted | 200 (160 train / 40 val) | **sealed** |
| `experiments/{smolvla,xvla}_local_varied/` | Local-varied finetunes | — | trained + evaluated |
| `experiments/{smolvla,xvla}_broad_workspace/` | Broad-workspace finetunes | — | trained + evaluated |
| `experiments/xvla_localvaried_broad_eval/` | Cross-eval: old checkpoint, new holdout | — | built, queued |

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

