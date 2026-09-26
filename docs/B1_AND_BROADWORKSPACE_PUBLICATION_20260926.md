# B1 + broad-workspace publication — 2026-09-26

## Why this exists

`docs/LOCAL_VARIED_DATASET_PUBLICATION_20260921.md` recorded that direct upload from this
machine is blocked: the corporate proxy returns `407 Proxy Authentication Required` on the HF
LFS storage origin (`hf-hub-lfs-*.s3-accelerate.amazonaws.com`), even though authentication and
token permissions check out fine. That blocker is environmental, not fixable from inside this
repo's Docker sandbox.

**New workflow, agreed with the user:** copy the reviewed payload from this machine to a Windows
machine (no proxy there), and have a Claude Code agent running on Windows do the actual
`huggingface_hub` upload from a local, unpacked directory. This document is the input to that
agent — it names exactly what to copy, what each thing is, and what a correct dataset/model card
says. Nothing in this document performs a network mutation; no repo was created, no file uploaded.

## What to publish

Two results are publication-worthy right now (see [RESULTS.md](RESULTS.md) for full numbers):

1. **`b1_language_data`** (dataset) + **`xvla_franka_stacking_b1_language_v1_20260925`** (model) —
   94.4% (151/160) on a held-out language-grounding test; the headline result.
2. **`broad_workspace_data`** (dataset) + **`xvla_franka_stacking_broadworkspace_long_v1_20260923`**
   (model) — 86.1% (62/72) on the broad spatial holdout; the prior headline, referenced by B1's
   own README as the exposure baseline.

`local_varied_data` (60 episodes, 2.8% on the broad holdout) is **not** recommended for
publication as a capability result — it demonstrates a generalization failure, not a capability.
Publish it later only if you specifically want a "what doesn't work" dataset for comparison.

## Exact payload — datasets

The LeRobot v3.0 payload is the `dataset/` subdirectory, **not** the collection root. The root
also contains `attempts/`, `rejected/`, `resets/` — raw per-episode physics/replay audit evidence,
15-17 GB each, needed only to re-verify the sealed admission, not to reuse the data. Do not
publish those.

| dataset | copy this exact path | size | + provenance files (small, optional but recommended) |
|---|---|---|---|
| B1 | `outputs/b1_language_data/b1_language_20260925/dataset/` | ~1.1 GB | `plan.json` (468K), `provenance.json` (20K), `seal.json` (4K), `validation.json` (4.3M) from the parent dir |
| broad-workspace | `outputs/broad_workspace_data/broad_workspace_20260920/dataset/` | ~1.1 GB | same four files from that parent dir |

The provenance files let anyone re-derive the seal fingerprint and confirm the published data is
bit-for-bit what training actually used — worth the extra ~5 MB.

**Schema** (identical for both): LeRobot v3.0, 30 fps, 9-dim state (joint + gripper), 8-dim action
(7 joint + gripper), two 512×512 camera streams (`observation.images.image`,
`observation.images.image2`), AV1-encoded. B1 additionally carries **two** task instructions
(`meta/tasks.parquet`); broad-workspace carries one.

- B1: 200 episodes, splits `train=[0,160)` `validation=[160,200)`, seal
  `b152cba2e035653d7a800d3a5e5112005e85080300e0616483d9fe74837fdf22`.
- Broad-workspace: 200 episodes, same split shape, seal
  `0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85`.

## Exact payload — models

Each checkpoint directory has this shape:

```
step_NNNNNN/
  model.safetensors                                    # ~3.3 GB — the weights, always include
  config.json                                           # small — X-VLA/Florence2 config
  policy_preprocessor.json + *_normalizer_processor.safetensors    # small
  policy_postprocessor.json + *_unnormalizer_processor.safetensors # small
  checkpoint_result.json                                # small — sha256 of model.safetensors, reload proof
  training_state.pt                                     # ~7 GB, FINAL checkpoint only — optimizer/RNG state
```

**Do not publish `training_state.pt`.** It exists only to resume training (which nobody will do
from a downloaded public checkpoint) and roughly doubles the transfer size for zero inference
value. `checkpoint_result.json` already proves the published `model.safetensors` reloads
FP32-exact with zero prediction error — that proof travels with the small files.

| model | which checkpoint(s) | path | size (inference files only) |
|---|---|---|---|
| B1 (recommended) | `step_033400` only (the winner) | `outputs/train/xvla_franka_stacking_b1_language_v1_20260925/checkpoints/step_033400/` | ~3.3 GB |
| B1 full ablation | + `step_000001`, `step_008000`, `step_016700`, `step_025000` | same dir, other `step_*` folders | +~13 GB (16.3 GB total) |
| broad-workspace (recommended) | `step_016700` only (the winner) | `outputs/train/xvla_franka_stacking_broadworkspace_long_v1_20260923/checkpoints/step_016700/` | ~3.3 GB |
| broad-workspace full ablation | + `step_000001`, `step_002000`, `step_004000`, `step_006000`, `step_008000`, `step_010000`, `step_012000`, `step_014000` | same dir | +~26 GB (29.3 GB total) |

The full-ablation tiers exist because the checkpoint-selection curves are themselves findings
(B1: 55%→85%→**100%** screening across 16700/25000/33400; broad-workspace: the **0/72 collapse at
step 8000** mid-training, see [FAILURES.md](FAILURES.md) on offline-metric divergence). **Both
are now staged** — see the updated census below.

## Exact payload — evaluation evidence (recommended, small)

These are the rollout videos/traces/episode.json backing the published numbers. Cheap relative to
the checkpoints — worth including as supplementary evidence.

| eval | path | size |
|---|---|---|
| B1 report (the 94.4% result) | `outputs/train/xvla_franka_stacking_b1_language_v1_20260925/evaluation_report_step_033400/` | 194 MB |
| B1 screening (all 3 candidates) | `evaluation_screen_step_{016700,025000,033400}/` (same run dir) | 77+57+46 MB |
| broad-workspace (the 86.1% result) | `outputs/train/xvla_franka_stacking_broadworkspace_long_v1_20260923/evaluation/` | 121 MB |
| broad-workspace curve points | `evaluation_step_{004000,008000,012000}/` (same run dir) | 194+192+128 MB |

## Recommended minimal-tier total

Datasets (2×1.1 GB + provenance) + winning checkpoints (2×3.3 GB) + report/headline eval evidence
(194+121 MB) ≈ **~10 GB**. Comfortably transferable; does not require the disk cleanup this
machine would otherwise need.

## Full-ablation tier total

Minimal tier + all 4 remaining B1 checkpoints (~13 GB) + all 8 remaining broad-workspace
checkpoints (~26 GB) + the intermediate eval evidence for both curves (~0.9 GB) ≈ **~50 GB**.
Still zero extra disk on this machine (hardlinks), but the Windows-side transfer and eventual HF
upload both scale with this number — choose the tier before starting a multi-hour transfer.

## Staging directory (already built, both tiers)

`outputs/downloads_20260926/` now contains **both** tiers, hardlinked (confirmed zero extra
disk before/after each addition):

- Minimal tier (the six items above): datasets, both winning checkpoints, both headline eval
  directories. **Already downloaded by the user as of 2026-09-26.**
- Full-ablation tier (added the same day): every other checkpoint for both runs, plus every
  intermediate/screening eval directory. Directory names: `xvla_b1_language_step_{000001,
  008000,016700,025000}/`, `xvla_broadworkspace_step_{000001,002000,004000,006000,008000,
  010000,012000,014000}/`, `b1_evaluation_screen_step_{016700,025000,033400}/`,
  `b1_evaluation_smoke_step_016700/`, `broadworkspace_evaluation_step_{004000,008000,012000}/`,
  `broadworkspace_evaluation_placement_modes/`.

See the `README.md` inside that directory for the exact `rsync`/`scp` command and a post-download
integrity check (re-hash `model.safetensors` against `checkpoint_result.json`, and re-derive each
dataset's seal fingerprint against `seal.json`).

**Nothing on this machine has been deleted, moved, or modified.** The staging directory only adds
new hardlinks; it does not touch `outputs/train/` or `outputs/*_data/`.

## The 2026-09-20 bundle (older results, re-verified 2026-09-26)

`outputs/downloads_20260920/` predates this document — fixed-layout SmolVLA + X-VLA portable
baselines (100% at base pose) and the local-varied SmolVLA fine-tune (88.9% on its own narrow
holdout, 100% fixed-parity; **not** the same protocol as the 2.8%/72 broad-holdout number for
local-varied in RESULTS.md — different tests of different things, not a contradiction). Its own
`README.md` has the full breakdown.

Re-verified 2026-09-26: all hardlinks still resolve, sizes still match, all three original
source directories still present. Two datasets were missing from that bundle and have now been
added: `local_varied_dataset_60ep/` (327 MB, seal `ae8828ea...25dcf52c`) and
`fixed_layout_dataset_24ep/` (131 MB, seal `403b8ec2...8d8fba5`, shared by both fixed-layout
baselines).

That bundle uses an older, more elaborate packaging convention (`publication/publish.py`'s
portable-package builder — ships `LICENSE`/`NOTICE`/`load_baseline.py`, and keeps
`training_state.pt` per checkpoint rather than stripping it). Both conventions are valid; neither
needs to be rebuilt to match the other. These four results are lower priority than B1/broad-
workspace (superseded Tier 2-4 baselines, not headline results) — publish them only if you
specifically want the full baseline history alongside the two headline results.

## Licensing

All bases fine-tuned here are Apache-2.0, per evidence already gathered in
`outputs/publication/baselines_20260918/public-license-evidence/`:

| component | source | revision | license |
|---|---|---|---|
| X-VLA base | `lerobot/xvla-base` | `cdb7964e4fe842935d671bfab5a5ebe00a96648c` | Apache-2.0 |
| Florence2/BART backbone | `facebook/bart-large` | `cb48c1365bd826bd521f650dc2e0940aee54720c` | Apache-2.0 |

Recommendation: publish fine-tuned checkpoints and derived datasets under **Apache-2.0**, with a
model-card line attributing `lerobot/xvla-base` as the base. Do not invent a license if the
Windows-side agent finds anything inconsistent — stop and ask.

## Repo naming (matches existing account convention)

Prior repos on this account used `VivekSai07/franka-panda-cube-stacking-<topic>`. Suggested:

- `VivekSai07/franka-panda-cube-stacking-b1-language-200` (dataset)
- `VivekSai07/franka-panda-cube-stacking-broad-workspace-200` (dataset)
- `VivekSai07/xvla-franka-b1-language-33400` (model)
- `VivekSai07/xvla-franka-broadworkspace-16700` (model)

## What the Windows Claude Code agent must NOT do

- Never read, transfer, or embed `.env`, any `*token*`, `*credential*`, or `HF_TOKEN`/`WANDB_API_KEY`
  value. None of those files are part of this payload.
- Never upload `training_state.pt`, `attempts/`, `rejected/`, or `resets/`.
- Default to **private** repo visibility first; flipping to public is a separate, explicit,
  reversible decision the user makes after reviewing the rendered card.
- Verify local file hashes against `checkpoint_result.json` / re-derived `seal.json` fingerprints
  **before** upload, not after — catches a corrupted transfer before it becomes a public artifact.
- Do not claim the model "generalizes" or is "production-ready" in the card. State the measured
  numbers and their protocol only (see RESULTS.md wording) — this repo's convention throughout is
  numbers-with-protocol, not marketing language.

## Model/dataset card content (pull directly from RESULTS.md, do not re-derive)

For B1: the 151/160=94.4% headline, the familiar-vs-swapped contrast table (93.8% vs 95.0%,
Fisher p=1.0 — the load-bearing evidence against a positional shortcut), the per-cell table, and
the checkpoint-screening table. For broad-workspace: the 86.1% headline and the matched-exposure
comparison to the 68.1% short run. Copy the tables verbatim from
[RESULTS.md](RESULTS.md) — they are already reviewed and are the canonical numbers for this repo.
