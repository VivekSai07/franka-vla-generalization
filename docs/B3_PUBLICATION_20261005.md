# B3 checkpoints — staged for Windows and Hugging Face — 2026-10-05

What the user chose to publish (2026-10-05): X-VLA steps **124,800** (best of four) and **156,000**
(final) from `outputs/train/xvla_franka_stacking_b3_robosuite_v2`. The other four checkpoints and
every superseded run stay on the RTX machine; nothing was deleted.

## Staged

`outputs/downloads_20261005/b3_robosuite/` — hardlinks only, zero extra disk.

- `hf_model_repo/` is the model repo as it should appear on Hugging Face: step 124,800 at the root
  with the model card (`README.md`), step 156,000 under `checkpoints/step_156000/`, the screen and
  report `result.json` for both checkpoints, the expert feasibility result, and the training record
  (`result.json`, `training_stats.json`, `metrics.jsonl`, initial config). About 6.7 GB.
- Not included: `training_state.pt` (optimizer state), evaluation videos and traces, the dataset.
- The staging `README.md` holds the Windows transfer, verify and upload commands.

| checkpoint | `model.safetensors` sha256 (verified on RTX against `checkpoint_result.json`) |
|---|---|
| 124,800 | `e0d1eae45082e336e5ef14e7ecc82a873f0d79006058620c227c260b305edb99` |
| 156,000 | `04e7fdf45f9c83d39f9692ca787c34781abdb0dc3cf524b0f8a4a90c2b0c7a7e` |

## Repo and card

- Suggested repo, following the account's convention: `VivekSai07/xvla-franka-b3-robosuite-124800`
  (model), created **private**; public is a separate decision.
- Licence Apache-2.0, base `lerobot/xvla-base` (same evidence as
  [B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md](B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md)).
- Card numbers come from [RESULTS.md](RESULTS.md) and the report addendum: 141/160 = 88.1% for
  step 124,800, quoted as about 85% overall and about 75% with distractors; 105/160 = 65.6% for
  156,000; held-out instruction pairs as good as trained ones; robosuite 1.5.2 / MuJoCo 3.4.0.
- Portability caveat stated in the card: `policy_preprocessor.json` names its tokenizer by its
  training-time absolute path.

## Status

Staged and verified on the RTX side. Transfer, Windows verification and upload are the user's;
record them here when done.
