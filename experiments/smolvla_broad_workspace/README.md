# Broad-workspace SmolVLA fine-tune

Approved recipe: the same SmolVLA hyperparameters as the local-varied run
(`experiments/smolvla_local_varied/`), retargeted at the sealed 200-episode
broad-workspace dataset. No GPU, full semantic admission, actual physics,
training or rollouts are executed by the implementation tests.

## Dataset & splits

- Sealed parent `outputs/broad_workspace_data/broad_workspace_20260920` with
  fingerprint `0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85`.
- 200 episodes, train `0:160`, validation `160:200`. Training rows 84480 across
  76640 complete 50-action chunks (per-episode last complete start 478).
  Validation 21120 rows are never sampled for optimization or statistics.
- Envelope F7, ±1° home-pose jitter, ±3 cm hover-Z jitter. Two layouts
  (`validation_020`, `validation_032`) were deterministically resampled inside
  the same envelope; both fall in the validation split, so the 160 training
  episodes use their originally planned layouts.
- Fresh original SmolVLA revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
  with VLM/tokenizer `7b375e1b73b11138ff12fe22c8f2822d8fe03467` from the
  offline `.cache/benchmark_models` bundle (42 files, digest pinned).
- Fresh train-only mean/std/min/max/q01/q99 statistics; STATE and ACTION
  MEAN_STD, VISUAL IDENTITY (SmolVLA processor defaults). Native chunk 50.

## Training recipe

- micro 8 x accumulation 4, seed 20260917, 5000 AdamW updates, 160000 sampled
  chunks. Frozen VLM. FP32 trainable expert/state projection promoted BEFORE
  Adam construction. BF16 training autocast only. 450 046 176 total /
  99 880 992 trainable parameters.
- 500 update linear warmup to 1e-4 peak, cosine decay to 1e-5 floor at update
  5000. Betas (.9, .95), eps 1e-8, weight decay 1e-10, gradient clip 10.
- Chunk size 50, action steps 10, state 9, action 8. Two 512x512 RGB cameras.
- Stock LeRobot flow-matching loss; complete unpadded chunks only.

Only the dataset changes relative to `experiments/smolvla_local_varied/`; every
hyperparameter is identical so the two runs are directly comparable, and the
seed matches `experiments/xvla_broad_workspace/` for cross-model parity.

## Checkpoint schedule and naming

- Directories under `outputs/train/smolvla_franka_stacking_broadworkspace_v1_20260921`.
- Preflight `[1, 2]`; full run `[1, 1000, 2000, 3000, 4000, 5000]`.
- Naming is exactly `checkpoints/step_{update:06d}` (six-digit padded, `step_`
  prefix). Any legacy plain `{update:06d}` directory is rejected as a
  regression. File names inside each step directory are unchanged from LeRobot:
  `model.safetensors`, `config.json`, `training_state.pt`,
  `policy_preprocessor.json`, `policy_postprocessor.json`,
  `checkpoint_result.json`. Preflight also writes `adam_reload.json` with
  exact FP32 Adam roundtrip evidence.
- Every marker must satisfy `reload_max_action_error <= 1e-5` and exact FP32
  reload. Scratch additionally restores the saved Adam state to a fresh CPU
  optimizer with bit-exact tensor equality.

## Isolation

- No monkeypatching or in-place editing of the fixed-layout SmolVLA package,
  the broad-workspace dataset package or any installed lerobot/transformers code.
- Package imports from `training/smolvla_pilot.py`, `validation/precision.py`
  and `experiments.single_task.contract.learning_rate` are pure helpers that
  do not know about this dataset. The local `save_checkpoint` mirrors the
  pilot saver but writes to `step_{update:06d}` directly instead of editing
  the frozen helper.
- `experiments.broad_workspace_data.validate.require_broad_workspace_pass` is the
  only full-semantic audit call, invoked exactly once in `--check`. Later
  phases rehash the seal, plan, validation report, source resets, snapshots
  and cache byte inventory without re-running the audit.
- `current_binding()` compares the recorded `provenance` snapshot against a
  fresh one, so `experiments/broad_workspace_data/**/*.py` is frozen for the
  lifetime of this run.

## Phase entry points

Use module `experiments.smolvla_broad_workspace.runner` with exactly one of
`--check`, `--preflight` or `--train`. No CLI override, resume, batch tuning
or fallback. `--train` never launches evaluation; policy comparison is a
separate protocol.

Main-agent launch only after independent review: pinned image
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`,
canonical repository read-only, network `none`, no credentials. Bind only the
phase's own empty output root read/write. Existing empty bind mount is
permitted only via exclusive `run_claim.json`; any content, symlink or prior
claim fails closed. Paths are
`outputs/train/smolvla_franka_stacking_broadworkspace_v1_20260921_admission`,
`..._preflight` and `...` respectively.

Before either GPU phase require 30 GiB disk and 12 GiB GPU free, BF16-capable
CUDA. Data, cache and repository are read-only; all failed and successful
artifacts are retained.
