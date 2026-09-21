# Local-varied X-VLA fine-tune (Task 1 implementation only)

Approved recipe: same X-VLA hyperparameters as the fixed-layout baseline
(`experiments/xvla_fixed_layout/`), retargeted at the newly sealed 60-episode
broad-workspace dataset. No GPU, full semantic admission, physics, training or
rollouts are executed by the implementation tests. Independent review and
main-agent phase gates remain.

## Dataset & splits

- Sealed parent `outputs/broad_workspace_data/broad_workspace_20260921` with
  fingerprint `f8b10a498798217a63564416ab6561e25bd7e9c87e4fdddcf42f288cdb9e9a28`.
- 60 episodes, train `0:48`, validation `48:60`. Training rows 25344 across
  23952 complete 30-action starts (per-episode last complete start 498).
  Validation 6336 rows are never sampled for optimization or statistics.
- Fresh original X-VLA base bundle at `.cache/xvla_trial/bundle` (model and
  BART-large tokenizer), byte-verified via `xvla_trial.assets.require_assets`.
- Fresh train-only mean/std/min/max statistics; STATE IDENTITY, ACTION MEAN_STD,
  VISUAL IDENTITY (X-VLA processor defaults). Native chunk 30.

## Training recipe

- Microbatch 4 × accumulation 2, effective batch 8, seed 20260917 (shared with
  `smolvla_broad_workspace` for cross-model reproducibility), 5000 AdamW updates.
  Full-model fine-tune (all 901 named FP32 parameters, ~879M).
- FP32 parameters/gradients/Adam moments; BF16 autocast only during forward.
- 500-update linear warmup to 1e-4 peak, cosine decay to 1e-5 floor at update
  5000. VLM group LR scale 0.1, other groups 1.0; betas (.9, .95), eps 1e-8,
  weight decay 1e-4 (VLM 1e-5), gradient clip 10.
- Stock X-VLA loss; complete unpadded chunks only. Optimizer state persisted
  only on the FINAL checkpoint (`optimizer_storage='final_only'`).

## Semantic shift vs. `smolvla_broad_workspace.contract` (2026-09-20 surgery)

The sealed parent's `validation.json` recorded a `provenance` snapshot at seal
time that included a pilot-manifest hash reference. That referenced manifest is
no longer present, so today `experiments.broad_workspace_data.scenes.provenance()`
returns a snapshot that no longer matches the recorded one byte-for-byte, and
`smolvla_broad_workspace.contract.current_binding()` fails with `"Local-varied
provenance/semantic proof changed"`.

`xvla_broad_workspace.contract.current_binding()` deliberately does NOT compare
`semantic.get('provenance')` against a fresh `provenance()`. It still verifies:

- SEAL fingerprint of the parent (the seal covers data files, not the recorded
  snapshot dict, so it still matches);
- `plan.json` via `broad_workspace_data.plan.validate_plan`;
- `semantic.status == 'passed'`, `semantic.training_ready is True`,
  `semantic.automatic_training is False`;
- `semantic.train_ids == list(range(48))`,
  `semantic.validation_ids == list(range(48, 60))`,
  `semantic.statistics_episode_ids == list(range(48))`.

It then records the FRESH `provenance()` snapshot as `broad_workspace_provenance`
in the returned binding, so future audits (e.g. `admission()` re-comparison)
compare against this new binding rather than the stale seal-time snapshot. This
is the ONLY semantic difference from `smolvla_broad_workspace.contract`.

## Ordered entry points (main agent only for GPU)

Module `experiments.xvla_broad_workspace.runner` accepts exactly one of:

- `--check`: claims the exclusive `_admission` directory, performs one fresh
  `require_broad_workspace_pass`, actual data/statistics and CPU micro-4 processor
  checks without constructing the backbone. Persists the semantic report and
  bindings in that directory, never in bound source.
- `--preflight`: requires that CPU proof; fresh base, two updates, checkpoints
  1/2, exact FP32 weights/Adam and zero prediction-error reload, no probes/rollouts.
- `--train`: requires byte-bound scratch/CPU/stats proofs; starts **fresh base**,
  not scratch. Checkpoints 1/1000/2000/3000/4000/5000; `training_state.pt`
  (optimizer state) only on 5000. No automatic evaluation; evaluation lives in
  a separate package.

Output roots beneath `outputs/train/`:

- `xvla_franka_stacking_broadworkspace_v1_20260921_admission/`
- `xvla_franka_stacking_broadworkspace_v1_20260921_preflight/`
- `xvla_franka_stacking_broadworkspace_v1_20260921/`

Each root is exclusively claimed via `run_claim.json`. Existing directories are
rejected unless they are actual empty bind mount points. No overwrite, resume
or retry paths exist.
