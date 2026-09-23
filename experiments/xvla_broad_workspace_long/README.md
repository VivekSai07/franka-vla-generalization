# Broad-workspace X-VLA fine-tune — matched exposure (16700 updates)

Identical to `experiments/xvla_broad_workspace/` except for the training horizon. That run used
5000 updates, which on this dataset is only **0.50 epochs** (40,000 chunks over 79,840 starts),
against 1.67 epochs for the local-varied run — updates were held fixed while the data tripled.
16700 updates restores 1.67 epochs. Motivated by the measured checkpoint curve on the identical
72-trial holdout: 0.0% @3000 → 20.8% @4000 → 68.1% @5000, still accelerating at the end with
val/train arm MAE 1.01–1.06 (no overfitting).

The LR schedule is redefined locally. `xvla_single_task.contract.learning_rate` hard-caps updates
at 5000 and its `make_config` fixes `scheduler_decay_steps=5000`, so a longer run raises at update
5001. That module is bound by four contracts, so it cannot be edited without invalidating three
sealed runs; this package therefore carries its own `learning_rate`/`set_learning_rate`/`make_config`
with the same 500-update warmup and same 1e-5 floor, stretched over the full horizon.

No GPU, full semantic admission, physics, training or rollouts are executed by the implementation
tests.

## Dataset & splits

- Sealed parent `outputs/broad_workspace_data/broad_workspace_20260920` with
  fingerprint `0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85`.
- 200 episodes, train `0:160`, validation `160:200`. Training rows 84480 across
  79840 complete 30-action starts (per-episode last complete start 498).
  Validation 21120 rows are never sampled for optimization or statistics.
- Envelope F7, ±1° home-pose jitter, ±3 cm hover-Z jitter. Two layouts
  (`validation_020`, `validation_032`) were deterministically resampled inside
  the same envelope; both fall in the validation split, so the 160 training
  episodes use their originally planned layouts.
- Fresh original X-VLA base bundle at `.cache/xvla_trial/bundle` (model and
  BART-large tokenizer), byte-verified via `xvla_trial.assets.require_assets`.
- Fresh train-only mean/std/min/max statistics; STATE IDENTITY, ACTION MEAN_STD,
  VISUAL IDENTITY (X-VLA processor defaults). Native chunk 30.

## Training recipe

- Microbatch 4 × accumulation 2, effective batch 8, seed 20260917 (shared with
  the local-varied runs for cross-model reproducibility), **16700** AdamW updates.
  Full-model fine-tune (all 901 named FP32 parameters, ~879M).
- FP32 parameters/gradients/Adam moments; BF16 autocast only during forward.
- 500-update linear warmup to 1e-4 peak, cosine decay to 1e-5 floor at update
  16700. VLM group LR scale 0.1, other groups 1.0; betas (.9, .95), eps 1e-8,
  weight decay 1e-4 (VLM 1e-5), gradient clip 10.
- Stock X-VLA loss; complete unpadded chunks only. Optimizer state persisted
  only on the FINAL checkpoint (`optimizer_storage='final_only'`).

Only the dataset changes relative to `experiments/xvla_local_varied/`; every
hyperparameter is identical so the two runs are directly comparable.

## Stricter provenance than the local-varied contract

`experiments/xvla_local_varied/contract.py` had to waive the recorded-versus-fresh
`provenance()` comparison: commit `2435dca` edited two shared collector helpers
(`experiments/fixed_layout/media.py`, `record.py`) after that dataset was sealed,
so its seal-time snapshot no longer reproduces.

This dataset was sealed after those edits and its recorded snapshot still matches
a fresh `experiments.broad_workspace_data.scenes.provenance()` byte-for-byte, so
`current_binding()` here enforces the comparison instead of skipping it, on top of:

- SEAL fingerprint of the parent;
- `plan.json` via `broad_workspace_data.plan.validate_plan`;
- `semantic.status == 'passed'`, `semantic.training_ready is True`,
  `semantic.automatic_training is False`;
- `semantic.train_ids == list(range(160))`,
  `semantic.validation_ids == list(range(160, 200))`,
  `semantic.statistics_episode_ids == list(range(160))`.

Consequence: `experiments/broad_workspace_data/**/*.py` is frozen for the
lifetime of this run. Editing any of it invalidates the binding mid-run.

## Ordered entry points (main agent only for GPU)

Module `experiments.xvla_broad_workspace.runner` accepts exactly one of:

- `--check`: claims the exclusive `_admission` directory, performs one fresh
  `require_broad_workspace_pass`, actual data/statistics and CPU micro-4 processor
  checks without constructing the backbone. Persists the semantic report and
  bindings in that directory, never in bound source.
- `--preflight`: requires that CPU proof; fresh base, two updates, checkpoints
  1/2, exact FP32 weights/Adam and zero prediction-error reload, no probes/rollouts.
- `--train`: requires byte-bound scratch/CPU/stats proofs; starts **fresh base**,
  not scratch. Checkpoints 1/2000/4000/6000/8000/10000/12000/14000/16700;
  `training_state.pt` (optimizer state) only on 16700. No automatic evaluation; evaluation
  lives in
  a separate package.

Output roots beneath `outputs/train/`:

- `xvla_franka_stacking_broadworkspace_long_v1_20260923_admission/`
- `xvla_franka_stacking_broadworkspace_long_v1_20260923_preflight/`
- `xvla_franka_stacking_broadworkspace_long_v1_20260923/`

Each root is exclusively claimed via `run_claim.json`. Existing directories are
rejected unless they are actual empty bind mount points. No overwrite, resume
or retry paths exist.
