# pi0.5, action expert fine-tuned (VLM frozen), on the sealed P2 dataset (M1, arm A1)

Part of the [VLA inputs programme](../../docs/superpowers/specs/2026-10-30-vla-inputs-programme-design.md); design in the
[M1 spec](../../docs/superpowers/specs/2026-10-30-vla-inputs-m1-pi05-design.md), tasks in the
[M1 plan](../../docs/superpowers/plans/2026-10-30-vla-inputs-m1-pi05.md). Question: on the same 540 sealed P2 episodes and the
same inputs, does pi0.5 with its action expert fine-tuned and the vision-language model frozen, at 5.0 epochs, beat X-VLA
checkpoint 101,520?

**The arm is "pi0.5, action expert fine-tuned (VLM frozen)", not full fine-tuning.** SigLIP and PaliGemma stay at their
`pi05_base` values (bf16); the action expert and the action and time projections train in FP32 with FP32 Adam under bf16
autocast: 693,422,112 of 3,616,757,520 parameters (the expert's unused `lm_head`, 263 M, gets no gradient, so 430 M move). Any
difference from A0 is a difference of model, recipe and this restriction together. Decided by the user on 2026-10-31 from the
`--probe-memory` measurements (docs/BENCHMARK_STATUS.md): full tuning in bf16 loses most updates (5% of bf16 elements change
per update at the peak learning rate), FP32 masters on the CPU take 12 days, this takes about 23 h raw.

**Status: Tasks 1-4 written; decision recorded in `contract.DECISION` (strategy `expert_only`, microbatch 8 of an effective
batch of 8, a snapshot every 2,600 updates, one snapshot kept plus the one being written).** The measured bytes in it come from
the probe. Nothing here has trained until the gates in the plan's Task 5 have passed. The run's output will be
`outputs/train/pi05_props_p2_v1`.

### Disk, and what `--train` re-verifies from the gate outputs

`--train` recomputes the hash of every file in every checkpoint directory of the preflight and resume-proof outputs (weights,
config, marker, and `training_state.pt` on the final one), requires exactly `step_000001` and `step_000002`, compares the two
weights hashes, and re-hashes both `result.json` at the end. It never reads the `resume/` snapshots. Those are therefore
deleted once the resume-proof has passed (the resume-proof itself reads the preflight's), which the user authorised on
2026-10-31; the checkpoints stay.

## Modules

| file | what |
|---|---|
| `contract.py` | binding, windows, statistics, recipe, config per strategy, lr schedule (equal to LeRobot's scheduler), row validation, sampler, claims, gate proofs |
| `assets.py` | pinned weights and tokenizer files |
| `weights.py` | tie-aware loader (never `from_pretrained`); optionally returns the exact FP32 masters |
| `strategy.py` | the three D5 candidates behind one interface: `bf16_all`, `bf16_cpu_master` (FP32 masters, gradients and Adam on the CPU), `expert_only` |
| `runner.py` | `--probe-memory`, `--check`, `--preflight`, `--resume-proof`, `--train [--stop-at T] [--resume]` |

The three strategies share one update (`runner.one_update`: `8 / microbatch` forward and backward passes, loss divided by the
accumulation, FP32-accumulated global-norm clip at 1.0, step). Checkpoints are saved with `save_pretrained` and verified
tensor by tensor against the live policy; a second 3.6 B model does not fit next to the first for a prediction comparison.

## What is bound

`contract.SOURCE_FILES` is an explicit tuple (this README, the package modules and tests, the P2 collection modules the
contract imports, and the installed LeRobot pi05/rtc/processor/optim/dataset sources). `tests/test_fork_hygiene.py` asserts it
equals the repo modules importing the contract loads, plus this package's own files; the evaluation adapter
(`eval_adapter.py`, `eval_tests/`, written for M1 Task 6) is deliberately not listed, so writing it does not invalidate a
training binding.

| item | pin |
|---|---|
| dataset | sealed `outputs/props_p2_data/props_p2_20261013`, fingerprint `b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`, bound exactly as the X-VLA P2 contract binds it (copied, not imported; `tests/test_p2_parity.py` checks the copy against the original on the real dataset) |
| image | `admission-pi`, `sha256:a229f2fbe98f589caaca37068c1ba4a8761d111c6e4855b2360f783c1f290eae`; rollouts later in `robosuite-pi` (`docker/pi05/README.md`) |
| transformers | commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (4.53.3) with tokenizers 0.21.4, hash-checked at build |
| weights | `lerobot/pi05_base` revision `9e55186ad36e66b95cda57bc47818d9e6237ae30`, `model.safetensors` sha256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f` |
| tokenizer | `google/paligemma-3b-pt-224` revision `35e4f46485b4d07967e7e9935bc3786aad50687c`, five files hashed in `assets.py` |

`assets.require_assets()` hashes every file under `outputs/research/pi05_assets/` against the table and rejects any unlisted
file or symlink. The hub's own `policy_preprocessor.json` is pinned but never read: processors are built by
`contract.make_processors` from the local config and the train-only statistics, pointing the installed factory at the local
tokenizer files.

## Recipe (`contract.recipe`)

Native pi0.5: chunk 50, quantile normalisation with train-only q01/q99, 224 x 224 images, lr 2.5e-5, betas (0.9, 0.95), weight
decay 0.01, clip 1.0, warmup 1,000, cosine to 2.5e-6. Three named deviations: effective batch 8, the cosine decay spanning
the whole run, 10 of 50 actions executed per inference. The horizon is derived (`round(5.0 * len(windows()) / 8)`), never
typed; the checkpoint grid is update 1 plus the fractions 0.4, 0.6, 0.7, 0.8, 0.9, 1.0 of it.

Exposure is reported as three numbers (`contract.exposure`): observation starts per epoch (193,440, against X-VLA's
203,040), samples, and supervised action positions (50 per sample, against 30). Chunk 50 excludes 20 more observation
starts per episode than chunk 30, so the two arms are not identical in exposure, only in epochs over each model's complete
windows.

### Native out-of-range behaviour (recorded, not changed)

The installed quantile normaliser does not clip. The state discretiser maps a normalised value below -1 to token **-1** and
saturates values above 1 at **255**; q01 maps to 0 and q99 to 255 (`tests/test_processor.py`). The recipe keeps this native
behaviour (`normalization = quantiles_train_only_q01_q99_no_clipping`); about 2% of training values per dimension lie outside
[q01, q99] by definition.

### Loader caveat

`PI05Policy.from_pretrained(strict=True)` reports `language_model.embed_tokens.weight` as missing (the checkpoint stores the
tied embedding once, under `lm_head`), prints a warning and carries on: every tensor still loads, and any other failure
inside it also only prints. `tests/test_weights_load.py` therefore asserts on key sets and tensor equality, and the runner
must not rely on the exception path.

## Tests

Admission-pi image, repository mounted at `/vla`, the pi05 assets at `/vla/outputs/research/pi05_assets` and the sealed P2 root
at `/vla/outputs/props_p2_data/props_p2_20261013` (both read-only; tests skip without them), `PYTHONPATH=/vla`:

```bash
python -m unittest discover -s experiments/pi05_props_p2/tests -t experiments/pi05_props_p2/tests
```

`test_image.py` also runs in `robosuite-pi`. `test_weights_load.py` needs about 30 GB of RAM. `test_p2_parity.py` reads all
244,080 rows twice.
