# M1 — pi0.5 on the P2 dataset, against X-VLA — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). No code exists. The checks below were read-only.

## Question

On the same 540 sealed P2 episodes, with the same inputs (two 512 × 512 cameras, the 9-value state, P2's instructions), does
pi0.5 fine-tuned with its own recipe at the same exposure (5.0 epochs) beat X-VLA checkpoint 101,520 (A0) on M0's comparison
tier?

## What is established (read 2026-10-30, pinned admission image, LeRobot 0.4.4)

| fact | where |
|---|---|
| LeRobot 0.4.4 ships `pi0`, `pi05`, `pi0_fast`, `groot`, `xvla` | `lerobot/policies/` in the image |
| `PI05Config` defaults: PaliGemma `gemma_2b` + action expert `gemma_300m`, chunk 50, `n_action_steps` 50, `max_state_dim` 32, images 224 × 224, quantile normalisation for state and action, AdamW lr 2.5e-5, betas (0.9, 0.95), wd 0.01, clip 1.0, warmup 1,000, cosine to 2.5e-6, `gradient_checkpointing` off, `freeze_vision_encoder` and `train_expert_only` off | `configuration_pi05.py` |
| the state is **normalised to [−1, 1] by quantiles, discretised into 256 bins and written into the prompt as text** (`"Task: …, State: …;\nAction: "`); tokenizer `google/paligemma-3b-pt-224`, max 200 tokens | `processor_pi05.py` lines 48–90, 143–150 |
| **pi0.5 cannot be constructed in the pinned image:** `PI05Pytorch.__init__` requires `transformers.models.siglip.check` (LeRobot's patched transformers); the image has transformers 4.57.6 and the import fails (`ImportError: cannot import name 'check'`) | `modeling_pi05.py` lines 576–583, probe 2026-10-30 |
| the PaliGemma tokenizer repository is gated on Hugging Face (licence acceptance); containers run with `--network none` | — |
| X-VLA (A0) recipe: chunk 30, 10 actions executed per inference, effective batch 8, 126,900 updates = 5.0 epochs over 203,040 windows, FP32 parameters and Adam with bf16 autocast, 0.524 s/update raw on the RTX 6000 Ada | `experiments/xvla_props_p2/contract.py`, P2 training notes |
| GPU: one RTX 6000 Ada, 48 GiB, shared with other users | `nvidia-smi` |

### Measured on the RTX machine, 2026-10-30 (throwaway containers; no image built, nothing installed into an image)

| fact | how |
|---|---|
| LeRobot 0.4.4's `pi` extra is `transformers @ git+…/transformers.git@fix/lerobot_openpi` plus `scipy>=1.10.1,<1.15`; the extra is declared **in conflict with** `xvla` and `transformers-dep` | `pyproject.toml` at tag v0.4.4 |
| that branch is commit **`dcddb970176382c0fcf4521b0c0e6fc15894dfe0`**, transformers **4.53.3**, which requires **`tokenizers>=0.21,<0.22`**; the pinned image has tokenizers **0.22.2**, so transformers alone is not a consistent install | `git ls-remote`, `setup.py` at that commit, `pip list` in the image |
| with transformers@`dcddb97` + tokenizers **0.21.4** on top of the admission image: `check_whether_transformers_replace_is_installed_correctly()` is **True** and `PI05Policy` constructs (meta device). Neither pi0 nor pi05 imports scipy directly | probe with `pip install --no-deps --target`; `grep` |
| **pi0.5 has 3,616,757,520 parameters:** SigLIP vision 412,442,352; PaliGemma language model 2,508,531,712; action expert 691,256,320; other 4,527,136 | `PI05Policy` on the meta device, P2 features (state 9, two cameras, action 8) |
| `lerobot/pi05_base` is public (not gated); `model.safetensors` 14,467,165,872 bytes (FP32), LFS sha256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`, identical in every revision checked | Hugging Face API |
| the latest `pi05_base` revision (`22c30e5b`, 2026-10-06) saves a preprocessor with a `relative_actions_processor` step that **LeRobot 0.4.4 does not have**; revision **`9e55186ad36e66b95cda57bc47818d9e6237ae30`** (2026-01-22, before LeRobot 0.4.4's release on 2026-02-27) has only steps 0.4.4 knows, and the same weights | Hugging Face API, `grep` in the installed package |
| `google/paligemma-3b-pt-224` is gated with **manual** approval; it ships `tokenizer.json` (so `sentencepiece`, absent from the image, should not be needed) | Hugging Face API |
| the image has **no** `peft`, `bitsandbytes`, `deepspeed`, `scipy` or `sentencepiece` | `pip list` |
| quantile normalisation does not clip (`normalize_processor.py` sha256 `5b610f5b…`); the state discretiser (`processor_pi05.py` sha256 `10bfff02…`) maps values below −1 to token **−1** and saturates values above 1 at **255** | installed source |

## Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | **Two new images**, each the existing image plus exactly **transformers@`dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (4.53.3) and tokenizers 0.21.4**, installed with `--no-deps`: `admission-pi` (training, offline tests) and `robosuite-pi` (rollouts). Pinned by digest. A test asserts `pip freeze` differs from the parent image in those two lines alone and `pip check` passes; scipy is added only if a pi0.5 code path is shown to need it (recorded). X-VLA is never run in these images (the extras conflict) | the existing images must not change (every sealed run is bound to them); one new variable at the software level |
| D2 | Base weights `lerobot/pi05_base` at revision **`9e55186ad36e66b95cda57bc47818d9e6237ae30`** and the tokenizer `google/paligemma-3b-pt-224` (tokenizer files only) downloaded on the host through the proxy into `outputs/research/pi05_assets/`, sha256 of every file pinned in the contract (weights: `0eb11ca9…9b0f`); **processors are always built locally from the config and P2 statistics, never loaded from the hub**; the Hugging Face token is never written to the repository or a container. **Request PaliGemma access now: approval is manual** | offline containers; reproducibility; the latest hub preprocessor does not load in 0.4.4 |
| D3 | **pi0.5's own recipe** (chunk 50, quantile normalisation with train-only q01/q99, state in the prompt, 224 × 224 with padding, its LR, warmup, decay, betas, weight decay, clip), except: **effective batch 8**, and the decay spans the whole run | user-approved "native recipe"; batch 8 because of memory (D5) and so the checkpoint grid aligns with X-VLA's by schedule fraction |
| D4 | **Exposure 5.0 epochs:** with chunk 50 a training episode has 452 − 49 = 403 windows, 480 × 403 = 193,440; 5.0 epochs = 967,200 samples = **120,900 updates** at batch 8. Checkpoints at the same fractions as X-VLA's grid (0.4, 0.6, 0.7, 0.8, 0.9, 1.0): 48,360, 72,540, 84,630, 96,720, 108,810, 120,900, plus update 1 | exposure, not update count, is what the repo holds constant |
| D5 | **Memory strategy decided by measurement, before training, and recorded as part of the arm's name.** With the measured 3.617 B parameters, before activations: FP32 parameters + gradients + Adam (16 B/param) **57.9 GB**, does not fit; bf16 parameters and gradients with FP32 moments (12 B) **43.4 GB**, no realistic headroom; vision encoder frozen with FP32 Adam for the rest **~52 GB**, does not fit; all-bf16 parameters, gradients and moments (8 B) **28.9 GB**; `train_expert_only` (0.70 B trainable, FP32 Adam; frozen 2.92 B in bf16) **~17 GB**. FP32 master weights for full tuning therefore need either an optimizer offloaded to host RAM (FP32 master + moments ≈ 43 GB of the 125 GB) or a library the image lacks (8-bit Adam, LoRA via `peft`): each is a named recipe deviation and a new dependency or code path. Candidates measured by `--probe-memory`: (a) all-bf16, gradient checkpointing, nothing frozen; (b) bf16 on GPU with FP32 master weights and moments offloaded to CPU; (c) `train_expert_only`. **(a) has a known risk:** updates of size lr × step can fall below bf16's resolution and be lost; the preflight measures, per tensor, the fraction of elements whose value changes in one update at the peak and at the floor learning rate, and reports it next to (b)'s. No threshold is fixed here (none is justified yet); the user picks from what the preflight shows fits under the runner's free-memory guard | an unmeasured guess here would decide the result |
| D6 | **Execute 10 actions per inference** (`n_action_steps` 10), as A0 does; pi0.5's native 50 (1.7 s open loop at 30 Hz) is not used | otherwise the comparison mixes the model with the re-planning rate. This is a deviation from "native recipe", flagged for review |
| D7 | Same prompts as P2 (`put the <object> in the <bin> bin`), same two cameras in the same order, no third camera. `pi05_base` was trained with three camera slots (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); with two configured image features the model receives two image blocks, and adding an empty third (`empty_cameras=1`) is a separate recipe choice not taken here | inputs held equal |
| D8 | Training package `experiments/pi05_props_p2/` (admission-pi image), gates `--check`, `--preflight`, `--resume-proof`, `--train [--stop-at T] [--resume]`, the thermal guard and snapshot pause from the X-VLA runner. No `--equivalence` (no reference loop exists for pi0.5) | the X-VLA runner's proven structure |

## Review requirements before implementation

[Audit R8](../../VLA_INPUTS_AUDIT_20261030.md) qualifies D1/D3/D5/D8:

- Pin installed LeRobot source hashes and the transformers commit; run `pip check` plus actual offline forward,
  backward, optimizer step and queued inference. If the two-package image (transformers + tokenizers) fails, stop and amend its dependency
  manifest before runs; do not bypass dependency or physics-equivalence gates.
- Effective batch 8 is a proposed experimental choice. Gradient accumulation can preserve batch 32 despite small
  microbatches; memory alone does not require changing effective batch. Freeze the choice and schedule together.
- Both D5(a) and D5(b) can lose small updates when weights are stored in bf16 without FP32 master weights. Compare a
  supported optimizer path with FP32 reference updates; report dtypes of parameters, gradients, moments and any master
  weights. Fraction-changed is diagnostic, not proof of training fidelity. Measure actual optimizer compatibility,
  round-trip and resume. A supported LoRA or frozen-backbone recipe is a separately named alternative if needed.
- D8's absence of an old project runner does not remove the need for reference parity: match the installed policy and
  processors on a fixed batch, forward/backward/update and action queue. Check padding, action masks/reduction, camera
  order, prompt truncation and normalization round-trip, including quantile tails and constant dimensions. The upstream
  quantile transform need not clip; do not assert all state tokens are 0–255 without inspecting the exact code path.
- Five epochs means each model's complete-window epoch. Chunk 50 excludes 20 more observation starts per episode than
  chunk 30. Report observation starts, samples and supervised action positions separately; do not claim identical exposure.
- Measure disk for assets, image layers, videos, gate/preflight/resume-proof outputs, final optimizer state and atomic-write
  peaks. Host free space and syslog growth quoted below are historical workstation observations, not current laptop facts.
- M0's frozen core is not edited for A1. Create `experiments/pi05_props_p2/eval_adapter.py`, loaded lazily and independently
  bound. GPU evaluation is separately scheduled from training. Include M0's matched-phase secondary evaluation.

## Primary result

M0's primary test (paired McNemar on C, two-sided, α = 0.05) of A1 against A0. **Secondary (exploratory):** milk success, and
the `never_lifted` share of failures.

## Prediction

None confident. pi0.5 has a larger language model and reads the state as text; on 480 training episodes it may underfit or
overfit differently from X-VLA. A null result is reported as such.

## Costs and blockers

- **Not yet measured:** seconds per update, peak memory with activations, snapshot sizes. From the measured parameter
  count: a checkpoint is 14.5 GB in FP32 or 7.2 GB in bf16 (×7 ≈ 101 or 51 GB); a snapshot with FP32 master weights and moments
  about 43 GB, all-bf16 about 29 GB. Against **123 GB free on 2026-10-30** (the earlier `/var/log/syslog` growth is no longer
  visible: only `syslog.1`, 32 MB, dated 2026-10-25, is present), FP32 checkpoints plus snapshots do not fit: save checkpoints
  in bf16 or fewer of them, keep one snapshot, or reclaim; the user decides. Re-measure `df -h` before every gate.
- Time: if an update is 2–4 × X-VLA's 0.524 s, 120,900 updates are 35–70 h raw, more with thermal pauses (X-VLA's effective
  rate was ~0.9 s). The preflight's measured rate gives the projection; the user decides before `--train`.
- Rollouts: per-worker GPU memory and inference time are measured in M0's smoke for this arm; worker count follows.

## What M1 cannot show

- That pi0.5 is a better model in general: A1 is pi0.5 with its recipe (and the D5 memory strategy) on the 480 training episodes (60 validation episodes), against X-VLA
  with P2's recipe.
- Anything about pi0.5 with richer inputs (that would be a second variable; not planned).

## Out of scope

pi0, pi0-FAST, GR00T, LoRA or other adapters (unless D5 forces one, which would then be named), a second pi0.5 run.

## Plan

[../plans/2026-10-30-vla-inputs-m1-pi05.md](../plans/2026-10-30-vla-inputs-m1-pi05.md).
