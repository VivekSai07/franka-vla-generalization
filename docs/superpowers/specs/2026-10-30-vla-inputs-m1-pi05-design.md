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

## Decisions (proposed, for review)

| id | decision | reason |
|---|---|---|
| D1 | **Two new images**, each the existing image plus only the transformers build pi0.5 requires, installed with `--no-deps`: `admission-pi` (training, offline tests) and `robosuite-pi` (rollouts). Pinned by digest. A test asserts `pip freeze` differs from the parent image in transformers alone | the existing images must not change (every sealed run is bound to them); one new variable at the software level |
| D2 | Base weights `lerobot/pi05_base` and the tokenizer `google/paligemma-3b-pt-224` downloaded on the host through the proxy into `outputs/research/pi05_assets/`, sha256 of every file pinned in the contract; the Hugging Face token is never written to the repository or a container | offline containers; reproducibility |
| D3 | **pi0.5's own recipe** (chunk 50, quantile normalisation with train-only q01/q99, state in the prompt, 224 × 224 with padding, its LR, warmup, decay, betas, weight decay, clip), except: **effective batch 8**, and the decay spans the whole run | user-approved "native recipe"; batch 8 because of memory (D5) and so the checkpoint grid aligns with X-VLA's by schedule fraction |
| D4 | **Exposure 5.0 epochs:** with chunk 50 a training episode has 452 − 49 = 403 windows, 480 × 403 = 193,440; 5.0 epochs = 967,200 samples = **120,900 updates** at batch 8. Checkpoints at the same fractions as X-VLA's grid (0.4, 0.6, 0.7, 0.8, 0.9, 1.0): 48,360, 72,540, 84,630, 96,720, 108,810, 120,900, plus update 1 | exposure, not update count, is what the repo holds constant |
| D5 | **Memory strategy decided by measurement, before training, and recorded as part of the arm's name.** Full fine-tuning with FP32 parameters and Adam needs about 16 bytes per parameter: about 53 GB for ~3.3 B parameters, more than the card. Candidates, in order: (a) bf16 parameters and bf16 Adam states, gradient checkpointing, nothing frozen (~8 B/param ≈ 27 GB + activations); (b) as (a) with FP32 Adam states (~12 B/param ≈ 40 GB); (c) vision encoder frozen; (d) `train_expert_only`. **(a) has a known risk:** updates of size lr × step can fall below bf16's resolution and be lost; the preflight measures, per tensor, the fraction of elements whose value changes in one update at the peak and at the floor learning rate, and reports it next to (b)'s. No threshold is fixed here (none is justified yet); the user picks from what the preflight shows fits under the runner's free-memory guard | an unmeasured guess here would decide the result |
| D6 | **Execute 10 actions per inference** (`n_action_steps` 10), as A0 does; pi0.5's native 50 (1.7 s open loop at 30 Hz) is not used | otherwise the comparison mixes the model with the re-planning rate. This is a deviation from "native recipe", flagged for review |
| D7 | Same prompts as P2 (`put the <object> in the <bin> bin`), same two cameras in the same order, no third camera | inputs held equal |
| D8 | Training package `experiments/pi05_props_p2/` (admission-pi image), gates `--check`, `--preflight`, `--resume-proof`, `--train [--stop-at T] [--resume]`, the thermal guard and snapshot pause from the X-VLA runner. No `--equivalence` (no reference loop exists for pi0.5) | the X-VLA runner's proven structure |

## Primary result

M0's primary test (paired McNemar on C, two-sided, α = 0.05) of A1 against A0. **Secondary (exploratory):** milk success, and
the `never_lifted` share of failures.

## Prediction

None confident. pi0.5 has a larger language model and reads the state as text; on 480 training episodes it may underfit or
overfit differently from X-VLA. A null result is reported as such.

## Costs and blockers

- **Not yet measured:** seconds per update, peak memory, checkpoint and snapshot sizes. Estimates for planning: bf16 weights
  ≈ 6.6 GB per checkpoint (×7 ≈ 46 GB); a snapshot with optimizer state 27–40 GB. With one kept snapshot plus one being written
  the working set is about 100–130 GB, against **123 GB free** today and syslog growing ~30 GB/day: **disk is the first
  blocker**, and what to reclaim is the user's decision.
- Time: if an update is 2–4 × X-VLA's 0.524 s, 120,900 updates are 35–70 h raw, more with thermal pauses (X-VLA's effective
  rate was ~0.9 s). The preflight's measured rate gives the projection; the user decides before `--train`.
- Rollouts: per-worker GPU memory and inference time are measured in M0's smoke for this arm; worker count follows.

## What M1 cannot show

- That pi0.5 is a better model in general: A1 is pi0.5 with its recipe (and the D5 memory strategy) on 540 episodes, against X-VLA
  with P2's recipe.
- Anything about pi0.5 with richer inputs (that would be a second variable; not planned).

## Out of scope

pi0, pi0-FAST, GR00T, LoRA or other adapters (unless D5 forces one, which would then be named), a second pi0.5 run.

## Plan

[../plans/2026-10-30-vla-inputs-m1-pi05.md](../plans/2026-10-30-vla-inputs-m1-pi05.md).
