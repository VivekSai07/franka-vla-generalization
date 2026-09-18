# Fixed-layout SmolVLA (Task1 implementation only)

Approved design: docs/superpowers/specs/2026-09-18-smolvla-fixed-layout-training-design.md.
No GPU, full semantic admission, actual physics, training or rollouts executed by
the implementation tests. Independent review and main-agent phase gates remain.

## Immutable comparison

- Same sealed24 demos: train0..19, validation20..23. Training10560 rows,
  9580 complete50-action chunks (last offset478). Validation never sampled for
  optimization/statistics; fresh prior-style mean/std/min/max/q01/q99.
- Fresh original local SmolVLA revision c83c3163b8ca9b7e67c509fffd9121e66cb96205;
  VLM/tokenizer7b375e1b73b11138ff12fe22c8f2822d8fe03467. All42 bundle files are
  independently pinned by manifest and byte inventory, including local HF metadata.
  No X-VLA model/cache/report dependency or old recursive source scan.
- Prior SmolVLA recipe: micro8 x accumulation4, seed20260917, 5000 AdamW updates,
  frozen VLM, FP32 trainable expert/state projection before optimizer,
  BF16 training autocast only.450046176 total/99880992 trainable parameters.
  LR500 warmup to1e-4, cosine1e-5 at5000; betas(.9,.95), eps1e-8, decay1e-10,
  clip10.50-action chunk, execution10, state9/action8, two512RGB cameras.
- Checkpoints1/1000/2000/3000/4000/5000 retain original training_state artifacts
  and processors. Every saved marker must meet absolute action error<=1e-5.
  Scratch saves1/2 and additionally restores actual Adam state to a CPU optimizer
  with exact FP32 checks before accepting either checkpoint. Never initialize
  main from scratch. All checkpoint file inventories, CPU/data/cache/source/
  stats proofs rechecked before main and after training.
- Final5000 only:72 observation-only probes (60/107/347 in all20+4 episodes),
  preserving Python/NumPy/Torch RNG and policy mode.20 real explicit policy seeds
  20260918..20260937 share unchanged sourceepisode7 full reset.40s/1200ticks,
  30 strictly consecutive released-stack ticks; gate18/20 only after all20.
  Videos, full state/action traces, separate closure/contact/lift/release/clipping
  metrics. No policy oracle. Same geometry is not spatial generalization.

## Deliberate small ports

The old single_task runner hardcodes pilot metadata/nine demonstrations, so its
orchestration is explicitly ported here, not monkeypatched or dynamically rewritten.
Reuse is limited to pure LR, batch/backward, statistics, FP32 loader and original
save_checkpoint helpers. The baseline SmolVLA rollout loop is explicitly ported
with real seed argument, observation whitelist and subclass chunk counter.
Action/governor/physics/scoring AST is regression-tested unchanged. The counted
loader promotes trainable storage BEFORE strict copy, like the original loader.
No X-VLA autocast, model input adapter or model import is used.

## Future phase entry points (not run by Task1)

Use module experiments.smolvla_fixed_layout.runner with exactly one of --check,
--preflight or --train. No CLI output override, resume, batch tuning or fallback.
--check performs require_fixed_layout_pass once, actual CPU eight-row processor
check and train-only statistics. It alone produces the authoritative CPU proof.
Later phases use exact full parent fingerprint/current fixed provenance, all
source/spec/installed processor hashes, cache bytes and CPU proof, without
repeating full semantic validation at checkpoints.

Main-agent launch only after independent review: pinned image
sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
canonical repository /vla read-only, network none, no credentials. Bind only the
phase's new empty output root writable. Existing empty bind mount is permitted
only via exclusive run_claim.json; any contents/symlink/prior claim fail closed.
Paths are outputs/train/smolvla_fixed_layout_20260918_admission,
outputs/train/smolvla_fixed_layout_20260918_preflight, and
outputs/train/smolvla_fixed_layout_20260918. No overwrite/retry after failure.
Before either GPU phase require35GiB disk and12GiB GPU free, BF16-capable CUDA.
Keep data/cache/repository read-only and retain all failed/successful artifacts.

SmolVLA160000 sampled chunks/frozenVLM/chunk50 differs from X-VLA40000 chunks/
all-trainable/chunk30. This is the approved prior-recipe comparison, not a
compute-matched architecture-only experiment.

Implementation evidence is under outputs/implementation/smolvla_fixed_layout_20260918.