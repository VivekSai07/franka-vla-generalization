# SmolVLA Pilot Training Implementation Plan

> **For agentic workers:** execute tasks in order with test and review gates.

**Goal:** launch the user-approved 20,000-update local pilot only after unit and
GPU preflight checks, without changing the validated dataset or collection code.

**Architecture:** separate `training/smolvla_pilot.py` runner and small
`training/pilot_protocol.py` helpers. Reuse the existing split-safe statistics,
complete-chunk, model processor and FP32 reload utilities. No Hub dependency.

**Tech Stack:** existing `smolvla:rtc-offline` image, LeRobot 0.4.4, PyTorch
2.7.1+cu128, single RTX 6000 Ada, Python in Docker only.

## Approval and global constraints

User explicitly approved **20,000 optimizer updates**, effective batch **32**
(microbatch **8**, **4** accumulation steps), AdamW peak LR **1e-4**, **1,000**
linear warmup updates then cosine decay to **1e-5**, FP32 trainable weights and
optimizer moments with BF16 training autocast. Fresh pinned `lerobot/smolvla_base`,
frozen VLM, trainable expert/state/action projections; unchanged stock loss.

Chunk length 50, action horizon 10. Exclude all boundary-padded chunks, assert
the loader's `action_is_pad` is false rather than relying on its mismatched loss
mask name. The stock loss still includes padded action dimensions; physical
8-D action metrics must be reported separately. Train-only physical statistics.
No policy input includes replay states or object poses. No image augmentation.

Use dataset `datasets/franka_pilot_20260915`, explicit 300/120 manifest splits;
all 120 validation episodes get fixed probes at frames 60, 107 and 347 (approach,
nominal close and nominal release). These probes are diagnostics, not an exhaustive
validation loss or success score. Nine evenly spaced training probes provide
a fit comparison. Evaluate at update 0 and every 2,000 updates. Keep every
2,000-update checkpoint and record the lowest held-out arm-MAE checkpoint
(ties retain earlier); do not select by training loss or touch final scenes.

Seed 20260916; AdamW betas (0.9,0.95), weight decay 1e-10, gradient norm cap 10.
Uniform shuffled complete-chunk sampling, drop incomplete microbatches so every
optimizer step sees exactly 32 samples. Preserve epoch/cursor/RNG in checkpoint
state for audit only: no resume implementation or automatic restart.

No upload, final/tuning scene rollout, real robot action or automatic retraining.
Closed-loop 40-second benchmark evaluation is a later explicit step. Offline
reconstruction cannot establish task success. No success guarantee.

## Tasks and acceptance

### 1. Protocol helpers and tests
- [ ] Add tests in `tests/test_pilot_training.py` and observe failure before implementation.
- [ ] Implement strict 300/120 coverage/disjoint IDs, selected row indices, complete
  chunks, fixed validation probes, finite no-padding batch checks and LR function.
- [ ] Test overlap/duplicate/incomplete split rejection, out-of-range probes,
  LR endpoints, padding/NaN rejection and accumulation equivalence on a toy model.
- [ ] Run tests in the existing Docker image; no host dependency changes.

### 2. Isolated trainer
- [ ] Require fresh bound validation, CUDA and fresh output before optimizer work.
- [ ] Strict-load cached base with custom 9-state/8-action/two-camera schema,
  promote only trainable storage, save protocol/base/source hashes and train statistics.
- [ ] Train with finite checks, fixed full batches and correctly scaled accumulation;
  check first-step FP32 optimizer state and actual projection weight change.
- [ ] Save model/processors/optimizer/RNG and a completion marker per checkpoint;
  verify precise FP32 reload and seeded action agreement before sustained training.
- [ ] Preserve RNG/mode across probes; write progress at least every 20 updates
  and record failure status without silently changing settings or restarting.
- [ ] Recheck provenance-bound data at run completion; never rewrite source/data gates.

### 3. Verification and launch
- [ ] Run full unit suite, read-only review and editor/whitespace checks.
- [ ] Run an isolated two-update GPU preflight with identical batch/precision and
  save/reload checks plus a held-out probe. This scratch run is not resumed.
- [ ] Recheck idle GPU/disk; launch one detached offline GPU container using fresh
  output `outputs/train/smolvla_pilot_20260916`, read-only source/data, writable
  output only. No credential mounts or network.
- [ ] Inspect actual startup/progress and document run/container identifiers and
  measured throughput; distinguish running from completed training.

## Expected exposure and resources

528 frames/episode yields 479 valid 50-action starts. Train has 158,400 frames
and 143,700 valid starts. 20,000 x 32 = 640,000 sampled starts, about 4.45 passes
over the eligible starts (not 32 million independent observations). Each full
epoch drops four starts due to microbatch divisibility; reshuffle next epoch.
Checkpoint budget is approximately 25 GB including optimizer state; require at
least 35 GiB free. Measure actual wall time rather than promising a duration.

Existing work remains on `feat/reusable-validation` in the user-selected checkout.
Do not stage unrelated files, credentials or generated artifacts.