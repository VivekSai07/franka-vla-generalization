# Props P2 — X-VLA training Implementation Plan

> **For agentic workers:** executed inline in the session that wrote it. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train X-VLA on the sealed P2 dataset with P1's exact recipe, about 2x faster with bit-identical
math, in unattended overnight batches that pause themselves before the morning.

**Architecture:** Fork `experiments/xvla_props_p1/` into `experiments/xvla_props_p2/`, rebind it to the P2
seal, and change only the loop's plumbing: fused finiteness checks, background sample decoding, a
deadline pause at snapshots, a thermal guard and GPU telemetry. A new `--equivalence` gate proves the new
loop reproduces P1's loop bit for bit before `--train` is allowed.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, LeRobot 0.4.4 (admission image), RTX 6000 Ada (300 W cap).

**Inputs (measured 2026-10-13 on a real P1 update, 0.896 s):** forward+backward 0.332 s; `verify_precision`
x2 0.321 s (4,505 per-tensor syncs; fused with `torch._foreach_norm` 0.018 s for 3,604 tensors, NaN/inf still
caught); serial sample decode 0.129 s (GPU idle meanwhile); optimizer step 0.090 s; clip 0.017 s.
Projected ~0.47 s/update; 126,900 updates ~16-17 h.

## Global Constraints

- Recipe identical to P1: seed 20261007, microbatch 4 x accumulation 2, effective batch 8, bf16 autocast over
  FP32 parameters/gradients/Adam, AdamW betas (.9, .95), eps 1e-8, clip 10, warmup 500 then cosine to 1e-5,
  126,900 updates (5.0 epochs over 203,040 starts), checkpoints 1, 50760, 76140, 88830, 101520, 114210,
  126900; snapshots every 2,600 updates, 2 kept. Deterministic algorithms on.
- Never edit: `experiments/xvla_props_p1/**`, `experiments/xvla_trial/**`, `experiments/xvla_single_task/**`,
  `experiments/props_p2_data/**`, the P2 spec and collection plan (all bound). Override locally.
- Every check that P1 ran per update still runs per update, equally strict; only its implementation changes.
- Night window (user): about 19:00 to 06:00. Runs pause themselves before the deadline and never need polling.
- Admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`, `--runtime nvidia`,
  `--ulimit nofile=65536:65536`, `--tmpfs /tmp:rw,size=16g`.
- Commits: user identity, no trailers, short messages; never commit `outputs/`; do not push.

## Gate order

`--check` (CPU) -> `--preflight` (2 updates) -> `--resume-proof` -> `--equivalence` (30 updates, new loop vs
P1 loop, byte-equal weights and equal losses) -> `--train --stop-at <epoch>` (night 1) ->
`--train --resume --stop-at <epoch>` (night 2).

## Tasks

- [ ] **1. Fork and rebind** to `outputs/props_p2_data/props_p2_20261013` (seal, plan, empty substitutions,
  provenance, validation record); residue scan; fork-hygiene test; package README.
- [ ] **2. Fused checks** `verify_precision_fast`: same host-side dtype/requires-grad/presence checks, one
  `torch._foreach_norm(ord=inf)` pass for finiteness. Test: accepts/rejects exactly what
  `xvla_trial.preflight.verify_precision` does on fixtures (NaN, inf, non-FP32, missing grad, missing Adam).
- [ ] **3. Background decoding** `Prefetcher`: one thread draws from the same sampler in the same order and
  collates the next update's two microbatches; a lock serializes dataset access with checkpoint probes.
  Test: yields exactly the sampler's sequence.
- [ ] **4. Deadline pause** `--stop-at <unix seconds>`: at each snapshot, pause (status `paused`, exit 0) if the
  next snapshot could not finish before the deadline. Container-level `timeout` as a backstop. Test the
  decision function.
- [ ] **5. Thermal guard + telemetry**: `nvidia-smi` every 20 updates (temperature, power, throttle flags)
  logged with the TRAIN row; at >= 83 C or any thermal slowdown, sleep until <= 75 C (sleep changes no math);
  after 30 min continuously hot, exit `paused_thermal` (resume from the last snapshot). Tests on parsing and
  the decision.
- [ ] **6. `--equivalence` gate**: 30 updates with the P1 loop (serial fetch, per-tensor checks), then 30 with
  the new loop from the same fresh state; model state bytes, per-update losses, grad norms and rows must be
  equal; s/update of both recorded. `--train` requires it.
- [ ] **7. Gates on the GPU** tonight (no office load): check, preflight, resume-proof, equivalence.
- [ ] **8. Night launches** with the deadline; report per night (completed updates, s/update, max temperature).
