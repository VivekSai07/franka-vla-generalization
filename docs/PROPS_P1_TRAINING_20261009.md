# Props P1 X-VLA training — 2026-10-07/09

A dated report; do not edit it to reflect later findings. Training runs in the admission image
(`sha256:08de1e09…714abc`, no simulator); sealed against the P1 collection
(`outputs/props_p1_data/props_p1_20261006`, fingerprint `86e3aadb678963eda8cb27d0104d6e22778d4cc25c497829e8fb47937ac53bcf`).
Package: `experiments/xvla_props_p1/`. Plan:
[docs/superpowers/plans/2026-10-07-props-p1-training-eval.md](superpowers/plans/2026-10-07-props-p1-training-eval.md).

## What was run

| gate | result |
|---|---|
| `--check` | passed — fingerprint recomputed and matched the seal; binding, statistics, processor checks all passed |
| `--preflight` | passed — 2 GPU updates, FP32 weight + Adam round trip; update 2 measured 0.895 s |
| `--resume-proof` | passed — a resumed process reproduced update 2 byte-for-byte (loss, grad norm, sampled rows, learning rates, model/Adam SHA-256) |
| `--train` | **126,900 / 126,900 updates = 5.0 epochs**, all 7 checkpoints present and FP32-reload-verified, `final_data_binding_passed: true` |

**Recipe**: identical to B3 except the horizon and the data — effective batch 8 (microbatch 4 ×
accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and cosine
decay to 1e-5, FP32 parameters with bf16 autocast, seed 20261007, from the base X-VLA checkpoint.
203,040 complete chunk starts (480 training episodes × 423 starts, 452 frames each — not 528) ×
5 epochs / effective batch 8 = **126,900 updates exactly, no rounding**.

**Checkpoint schedule** (dense from epoch 2, chosen from this repo's own checkpoint-peak evidence:
B1 peaked at 3.35 epochs, B2 at 5.0, B3 at 4.0 with 5.0 its worst by 22 points):

| update | epoch | size |
|---|---|---|
| 1 | — | 3.3 GB |
| 50,760 | 2.0 | 3.3 GB |
| 76,140 | 3.0 | 3.3 GB |
| 88,830 | 3.5 | 3.3 GB |
| 101,520 | 4.0 | 3.3 GB |
| 114,210 | 4.5 | 3.3 GB |
| 126,900 | 5.0 | 9.9 GB (carries `training_state.pt`) |

Final-update loss 0.00129 (final `grad_norm` 0.394); loss at updates 1 / 25,001 / 50,001 / 75,001 /
100,001 / 109,201 / 126,900: 1.145 / 0.00263 / 0.00424 / 0.00172 / 0.00160 / 0.000233 / 0.00129 —
monotonic collapse by epoch 1, as in every prior X-VLA run here. **Not treated as a capability
signal**: only the rollout evaluation below is.

## An interruption, handled by design

Measured rate once warmed: **0.84–0.87 s/update**, matching B3's quoted figure closely. At update
~111,280 (87.6% done, 26.2 h of container uptime), three colleagues' containers began sharing the
RTX workstation (`isaac-sim`, `determined_wozniak`); GPU utilization hit 99% and the measured rate
degraded to ~2.1 s/update (2.5×). The run was stopped deliberately
(`docker stop`, a plain kill — the runner has no SIGTERM handler, which is by design: every stop is
treated as a crash) and resumed later the same day once the GPU was confirmed idle. Newest verified
snapshot was `snap_109200`; the resume quarantined `result.json`/`metrics.jsonl` rows past that
point (`updates 109,201–111,280` recomputed, nothing else) and replayed the sampler to position.
`--train --resume` reproduced the uninterrupted run's design exactly (byte-verified in
`--resume-proof` beforehand): the resume cost ~2,089 updates of recomputation, not any correctness
risk. Post-resume: 17,700 updates in 15,206 s at the same ~0.85 s/update once the GPU was actually
free. **Combined container uptime ~30.4 h** (94,223 s pre-pause + 15,206 s post-resume) across the
two phases; true unique-update wall-clock is lower.

## Disk

Checkpoints 29.7 GB (6 × 3.3 + 9.9), plus the preflight/resume-proof gate directories (23 GB each,
retained for the training run's own validation, not deleted). 66.6 GB of completed/superseded B3
artifacts (already published) were reclaimed before the run to clear the guard; final free space
57 GB.

## Decision

Training complete and verified; proceeds to evaluation
([docs/PROPS_P1_EVALUATION_20261009.md](PROPS_P1_EVALUATION_20261009.md)).
