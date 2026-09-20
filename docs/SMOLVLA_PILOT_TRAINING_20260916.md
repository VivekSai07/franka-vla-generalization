# Local SmolVLA pilot training — 2026-09-16

## Verified completion

**Complete: 20,000/20,000 updates, exit 0, no OOM.** Container ran from
2026-09-16 16:25:22 to 21:16:00 UTC (approximately 4 h 51 min including setup,
evaluation and checkpoints). Optimizer-update wall time total: 4.584 hours.
640,000 sampled chunks represent approximately 4.45 training passes.

Fresh completion inspection verified all 20,000 consecutive metric records were
finite, all 11 saved model hashes matched their checkpoint markers, and current
dataset/runtime/source bindings passed. Each checkpoint's recorded GPU reload
check preserved FP32 trainable weights exactly with zero prediction difference.
The completion inspection did not rerun GPU inference or full demonstration replay.

| Update | Held-out arm MAE, 50 actions (rad) | First 10 MAE (rad) | Gripper accuracy |
|---|---:|---:|---:|
| 0 | 0.256941 | 0.175772 | 64.67% |
| 6,000 | 0.030419 | 0.016812 | 87.53% |
| 12,000 | 0.020165 | 0.012727 | 93.12% |
| 18,000 | 0.017288 | 0.007938 | 93.71% |
| 20,000 | 0.016913 | 0.008329 | 93.60% |

The predefined lowest held-out full-chunk arm-MAE selector chooses **20,000**:
[checkpoint marker](../outputs/train/smolvla_pilot_20260916/checkpoints/020000/checkpoint_result.json).
Model SHA256: `a4cc794608091d6b6cab552c373fe745b8c81b6dd4ca18d092bbef15d7402b29`.
Full-chunk validation error improved 93.42% from baseline. Final first-ten error
is approximately 0.477 degrees per joint/action. First/last 200-update mean
training loss: 0.238879 / 0.005393 (different batches/noise, not a fixed objective).

The 18,000 checkpoint is slightly better on first-ten arm MAE and gripper
agreement. This is a metric trade-off, not proof of significant degradation or a
reason to silently change the predefined selection criterion. Preserve both.
Training probes finish at 0.015995 rad arm MAE and 100% gripper agreement, while
held-out agreement is 93.60%; remaining generalization and grasp/release risks
must be assessed in closed loop. These are 360 fixed held-out probes, not an
exhaustive validation loss or autonomous success rate.

Evidence: [final result](../outputs/train/smolvla_pilot_20260916/result.json).
**No closed-loop task evaluation has run.** Recommended next milestone: evaluate
the selected 20,000 checkpoint on the independent 120 tuning scenes with the
fixed 40-second/horizon-10 protocol, saving videos and intervention diagnostics.
Keep the 240 final scenes sealed. No further training, rollout or upload was
started by this completion check; obtain authorization for the next job.

The launch/progress descriptions below are historical, not an active-job claim.

## Authorization

The user deferred the blocked Hugging Face upload and authorized actual local
training. After reviewing the proposed recipe, the user selected **Approve
20,000-update pilot**. This supersedes prior no-training/upload-first holds.
No upload, new collection or final-scene evaluation is part of this run.

## Approved recipe

| Setting | Value |
|---|---|
| Initialization | Fresh cached `lerobot/smolvla_base`, revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Demonstrations | Explicit manifest IDs: 300 train / 120 validation |
| Training frames / complete chunk starts | 158,400 / 143,700 |
| Optimizer updates | 20,000 |
| Microbatch / accumulation / effective batch | 8 / 4 / 32 |
| Sampling exposure | 640,000 chunks, approximately 4.45 passes over eligible starts |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 1e-10, norm cap 10 |
| Learning rate | Linear warmup 1,000 updates to 1e-4, cosine decay to 1e-5 at update 20,000 |
| Precision | FP32 trainable weights/Adam moments; BF16 training autocast |
| Trainable / total parameters | 99,880,992 / 450,046,176; VLM frozen |
| State / action / cameras | 9 / 8 physical dimensions; two 512x512 RGB views |
| Chunk / execution horizon | 50 / 10 |
| Seed | 20260916 |
| Checkpoints | Update 1 gate, then every 2,000 updates; all retained |
| Offline evaluation | Baseline and every 2,000 updates; 9 train probes, 360 held-out probes |

Statistics are recomputed from the 300 training demonstrations, not whole-dataset
metadata. The data loader samples full 50-action chunks only and asserts no
`action_is_pad` entries. This avoids relying on the upstream mismatched temporal
mask key. The **stock loss is unchanged**, including its 24 padded action
dimensions; physical arm MAE and gripper metrics are reported separately.

Validation probes use offsets 60, 107 and 347 in each of 120 held-out episodes;
these correspond to approach and nominal close/release phases, not necessarily
every episode's exact gripper transition. They are fixed diagnostics, not an
exhaustive validation-set loss or closed-loop score. Lowest held-out full-chunk
arm MAE identifies a candidate checkpoint; ties retain the earlier checkpoint.
No tuning/final scene rollouts run automatically. The eventual fixed benchmark
uses 40 seconds/1,200 ticks, unlike older 20-second smoke rollouts.

## Implementation and verified preflight

- [Trainer](../training/smolvla_pilot.py), [protocol helpers](../training/pilot_protocol.py),
  [tests](../tests/test_pilot_training.py), [implementation plan](superpowers/plans/2026-09-16-smolvla-pilot-training.md).
- **54/54 regression tests passed** in Docker. New tests first failed on the
  absent implementation; split/schedule/padding/accumulation tests then passed.
- Read-only review found no confirmed actionable correctness defect.
- Container `smolvla-pilot-training-preflight` exited **0**, no OOM.
- [Preflight result](../outputs/train/smolvla_pilot_preflight_20260916/result.json):
  two scratch optimizer updates, finite losses, verified FP32 Adam state,
  actual projection changes, exact saved/reloaded trainable weights and zero
  prediction difference at both checkpoints, final data binding PASS.
- The preflight uses three validation probes only; the full run uses all 360.
- Measured update times: 1.066 and 0.822 seconds. Observed peak Torch allocation
  including checkpoint reload: approximately 5.53 GB (not total GPU usage).
- Scratch weights are **not resumed** by the real run.

## Launched full run

- Container: `smolvla-pilot-train-20000`.
- ID: `a6cb77d67efd1696f3a6a40601e40856569c044d0086b79cec822a0e33444c1b`.
- Image: `smolvla:rtc-offline`; existing pinned runtime, no new packages.
- Fresh output: `outputs/train/smolvla_pilot_20260916`.
- Source/dataset mounted read-only; training outputs writable. Network disabled,
  model/tokenizer loading offline; no login or Hub upload.
- GPU was idle, no compute processes at launch; approximately 108 GB disk free.
- Detached container survives SSH disconnection; host and Docker must stay running.
- Initial rough estimate: 5–7 hours including overhead, subject to measured
  sustained throughput. This is not a promised finish time.

**Launch is not completion.** Inspect result status, retained logs and container
exit before claiming training finished. Progress is updated every 20 optimizer
steps; evaluations and checkpoints have distinct statuses. Failures stop without
automatic restart or recipe changes. Preserve partial outputs; no resume loader
is implemented. Checkpoint optimizer/RNG records are for audit, not a promise
of exact restart support. Never load untrusted optimizer pickle files.

## Completion gate

Require exit 0, result status complete, 20,000 completed updates, final data/source
binding PASS and completed precision-preserving checkpoint reload. Report measured
offline metrics and chosen candidate separately from task success. Checkpoints
must use `validation.precision.load_fp32_checkpoint`, not a loader that rounds
FP32 parameters into BF16 storage. Dataset integrity and decreasing reconstruction
error do not guarantee learned stacking success.