# Precision-only bounded overfit repeat

User authorization: repeat the recommended precision experiment, 2026-09-15.
No data collection or loss changes. Same replay-verified timing-v2 episode 0,
pinned base initialization, seed 20260915, batch 2, 2,000 updates, AdamW 1e-4,
50-action chunks, 10-action execution horizon and one 600-tick training-scene
rollout. Dataset/normalizers and fixed probe definitions match the earlier
[overfit diagnostic](OVERFIT_VALIDATION_20260915.md).

## Single changed factor

Before creating AdamW, promote only trainable parameters to FP32 without
changing their initial numeric values. Frozen VLM storage is untouched. Retain
the same BF16 autocast context for training and the same existing inference
path, if supported. The purpose is to retain small parameter updates and FP32
Adam moments, not to remove padded action dimensions or change temporal masks.

Verify small-update survival on normalization weights after the first step,
FP32 Adam moments, final normalization changes, fixed physical probes and
checkpoint integrity. On reload, construct/promote parameters before loading
saved weights so FP32 values cannot be rounded back into BF16 parameters.
Require exact saved/reloaded trainable weights and seeded prediction agreement.

## Artifacts and gates

New output: `outputs/validation/smolvla_overfit_fp32_20260915`.
Entrypoint: `validation/smolvla_smoke.py --overfit --fp32-trainable`.
Helper: `validation/precision.py`; tests: `tests/test_precision.py`.

- [x] Promotion preserves values and frozen dtypes; small AdamW update survives.
- [x] Real-policy mixed-dtype inference and gradient path pass before launch.
- [x] Full bounded repeat finishes, moment precision and normalization updates verified.
- [x] Reload preserves exact weights and predictions; training-scene rollout recorded.
- [x] Compare fixed probes and task outcome; no causal claim from one seed alone.

If mixed-dtype execution fails, stop and document before changing another
experimental factor. Do not silently extend the budget or resume earlier weights.
Bulk collection remains on hold. Completed results follow.

## Execution and interpretation notes

Twenty unit tests passed. Real-policy preflight confirmed inference without an
added autocast context and a finite BF16-autocast backward pass; it performed
no optimizer update. The bounded job is `smolvla-overfit-fp32-2000`, ID
`9b83123a9a408c0576161e49b293d6c62445408864c0bedfd2a85476322c319d`.

Static review found no blocking defect or objective change. The first-step
gate requires at least one normalization tensor to change; inspect per-tensor
results before claiming all 33 changed. Full reload exactness remains a runtime
gate. The tests are not a full multi-device or checkpoint-loader regression suite.

Precision is the changed experimental factor, but its numerical effects are
not limited to update accumulation: promotion can also change arithmetic in
operations not controlled by autocast, particularly inference. Do not attribute
any performance difference exclusively to recovering rounded-away Adam updates.
This remains one scene and one seed, not a generalization result.

## Completed outcome

Container exited 0, no OOM. Exactly 2,000 updates and one 600-tick rollout
ran. No optimizer-budget extension, new demonstration, loss change or checkpoint
resume occurred. All prior checkpoints remain untouched.

### Precision mechanism and reload checks

- 129 trainable tensors were promoted from BF16 to FP32, preserving their
	initial numeric values. Total trainable parameter count remained 99,880,992.
- All **33 expert normalization tensors changed on the first optimizer step**;
	maximum first-step change was 0.0001000166. All 33 also differ from their
	initialization at the final checkpoint.
- AdamW first/second moments were verified finite and FP32 after the first
	and final steps. Trainable parameter storage was verified FP32.
- Reloaded trainable parameters exactly matched every saved tensor; seeded
	postprocessed action predictions matched with maximum difference zero.
- Tracked action projection maximum change: 0.01179063.
- Training time including fixed probes: 248.81 seconds (prior: 238.23 seconds).

**Loader warning:** use `validation.precision.load_fp32_checkpoint` for these
new weights. It promotes the destination parameters before loading. The older
generic load path can instantiate BF16 expert parameters and round saved FP32
values during copying. The standard inference server and earlier standalone
diagnostics have not been adapted to this new checkpoint's storage contract;
do not claim they preserve it without an explicit loader check. The saved config
alone does not encode our custom trainable-storage policy; the run result does.

### Fixed training-example probes

| Update | Prior arm MAE (rad) | FP32-storage arm MAE (rad) | FP32-storage binary gripper agreement |
|---:|---:|---:|---:|
| 0 | 0.214860 | 0.214837 | 55.33% |
| 100 | 0.106771 | 0.125368 | 96.00% |
| 500 | 0.072774 | 0.086670 | 97.33% |
| 1,000 | 0.059276 | 0.055492 | 99.33% |
| 2,000 | 0.044300 | 0.047295 | 99.78% |

At the final probe, first-ten-action arm MAE was 0.045206 rad (prior 0.039062),
and gripper MAE was 18.6916 command units (prior 18.8231). Thus final arm fitting
did not improve under this budget, although binary gripper agreement improved
slightly on these fixed training samples. The small difference at update zero
already demonstrates that storage promotion also affects inference arithmetic;
the comparison is not solely an isolated optimizer-rounding measurement.

### Training-scene rollout

| Metric | Prior native storage | FP32 trainable storage |
|---|---:|---:|
| Released-stack successes | 0/1 | 0/1 |
| Control ticks / simulated time | 600 / 20 s | 600 / 20 s |
| Arm-joint clipping | 0 ticks | 0 ticks |
| Gripper clipping | 121 ticks | 103 ticks |
| Bilateral target contact | 0 sampled ticks | 0 sampled ticks |
| Maximum target rise | 8.22 mm | 4.47 mm |
| Closest finger-body midpoint to target centre | 66.85 mm | 61.03 mm |
| Final target/base XY separation | 566.00 mm | 482.21 mm |

The new rollout took 102.59 wall seconds and did not form a stack. Contact
metrics are sampled at control boundaries; the distance proxy is not calibrated
fingertip clearance. Fewer gripper clips are not proof of better control, since
clipping before binary thresholding does not change the final open/close decision.

## Interpretation and next decision

FP32 storage demonstrably allows the previously problematic small normalization
updates to survive. This supports the precision concern identified by the audit.
It does **not** establish that the concern caused the task failure: the capped
repeat still failed, and final arm reconstruction was slightly worse.

Do not collect 300 episodes or keep extending training merely because this
precision check passed. The remaining fitting error warrants a consolidated
review before another experiment. A possible next bounded test is memorizing
a fixed observation/action chunk with explicit physical-error criteria, keeping
the newly verified storage convention; padded-dimension loss can be examined
as a separate factor rather than bundled into a larger training run. Neither
that test nor a loss modification was performed here.

Report: `outputs/validation/smolvla_overfit_fp32_20260915/result.json`.
Video: `rollout_0.mp4`; trace: `rollout_0_trace.npz`, under that output.
Checkpoint/processors are in its `pretrained_model/` directory. Existing dataset,
training-only statistics and canonical split remain unchanged. No commits or
uploads were made; branch changes remain uncommitted for review.