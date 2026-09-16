# Fitting audit: normalization, parameters and grasp targets

Date: 2026-09-15. Audit only: zero optimizer updates, zero rollouts, no
collection or uploads. One diagnostic backward pass checked gradient reachability;
gradients were cleared, with no optimizer or checkpoint save. Existing checkpoint
file hashes and source/data/runtime replay binding were verified unchanged.

## Inputs and reproducibility

- Dataset: `datasets/franka_benchmark_validation_20260915_timing_v2`.
- Episode 0 only, as used by the completed overfit experiment.
- Checkpoint: `outputs/validation/smolvla_overfit_20260915/pretrained_model`.
- Audit entrypoint: `validation/audit_fitting.py`, existing offline Docker image.
- Completed evidence: `outputs/validation/smolvla_fitting_audit_20260915_final`.
  Contains `result.json`, `expert_precision.json`, `rounding_probe.json`.
- Supplementary precision evidence is reproducible with
  `validation/audit_precision.py --base <base-model.safetensors> --checkpoint
  <overfit-model.safetensors> --output <fresh-directory>` inside the same image.
  A rerun in the completed evidence directory's `precision_reproduced/`
  subdirectory was compared with the originally generated JSON artifacts.
- Initial audit outputs without `_final` are incomplete harness attempts, not
  successful audit results. The first needed DataLoader-style action batching;
  the second used a module hook bypassed by a direct `.forward()` call. The final
  audit observes the invoked action-output projection instead. Neither issue
  changes or invalidates the actual earlier batched training implementation.

## 1. Normalization and targets: no mismatch found in checked cases

Recomputed mean/std/min/max/q01/q99 from all 528 episode-0 rows. Every value
matched the saved training-statistics artifact exactly. No state/action standard
deviation was below 1e-5. Validation episodes were not used in this computation.

At frames 0, 107, 347 and 478:

- Saved processor state/action normalization matched `(value-mean)/(std+1e-8)`
  from recomputed training statistics with measured maximum error zero.
- Postprocessing normalized expert actions recovered physical targets with
  maximum error 1.1921e-7 (float32 rounding scale).
- Each 50-action target chunk matched the corresponding raw recorded command
  rows exactly. The 479 valid chunk starts end no later than frame 527, inside
  the training episode. This rules out an off-by-one target lookup in the
  checked chunks; it is not an exhaustive proof of every loader operation.
- Actual feature schema is 9-D state, 8-D action, two 512x512 images.
- Gripper values are exactly 0 and 255; close begins at 107, release at 347.

The [grasp-phase probes](REPLANNING_DIAGNOSTIC_20260915.md) already showed residual
joint prediction errors and release-adjacent gripper decision errors on expert
observations. This audit found no evidence that incorrect saved normalization or
those target lookups explain the residual error.

## 2. Trainability and precision

The VLM is frozen as intended: 350,165,184 parameters, 345 tensors, none changed
from the pinned base checkpoint and none received gradients in the diagnostic.
All 99,880,992 intended expert/projection parameters are marked trainable.

| Group | Parameter count | Changed tensors / total | Tensors with nonzero audit gradient |
|---|---:|---:|---:|
| Action expert | 98,245,840 | 112 / 145 | 145 |
| State projection | 31,680 | 2 / 2 | 2 |
| Action input projection | 23,760 | 2 / 2 | 2 |
| Action output projection | 23,072 | 2 / 2 | 2 |
| Time MLP input | 1,037,520 | 2 / 2 | 2 |
| Time MLP output | 519,120 | 2 / 2 | 2 |

Gradient reachability was checked on one current grasp-frame batch, not every
historical training step. Frozen modules and missing optimizer parameter groups
are not supported as explanations by these checks.

### Important precision concern

129 expert tensors are stored as BF16 and 16 as FP32. All **33 unchanged expert
tensors are normalization weights stored as BF16**, including the 32 layer
normalizations and final expert normalization. Their final values equal the
base snapshot despite the nonzero gradients observed now. This is endpoint
equality, not a historical trace proving they never changed during training.

At BF16 value 1.0, the adjacent representable values are 0.99609375 and
1.0078125. Disposable arithmetic showed:

| Calculation | Result |
|---|---:|
| BF16: 1.0 minus 1e-4 | 1.0 |
| FP32: 1.0 minus 1e-4 | 0.99989998 |

The custom training script constructs ordinary AdamW directly over these
parameters; it does not create FP32 master parameters. BF16 autocast changes
operation precision, not parameter-storage precision. Small normalized optimizer
updates can therefore round away rather than accumulate in those weights.
Exact historical updates/optimizer states were not saved, so the scalar test
demonstrates the mechanism, not every historical update's value.

This is a concrete optimization concern and the strongest actionable next
check, **not a proven sole cause of the unsuccessful grasp**. Many expert
tensors and every projection did change; training was not globally frozen.

## 3. Action-dimension padding contributes to the reported loss

The installed LeRobot 0.4.4 implementation pads the 8-D action to 32 dimensions.
It then slices loss to `max_action_dim`, also 32, and averages. Consequently,
the comment about removing padding does not remove the 24 nonphysical slots.
Those slots have zero clean targets but nonzero flow-matching noise targets.

Observed fixed-noise/time forward probes:

| Expert frame | Mean over physical 8-D slots | Mean over padded 24-D slots | Returned 32-D mean |
|---:|---:|---:|---:|
| 107, close | 0.165696 | 0.022343 | 0.058181 |
| 347, release | 0.119281 | 0.013099 | 0.039644 |

The returned loss equals `0.25 * physical_mean + 0.75 * padded_mean`.
In these probes, padded slots account for approximately 28.8% and 24.8% of the
scalar loss respectively. Thus neither "75% of loss is padding" nor "a small
reported loss proves accurate physical commands" would be correct statements.

This affects training-objective interpretation and potentially optimization.
Training padded dimensions may also be an inherited objective convention rather
than an unintended implementation choice. No active-dimension-only training
ablation has been performed, and the effect on task success is unknown. Do not
change this and parameter precision simultaneously if measuring their effects.

## 4. Temporal padding key mismatch: confirmed, but avoided in prior runs

The loader and processor emit `action_is_pad`, while SmolVLA's forward method
looks for `actions_id_pad`. An artificial all-padding-mask probe with identical
inputs/noise/time measured:

| Mask supplied | Returned loss |
|---|---:|
| `action_is_pad`, all true | 0.162443 |
| `actions_id_pad`, all true | 0.0 |

The final dataset frame really has 49 padded future positions, confirming this
is relevant when training on episode ends. However, our training selected only
complete within-episode chunks, so **no selected training chunk was temporally
padded**. This mismatch does not explain the completed overfit failure.
Any future loader that includes padded tails must address and test it explicitly.

## Conclusions and next decision

The current evidence supports these distinctions:

- Correct train-only statistics, processor roundtrips and sampled target chunks.
- Intended trainable groups receive gradients; VLM stays frozen.
- Actual residual fitting error remains, with a measurable BF16 update-resolution
  concern in trainable expert weights.
- Padded action dimensions contribute to the objective; a padding-mask spelling
  mismatch exists but was avoided by our chunk selection.

**Recommended next controlled change:** keep dataset, targets, learning rate,
seed, update cap and loss unchanged; use FP32 storage for trainable parameters
and optimizer state, retaining BF16 autocast for computation where supported.
First verify an actual small normalization-weight update survives, and that
mixed-dtype inference/save/reload remain valid. Then, only with approval, repeat
the same bounded overfit experiment from the same base initialization and compare
fixed physical probes and the exact training scene. Do not resume the old run.

This is not permission for another optimizer run. No training code, installed
LeRobot package or checkpoint was patched by the audit. Bulk collection remains
on hold. Report which precision/loss convention each eventual benchmark uses.