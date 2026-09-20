# Same-checkpoint grasp and replanning diagnostic

Date: 2026-09-15. User approved evaluation-only diagnosis after the bounded
overfit test. No optimizer updates, collection, upload or checkpoint resume.

## Question and controls

Compare execution horizons 10 and 1 using the same 2,000-update checkpoint,
same saved training scene (episode 0, red on green), same image/state/action
interface, synchronized 30 Hz timing, scoring and 600-tick limit. The policy
still predicts 50 actions in both cases. Only the number executed before a
new prediction changes. Reset the action deque between conditions and count
actual chunk predictions. Reload checkpoint/normalizers without fitting them.

Use the same initial rollout RNG seed for each condition. The first action
must match, but subsequent per-tick noise draws are not matched: horizon 1
invokes the stochastic policy more frequently. This is one paired rollout
per condition, not a seed-averaged success-rate estimate or a deterministic
noise-controlled causal experiment.

Before rollouts, probe expert observations around recorded close/release
transitions (offsets -15, -5, -1, 0, 1, 5, 15 frames). Compare first-action
and first-ten-action physical joint errors and binary gripper decisions with
recorded expert commands. Use only valid within-episode 50-action chunks.
These probes are on expert states, not the states encountered by the policy.

## Inputs and outputs

- Dataset: `datasets/franka_benchmark_validation_20260915_timing_v2`.
- Checkpoint: `outputs/validation/smolvla_overfit_20260915/pretrained_model`.
- Fresh output: `outputs/validation/smolvla_replanning_20260915`.
- Entrypoint: `validation/replanning_probe.py`.
- Both camera views, bounds/thresholding, strict released-stack scoring and
  diagnostic traces reuse the previous rollout helper unchanged.
- Hash checkpoint files before/after evaluation and recheck the dataset's
  source/runtime/replay binding. Do not overwrite old results.

## Interpretation

If shorter replanning succeeds or materially improves contact/lift, that supports
further execution-horizon testing; it does not establish generalization. If
both fail, inspect whether joint or gripper errors occur already on expert
observations versus only during rollout. An absolute-time trace comparison
after paths diverge is not a policy error on the same state.

Gripper saturation followed by binary thresholding does not itself change the
open/close decision; separate clips, threshold decisions and true contact.
Do not keep increasing the experiment budget without another explicit decision.

## Status

Completed normally, exit 0, no OOM. Container `smolvla-replanning-10-vs-1`, ID
`d1a5d083d567d705fb5581c5cb887b0a7e7296e062864251cf0a145d43427f83`.
Eighteen unit tests passed. Read-only review found no concrete new comparison
defect; live reset/count behavior is additionally checked by runtime assertions.

## Results: faster feedback did not produce a grasp or stack

| Metric | Execute 10 actions | Execute 1 action |
|---|---:|---:|
| Task successes | 0/1 | 0/1 |
| Actual chunk predictions | 60 | 600 |
| Control ticks / simulator time | 600 / 20 s | 600 / 20 s |
| Wall time including rendering/encoding | 101.40 s | 191.40 s |
| Arm-joint clipping | 0 ticks | 0 ticks |
| Gripper clipping | 121 ticks | 314 ticks |
| Bilateral target contact | 0 sampled ticks | 0 sampled ticks |
| Maximum target rise | 8.22 mm | 1.20 mm |
| Closest finger-body midpoint to target centre | 66.85 mm | 39.18 mm |
| Final target/base XY separation | 566.00 mm | 495.87 mm |

The initial robot states and first raw actions matched exactly between the
conditions. Horizon 10 also reproduced the previous overfit rollout's full
raw-action and robot-state traces exactly. Checkpoint files matched their
before/after hashes; zero optimizer updates occurred. Dataset replay provenance
was checked again after evaluation.

Horizon 1 reached a smaller sampled finger-midpoint distance but did not produce
a bilateral grasp or stack and did not improve target lift. It required ten
times as many chunk predictions and approximately 1.89 times the wall time;
that wall-time ratio is not a pure GPU inference-speed measurement. The simulator
waits for inference, so neither condition tests real-time deadline compliance.
The distance proxy is not calibrated fingertip clearance. Contacts are sampled
at control boundaries, not every physics substep.

## Grasp/release probes on expert observations

Recorded close begins at frame 107 (3.5667 s into the demonstration), and
release begins at frame 347 (11.5667 s). Probed seven observations around each
transition, using fixed per-observation noise and 50-action predictions.

| Transition / offset | Frame | First-action arm MAE (rad) | Expert gripper | Predicted gripper | Binary decision correct |
|---|---:|---:|---:|---:|---|
| Close -15 | 92 | 0.03532 | 255 | 280.20 | Yes |
| Close -5 | 102 | 0.06194 | 255 | 171.00 | Yes |
| Close -1 | 106 | 0.03737 | 255 | 192.97 | Yes |
| Close 0 | 107 | 0.06287 | 0 | 123.63 | Yes |
| Close +1 | 108 | 0.05257 | 0 | 104.60 | Yes |
| Close +5 | 112 | 0.02579 | 0 | 69.50 | Yes |
| Close +15 | 122 | 0.04099 | 0 | -7.60 | Yes |
| Release -15 | 332 | 0.02364 | 0 | 30.33 | Yes |
| Release -5 | 342 | 0.03804 | 0 | 35.85 | Yes |
| Release -1 | 346 | 0.02552 | 0 | 163.88 | No |
| Release 0 | 347 | 0.04242 | 255 | 66.70 | No |
| Release +1 | 348 | 0.01963 | 255 | 96.88 | No |
| Release +5 | 352 | 0.03602 | 255 | 218.98 | Yes |
| Release +15 | 362 | 0.02960 | 255 | 260.84 | Yes |

The close-transition prediction is only about 3.37 command units below the
127 threshold at its onset: correct for this sampled prediction, but not a
robust-margin finding. Arm-command errors remain present even on the recorded
expert observations (about 0.026–0.063 rad mean absolute error in the close
window). The three release-adjacent binary errors demonstrate imperfect phase
reconstruction despite high aggregate binary agreement in the overfit report.

These probes do not establish the executed gripper decisions when the learned
policy visits different states. In particular, release errors cannot explain
an earlier absence of successful grasp without additional causal evidence.
Changing execution horizon also changes subsequent noise consumption and visited
states, so this single paired test does not isolate all feedback effects.

## Conclusion and recommendation

The completed experiment does not support treating shorter replanning as the
fix for this checkpoint's failed grasp. It also does not justify collecting
more demonstrations solely from the current failures. Recorded expert replay,
exact horizon-10 rerun and checkpoint hash checks provide evidence against an
obvious reset/serialization regression in this diagnostic.

Before another experiment, consolidate the findings and review the remaining
training fit: action reconstruction is still imperfect on expert states. A
fixed-observation/action-chunk fitting check with explicit physical error targets
and inspection of trainable parameters/loss normalization would test this more
directly than further horizon sweeps. No additional experiment or training
extension is authorized automatically by this recommendation.

## Artifacts

Under `outputs/validation/smolvla_replanning_20260915`:

- `result.json`: paired results, probes, hashes and reproduction comparisons.
- `grasp_phase_chunks.npz`: predicted and expert action chunks for all probes.
- `horizon_10/rollout_0.mp4` and `horizon_1/rollout_0.mp4`: paired videos.
- Corresponding `rollout_0_trace.npz` files: raw/applied commands and diagnostics.

No data collection, training, commit, push or upload occurred. Existing
checkpoints/datasets remain unchanged. Work is uncommitted on
`feat/reusable-validation`; bulk collection remains on hold.