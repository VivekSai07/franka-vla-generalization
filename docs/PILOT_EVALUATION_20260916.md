# Pilot tuning evaluation — 2026-09-16

**Completed 2026-09-17:** 120 trials, 4 correct successes (3.33%), no
intervention-free successes, exit 0/no OOM, unchanged final fingerprints.
See [full inspection](FULL_EVALUATION_REVIEW_20260917.md) for independently
replayed outcomes, video checks and the next-step recommendation. Launch notes
below are historical; no baseline evaluation remains running.

User approved evaluation/inference/rollout after training completion.

## Fixed protocol

- Selected checkpoint: `outputs/train/smolvla_pilot_20260916/checkpoints/020000`.
- Exactly 120 independent tuning scenes, canonical order; no final-scene rollouts.
- 40 simulated seconds / 1,200 control ticks maximum; 30 Hz, 20 physics substeps.
- Predict 50-action chunks, execute ten before replanning; count actual predictions.
- Saved train-only processors; precision-preserving FP32 checkpoint loader.
- Strict released-stack criterion sustained for 30 consecutive ticks.
- Binary task success and success without joint/gripper clipping reported separately.
- Nonfinite predicted actions count as failed trials. Infrastructure/provenance
  failures stop the run, without retries, relaxed gates or automatic resume.
- Save two-camera video, raw/applied action/state traces, terminal simulator state,
  per-trial diagnostics and aggregate rates by ordered pair and cube count.
- Seed 20260916 + trial ordinal; deterministic Torch algorithms enabled. CPU OSMesa
  rendering and GPU policy inference in the existing pinned offline image.

## Verification and launch

69 CPU Docker tests passed, including trial selection, streak logic and failure
reporting tests. Simulated outcomes printed by unit tests are **not policy results**.
Static review found no confirmed major correctness defect. Fresh preflight verifies
data/source/runtime, selected model hash/config/processors and all tuning initial
states. Final completion repeats the artifact fingerprints.

- Container: `smolvla-pilot-eval-tuning-120`.
- ID: `613a95ef74fb7a7e9aa53e5362c547ccc25ea2c5a14140148df04685d4876e9f`.
- Output: `outputs/evaluation/smolvla_pilot_tuning_20260916`.
- Network disabled; source/data/checkpoint read-only, evaluation output writable.
- GPU idle and 86 GB disk available before launch.
- Startup confirmed: model loaded and **trial 1/120 started**; no completed
  trial or policy-success rate claimed at this checkpoint.

Read [live result](../outputs/evaluation/smolvla_pilot_tuning_20260916/result.json)
for progress. Each trial writes `episode.json`, `trace.npz` and `cameras.mp4`.
Completion requires all 120 trials, unchanged post-run fingerprints and exit 0.
Detached execution survives SSH closure; the host/Docker must stay running.
Do not edit bound sources in flight, start duplicates, train or upload.