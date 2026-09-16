# Overnight balanced pilot — 2026-09-15

## Verified completion — 2026-09-16

The run is complete; the launch/procedure text below is historical. Inspected
the retained container and persisted result artifacts on September 16:

- Container exited **0**, no OOM, finished **2026-09-16 08:01:57 UTC**.
- Runner and final validator status: **passed**; collection and validation
  subprocess exit codes both 0; replay manifest status **complete**.
- **420 episodes, 221,760 frames, 469 attempts; 300 train / 120 validation**.
- **840/840 replay results passed**: reference and recorded float32 commands
  for all 420 episodes, with no failed results.
- Alignment PASS: mean current-state error **0.021869**, next-state error
  **0.016087**.
- Independent scene banks: **120 tuning / 240 final**. Final validation covers
  planned quotas, labels, command reconstruction, media, replay and bank isolation.
- Accepted demonstration layouts are conditioned on expert success; independent
  scene banks are not. This does not establish learned-policy success.

Evidence: [runner result](../outputs/pilot_overnight_20260915/run.json),
[final validation](../datasets/franka_pilot_20260915/validation_result.json),
[replay result](../datasets/franka_pilot_20260915/replay/replay_result.json),
[final manifest](../datasets/franka_pilot_20260915/replay/manifest.json).
This completion check read existing results; it did not rerun the full validator.

No training or upload was started. Future training must explicitly use the
manifest's 300 training IDs and training-only statistics, not LeRobot's default
all-episode train range/global statistics. Preserve the 120 validation demos and
240 final evaluation scenes from training; use tuning scenes for development.
Use fresh pretrained initialization, never a pre-fix checkpoint resume.

## Authorization and gate

User requested full collection tonight, conditional on corrected preparation
and all validation gates. Preparation passed: 12 episodes, 6,336 frames,
16 attempts, 9/3 split. Full video, row-label, command reconstruction,
scene-bank isolation, reference/float32 replay and alignment checks passed.
Alignment errors: current 0.022387; next 0.016457. No tolerance was relaxed.

## Active collection

- Container: `smolvla-pilot-420-overnight`.
- ID: `2f48e41bd1f833a42a5950784fd1b90bfdeb2db55fdb0b2ec003ebbe86f6e4f1`.
- Dataset: `datasets/franka_pilot_20260915`.
- Budget: **300 training + 120 validation demonstrations**. No preparation
  episodes are counted toward this dataset.
- Plan: `outputs/benchmark_preparation_limits_20260915/pilot_plan.json`.
- Plan digest: `6cb795e04da110bc56343dfedef4c79ceaa683a32e22cd6b5ef0a2ae718775d1`.
- Independent scene bank: `outputs/benchmark_preparation_limits_20260915/pilot_scenes`.
- Image: `smolvla:rtc-offline`, verified ID
  `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
- User: host UID/GID, 348064031:348000513.
- Entrypoint: `validation/run_pilot.py`.

Startup confirmed the runner completed preparation/plan/bank preflight and
entered `collecting`, with the printed target of 420 episodes at the exact root
above. This is evidence of startup, not completion. Approximately 115 GB free
at launch. Recording alone is roughly 10 hours before retries; all-episode
decoding/replay validation adds time. No model performance is guaranteed.

## Completion and failure behavior

The runner invokes the planned collector, then automatically executes the full
pilot validator. Its persistent state is `outputs/pilot_overnight_20260915/run.json`:
`collecting` -> `validating` -> `passed`, or `failed`. Final PASS requires exactly
420 episodes and split counts 300/120, plus every validation gate. A nonzero
collection or validation exit stops the job without retry/resume or upload.

No training or Hub uploads follow completion. Keep the machine awake and Docker
running; detached mode survives terminal closure but not host shutdown/suspend.
No sleep inhibitor was installed. Do not edit hashed collection/validation
source or assets while this run is active. Do not start another collector.

```bash
docker logs --tail 20 smolvla-pilot-420-overnight
docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' smolvla-pilot-420-overnight
```

Before reporting final completion, inspect the runner state, container exit,
dataset `meta/info.json` and `validation_result.json`. Report exact counts and
directory, or the failure with preserved artifacts. The retained container
provides logs after exit. Code/doc changes remain uncommitted; no push occurred.