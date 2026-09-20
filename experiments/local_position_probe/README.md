# Local-position boundary screen

Task 1 implementation of the approved 2026-09-19 plan. No training, collection,
new model, checkpoint transfer, resampling, retry, resume or budget override.
Only main may launch `python -m experiments.local_position_probe.runner --run`,
after independent spec/quality review and a fresh real CPU `--check`.

## Contract and execution

- 17 conditions: control, then red/green × 5/10mm × +X/-X/+Y/-Y.
- Two seeds, 20260918 and 20260923, in each condition: 34/model, exactly 68 total.
  SmolVLA block first, X-VLA second; both controls precede each model's offsets.
- Baseline ordinals 1 and 6 are preserved, including source episode 7. Study IDs
  are outer metadata, never fake episode IDs. The seed-23 SmolVLA control is a
  known failure and must reproduce as a failure, byte-exactly across all arrays.
- Final 005000 weights only, original FP32-aware loaders, saved processors,
  one model load per block. The original `run_trial` implementations are called
  unchanged: native action queue reset, 40s/1200 ticks, 30Hz/20 substeps, strict
  30-stable-tick score, original model-specific autocast behavior. Baseline CPU
  thread counts (SmolVLA 8, X-VLA 4), disabled cuDNN benchmarking, deterministic
  algorithms and CUBLAS workspace configuration are restored before inference.
- Entire bank, per-file hashes, plan and CPU proof saved before model loading.
  Both models receive the same 17 files, built from the sealed full reset.
  Only one cube free-joint XY scalar changes; all integration state, including
  warmstart, qvel, controls, time, quaternion and other bodies is preserved.
  Geometry is forwarded without stepping or resettling. Bounds, clearance,
  settled Z, camera extrinsics and initial robot/cube contacts are checked.
- Control comparison requires all NPZ keys, dtypes, shapes and array bytes plus
  all explicitly listed episode semantics. Only wall time and artifact IDs are
  excluded; encoded-video byte equality is not required. Nonfinite/reproduction/
  seed/infrastructure failures stop immediately; ordinary task timeouts continue.
- Exclusive fresh output: outputs/diagnostics/local_position_probe_20260919.
  An existing path is refused except a truly empty dedicated bind mount, which
  is claimed exclusively. Journal start/end records are append-only and fsynced.
  Partial artifacts remain; partial summaries suppress success rates.
- Fixed 20GiB disk floor, 12GiB device-free GPU floor before model loading, and
  4-hour process alarm including admission/loading. No overrides or extensions.
  Disk is checked between trials. Device-free VRAM is not a peak-use guarantee.
- Explicit package/spec/plan hashes and old admission/import/asset bindings,
  final checkpoint inventories and bounded original control trace/result hashes
  are checked before and after policy execution. Old final-scene banks are not
  read. Old data are hashed by existing admission; no new data or expert execution.
  The two original evaluated reports plus four control episode/trace pairs are
  pinned to ten fixed hashes, not accepted solely because current files agree.
  Saved bank/plan/proof bytes are also rechecked before completion.

## CPU verification and telemetry

Use the pinned image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`;
CPU: runc, no network/GPU, read-only root and repository, offline caches in tmpfs.
Only outputs/implementation/local_position_probe_20260919 may be writable for
CPU evidence. `--check` itself writes nothing, constructs no policy or expert,
steps nothing, renders nothing and requires no GPU. Run tests with
`python -m unittest discover -s experiments/local_position_probe/tests -v`.
Tests use real nonstepped resets and original baseline arrays; scheduler tests
replace only model/trial/hardware seams and do not exercise actual policy scoring.

The local sampler uses a fixed bounded `nvidia-smi` subprocess (3s timeout),
every 5s with UTC/monotonic timestamps and current phase. It records UUID/name,
devicewide shared-GPU utilization/memory/power/temperature, not per-process
attribution. Missing/nonfinite fields become null with a reason. Sampling errors
are nonfatal and recorded; the thread is stopped/joined on all exits. No RNG,
model tensors, environment dump, credentials or process command lines are read.
CPU user/system time, wall time and peak RSS are recorded separately.
Cloud W&B logging is deferred until completed results exist; local raw samples
are authoritative. No networking or logger setup is needed for implementation.

## Interpretation

Two development-selected seeds per condition form a descriptive boundary screen,
not precise success-rate estimation, a new acceptance threshold or evidence of
general workspace generalization. Controls are separate from perturbations.
Reports include per-condition/object/magnitude denominators, paired outcomes and
stage failures derived from the unchanged stored stage metrics. No automatic
training or varied-data collection follows this screen.