# Reusable validation implementation plan

> For agentic workers: execute sequentially with tests and runtime gates;
> consult executing-plans before continuing. Do not start bulk collection.

**Goal:** Six replayable demonstrations and a bounded SmolVLA learning-pipeline
validation with a durable evidence report.

**Architecture:** Keep the current collector's controller/physics loop. Add
optional benchmark recording hooks and a reusable simulation/replay utility.
Use separate replay and learning-smoke commands; no changes to legacy training.

**Tech stack:** Existing smolvla:rtc-offline image, LeRobot0.4.4,
MuJoCo3.13.0, NumPy, unittest, PyTorch2.7.1+cu128.

## Global constraints

Use the [design](../specs/2026-09-15-reusable-validation-design.md).
Work on feat/reusable-validation; sixepisodes/30attempts/100updates/batch2/
two held-out rollouts. Docker only. Preserve existing data and credentials.

## Tasks and file boundaries

- [x] 1. `tests/test_benchmark.py`: failing tests for deterministic condition
  scheduling, MuJoCo integration-state round-trip and invalid actions. Run
  `python -m unittest discover -s tests -v` in the existing image with project
  bind-mounted at/vla; new helper absence must fail before implementation.
- [x] 2. `data_collection/benchmark.py`: implement seed/condition scheduling,
  capture/restore via mjSTATE_INTEGRATION, source/asset hashes, shared command
  stepping, strict geometric/release scoring and sidecar/manifest lifecycle.
  Re-run all unit tests; incorrect state length,NaN commands and duplicate
  output must fail closed.
- [x] 3. `data_collection/collect_libero_dataset.py`: add optional explicit
  root/seed/benchmark/attempt limit, preserving non-benchmark control behavior.
  Record pre-observation state/times and post-command controls, attempt ledger,
  per-success sidecar, training/validation split and finalization marker.
- [x] 4. `data_collection/replay_benchmark.py`: load actual LeRobot actions,
  restore source-matched simulation, replay reference and recorded actions,
  compare state trace, strict success and gripper fidelity. Returnnonzero on
  any failure. Run new six-episode collection, replay and unchanged alignment
  verifier in order. Do not weaken thresholds to make a failing gate pass.
- [x] 5. `validation/smolvla_smoke.py`: read actual installed policy/processor
  interfaces before implementing. Download revision-pinned base snapshot to
  ignored cache, inspect metadata and load exact architecture/custom schema;
  build fresh processors with train-only statistics. Train100steps, save in
  fresh ignored output, reload and compare seeded predictions. Evaluate held-
  out initial states only with shared control/scoring. If download or dependency
  access blocks, stop at that boundary and document exact missing artifact.
- [x] 6. `docs/BENCHMARK_STATUS.md` and a validation evidence report: list
  exact dataset root/image/source hashes, counts, provenance,replay errors,
  alignment metrics,training/reload/rollout evidence and unexecuted gates.
  Run source diagnostics and inspect diff. Do not upload/push or claim benchmark
  quality from a small integration run.

## Success criteria

Six saved and finalized episodes, manifest4train/2validation, both replays
passing for all episodes, original alignmentPASS; finite training with actual
weight updates, complete local save/reload agreement and two bounded rollouts
with honest success/failure metrics. A dependency block is a partial outcome,
not a passed end-to-end gate.