# Balanced pilot implementation plan

> For agentic workers: execute task-by-task using the execution workflow with
> test-first gates. See the approved balanced-pilot specification before continuing.

**Goal:** Implement reproducible quotas and independent scene banks, generalize
validation, then run only the 12-episode preparation gate.

**Architecture:** A pure schedule module defines immutable slots/splits/seeds.
The existing collector consumes slots without changing controller/timing. A
separate scene-bank generator and pilot validator enforce plan, replay, video
and provenance integrity. Existing six-episode diagnostic mode remains separate.

**Tech stack:** Python unittest, NumPy/MuJoCo/LeRobot inside smolvla:rtc-offline.

## Global constraints

Use timing-v2, 30 Hz/600 Hz, 9-D state/8-D absolute commands, two 512px views.
Preserve old artifacts. No optimizer, upload or 420-episode launch in this step.
No checkpoint resume. Keep rejected attempts and incomplete outputs visible.
The authoritative [specification](../specs/2026-09-15-balanced-pilot-design.md)
sets exact quotas and failure limits.

## Tasks

- [x] Schedule and tests: `data_collection/pilot_plan.py` exposes
  `make_plan(mode, seed)` for `pilot` (420 slots) and `preparation` (12 slots),
  `validate_plan(plan)`, `slot_seed(slot, retry)`, `plan_digest(plan)`.
  Each slot has slot_id, split, pair_id, pick/place, colours and ten seeds.
  Tests count per-pair/count quotas, enforce identity-balanced one-distractor
  cases, reject mutations and check deterministic collision-free retry streams.
- [x] Collector integration: add `--plan` immutable plan input, validate before
  creating root, derive count/splits from it, retry same slot ten times maximum,
  record initial geometry hash and outcomes for every attempt. Fail on invalid
  expert commands. No continue/resume on existing roots. Preserve source before edits.
- [x] Scene banks: generate 120 tuning / 240 final states from plan seed streams,
  using actual settling/geometry initialization and no success filtering. Save
  per-scene state and hashes, detect duplicate geometry, write manifest last.
- [ ] Generalized validation: support arbitrary counts, exact plan acceptance,
  full video decoding, finite schemas/commands, split/seed isolation and scene
  duplicate checks. Run all-episode reference/float32 replay and original alignment.
  Only publish final validated marker after every gate; bind content/runtime/source.
- [ ] Preparation run: generate plans/banks, inspect tests and code review, launch
  detached 12-slot collector; on normal exit run validator. Report gate failures
  without weakening them. Preserve evidence and update BENCHMARK_STATUS.

## Test commands (inside Docker)

`python -m unittest discover -s tests -v` must pass before collection.
`python data_collection/collect_libero_dataset.py --help` must expose plan mode.
Validator must reject edited quota, duplicated state, truncated video, mismatched
FPS, stale hash and unfinalized recording. No expensive run should be necessary
to exercise these rejection cases.

## Completion and launch boundary

Execution note: 44 tests passed and independent banks generated/verified.
The preparation job stopped after three saved episodes on a real expert joint
target bound violation; see `docs/PILOT_PREPARATION_20260915.md`. Generalized
validator is implemented but its full real-data gate remains unexecuted because
the recording is incomplete. Do not mark preparation or pilot-ready complete.

Report exact preparation count/split, attempts, checksums, scene-bank counts,
video frames, replay/clock/alignment evidence and any failed gate. Collection
is not validated simply because its process exits zero. Present the separately
verified 300+120 plan before requesting the actual pilot launch decision.