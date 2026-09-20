# Fixed-layout baseline implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build and test the approved 24-execution fixed-layout data pipeline without launching collection until its storage gate is met.

**Architecture:** A new isolated experiment package owns the immutable plan, reset binding, recorder and validator. It reuses the unchanged pose-snapshot-v2 expert and small existing media/timing helpers, never old hardcoded paired schedules. The original source episode is the only source of initial state.

**Tech Stack:** Existing Python3.10, NumPy, MuJoCo, Pinocchio, LeRobot0.4.4, PyAV in pinned offline Docker; unittest for CPU tests. No new dependencies.

## Global constraints

- Approved specification: `docs/superpowers/specs/2026-09-18-fixed-layout-baseline-design.md`.
- User selected subagent-led implementation and CPU testing on2026-09-18, explicitly keeping collection blocked until30GiB free. No actual experts, policy inference, physics-stepping preparation or collection in this implementation session.
- Work in existing `feat/reusable-validation` checkout per existing user preference, new package only. No old bound source/test/report changes.
- All Python inside pinned image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`, runc, network none, GPU disabled, OSMesa, read-only repo, temporary caches.
- Exactly24expert attempts,20train/4same-geometryvalidation; preparation first3 within24. Two control replays, not regenerated expert replay. First failure stops without resampling/retry/resume. Exactly same episode7 integration reset for all.
- State9/action8/two512RGB AV1/yuv420p,30Hz/600Hz. Unchanged snapshot-v2 expert, gripper schedule/threshold, waypoint jitter and strict final stack metric.
- 30GiB free reserve before output creation and between episodes; 3h wall cap enforced within episode/validation as well as at boundaries. No automatic reserve override in CLI.
- No model/optimizer/training code, dependency downloads, artifact deletion, secret-file access or pushes. Main agent alone stages/commits under configured user identity.
- Task reports/ledger under ignored `outputs/implementation/fixed_layout_20260918/`, not the future collection root. Never alter prior evidence.

## File map

- `experiments/fixed_layout/contract.py`: pure plan, seeds, schema/constants, budget/schedule/gate helpers.
- `experiments/fixed_layout/source.py`: pinned source episode/hash/reset restoration, explicit source/runtime fingerprint and preflight.
- `experiments/fixed_layout/record.py`: exclusive collection and failure-safe per-tick/episode persistence, schedule and resource checks.
- `experiments/fixed_layout/validate.py`: independent persisted-row/media checks, two control replays, contact/lift metrics, dataset fingerprints/duplicate report.
- `experiments/fixed_layout/__init__.py`: package marker.
- `experiments/fixed_layout/tests/`: regression tests separated by component.
- `experiments/fixed_layout/README.md`: exact safe preflight/collection commands and limitations. Collection command documented but not run.

### Task 1: Immutable contract and pinned reset preflight

**Files:** contract.py, source.py, __init__.py, tests/test_contract.py, tests/test_source.py.

**Consumes:** approved spec; source NPZ/manifest and old benchmark helpers.
**Produces:** `make_plan() -> dict`, `validate_plan(plan) -> None`, constants `ROOT`, `OUTPUT`, `MIN_FREE_BYTES`, `MAX_SECONDS`; `load_reset(expert) -> dict[str, ndarray]`; `provenance() -> dict`; `preflight() -> dict`.
`load_reset` returns initial_state, qpos, initial_ctrl, body_xpos, camera_xpos and camera_xmat derived by nonstepping restoration and checked against old trace geometry.
All report arrays must become JSON-safe explicitly; no loss of NPZ precision.

- [ ] Write failing contract tests before implementation. Concrete cases:

```python
plan = make_plan()
self.assertEqual(len(plan['episodes']), 24)
self.assertEqual(plan['train_ids'], list(range(20)))
self.assertEqual(plan['validation_ids'], list(range(20, 24)))
self.assertEqual(plan['preparation_ids'], [0, 1, 2])
self.assertEqual(len({e['seed'] for e in plan['episodes']}), 24)
changed = copy.deepcopy(plan)
changed['episodes'][0]['pick'] = 'green'
with self.assertRaises(ValueError):
    validate_plan(changed)
```

- [ ] Independently reconstruct SHA256 seed formula; test split/order/seed mutations, unknown fields, bool-vs-int type spoofing, fixed source SHA and same geometry declaration.
- [ ] Run tests in Docker, capture expected RED before adding minimal implementation. No host Python.
- [ ] Implement immutable canonical contract and explicit source binding. Hash actual NPZ before loading; verify manifest entry7 task/roles/frame count/source SHA. Restore on fresh MjData without mj_step; verify full qpos first row, initial integration state identity, body/camera geometry and finite arrays. Never resettle.
- [ ] Source tests fail on changed source hash, manifest/task mismatch and corrupted initial geometry; prove reset check never calls mj_step. Use actual source only in readonly no-step test. Synthetic fixtures may inject paths/metadata through test seams, never production CLI overrides of canonical source.
- [ ] Implement provenance for imported old collection/robot/media/source helpers and all new Python files plus the approved spec. Preflight reports actual free disk, storage eligibility separately from reset/source validity, no expert call, no output creation and no render.
- [ ] Run focused GREEN tests and current baseline suites in separate Python processes to avoid same-named unittest module collisions. Return exact commands/counts, source hashes and concerns. Main agent reviews and commits this task; do not self-commit.

### Task 2: Recorder, independent validation and safe failure lifecycle

**Files:** record.py, validate.py, README.md, tests/test_record.py, tests/test_validate.py; minimal same-package task1 changes only if needed for integration.

**Consumes:** task1 APIs and source/reset/provenance contract, unchanged snapshot Expert, PairedDataset/FEATURES/ENCODING and existing validate_trace/media helpers when appropriate.
**Produces:** `collect()` with no unsafe CLI overrides; `python -m experiments.fixed_layout.record --preflight` and `--collect`; `require_fixed_layout_pass(root)` that verifies fresh source/runtime and exact dataset fingerprint, never just a saved PASS.

- [ ] Write failing lifecycle tests with fake expert/media dependencies (no actual trajectory/physics). Assert all24 seeds/plan frozen before first execution, exact same initial state every slot and copied per call, preparation indices checked before fourth execution, and no24-slot completion on first failed task/admission.

```python
# Production mutation caught: swallowing a failed attempt and continuing.
with self.assertRaises(RuntimeError):
    run_schedule(plan, initial, execute_fails_on_second, validate, journal)
self.assertEqual(executed_ids, [0, 1])
self.assertFalse(report['training_ready'])
```

- [ ] Add RED tests: disk below30GiB creates no output and invokes no expert; preexisting root rejected; deadline expires in sink/replay/decode; state/plan/source tampering blocks next episode; incomplete frames preserved even on expert/media exception; original exception survives finalization error; no validation failure permits next expert.
- [ ] Implement exclusive root and source+plan snapshots, source checks before every expert, live deadline/resource checks, one attempt journal and per-tick durable evidence. Save original returned pre-step RGB/state/action alongside traces, including failed attempts, without unbounded RAM reliance. Terminal snapshot on interruption must be last actually known state, not fabricated expert output; clearly mark unavailable fields. Do not let dataset finalization mask original failure.
- [ ] Implement independent per-episode validation using persisted float32 rows and actual trace geometry. Assert shape/dtype/finite values, instruction/task indices, consecutive row/frame indices, timestamps, exact action/state equality to trace and governor reconstruction. Fully decode videos with original interval rules/codec checks and deadline.
- [ ] Implement two replays of persisted controls, no `expert.run` inside validator. Each captures actual post-step contacts/lift/stack. Reference full-qpos<=1e-5, float32 policy-state<=1e-3, successful stable stack required both; original recorded geometry/time compare in reference. Count replay modes uniquely; no missing/duplicate result accepted. Reuse proven apply_action/step_control/stack_metrics without changing thresholds.
- [ ] Gate original/reference traces for>=30closed ticks,>=5bilateral consecutive contacts and>50mm lift while contacting. Regression fixtures reject never-close/one-tick-close, transient contact, lift without contact and nonfinite/wrong-role data. Hand-distance is descriptive, never substitutes for contact.
- [ ] Final dataset validation:24episodes x528frames,20/4 split, task1/two cameras and no privileged policy fields; full fingerprint/source inventory unchanged. Save hashes and pairwise action RMS differences with units, separate gripper/arm where useful. Exact duplicate execution sequences must block training admission/design and remain explicit; no silent dedup or replacement.
- [ ] `require_fixed_layout_pass` must reject partial roots, altered reports/data/spec/source, stale PASS, duplicate trajectories and missing required media/replay outcomes. Test real small synthetic on-disk fixtures rather than exclusively mocked gates.
- [ ] README distinguishes prepared software from unexecuted expert gates and documents Docker paths, reserved disk, pending collection approval after resource restoration,3h timeout, no training. Include safe shell command for user reference only; do not run collection.
- [ ] Run RED/GREEN evidence and full focused suites; run real readonly no-step preflight only. Main agent reviews implementation against spec and code-quality criteria, fixes critical issues, independently reruns verification, then commits with short user-only message.

## Verification commands

Use Docker entrypoint Python and a subprocess per test suite:

```python
subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                '-s', 'experiments/fixed_layout/tests'], check=True)
```

Container environment: MUJOCO_GL=osmesa, HF_HOME=/tmp/huggingface,
XDG_CACHE_HOME=/tmp/cache, HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1,
NVIDIA_VISIBLE_DEVICES=void, CUDA_VISIBLE_DEVICES empty,
PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1.16GiB RAM/4CPUs,
read-only root and repository at/vla, tmpfs/tmp4GiB. Write generated test logs
only to ignored implementation output directory, not old report files.

## Execution ledger

- [x] Exact written scope approved through user question: implement/test now,
  subagent-led; no collection until reserve met, no deletion or training.
- [x] User's existing in-place feature-branch preference retained; no worktree copy
  needed, preserving canonical/vla paths and scarce disk.
- [ ] Task1 reviewed and committed.
- [ ] Task2 reviewed and committed.
- [ ] Final independent verification and status handoff.

Self-review: no task launches real collection; code paths implement future
24/48 budgets, while this session executes software tests and nonstepping reset
preflight only. User-selected implementation route supersedes another mode prompt.
