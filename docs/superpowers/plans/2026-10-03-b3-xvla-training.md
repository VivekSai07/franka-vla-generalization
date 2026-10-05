# B3 X-VLA training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train X-VLA on the sealed B3 robosuite dataset for 156,000 updates (5.0 epochs), from the base checkpoint, through the repo's `--check` → `--preflight` → `--train` gates.

**Architecture:** A new training package `experiments/xvla_b3_robosuite/`, forked from `experiments/xvla_b2_distractors/` (contract + runner + tests + README). The runner's training loop is kept; the contract changes what it binds (the B3 seal and provenance instead of B2's), what it reads (560 episodes, 10 instructions, split from the plan), and the horizon. Training needs no simulator and runs in the admission image on the GPU.

**Tech Stack:** Python 3.10, LeRobot 0.4.4 (X-VLA), PyTorch 2.7.1+cu128, admission image, one RTX 6000 Ada (48 GB).

**Spec:** [../specs/2026-10-03-b3-robosuite-cubes-design.md](../specs/2026-10-03-b3-robosuite-cubes-design.md), section 3. Plan 2 of 3; plan 1 (collection) is done and sealed; plan 3 (`2026-10-03-b3-xvla-evaluation.md`) can be implemented while this one trains.

**Branch / worktree:** `feat/b3-robosuite-cubes`, worktree `.worktrees/feat-b3-robosuite-cubes`. Load the skills `forking-an-experiment-package`, `training-a-policy` and `launching-a-run` before Task 1.

## The parent dataset (verified 2026-10-03)

| field | value |
|---|---|
| root | `outputs/b3_robosuite_data/b3_robosuite_20261003/` (main checkout) |
| seal fingerprint | `e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0` |
| `seal.json` | `fingerprint`, `provenance_sha256` `d99d2c62…2de7ba`, `plan_sha256` `ea6b572f…50140f`, `substitutions_sha256` `44136fa3…aff8a`, `image` (robosuite image), `min_clearance_m` 0.08, `episodes` 560 |
| episodes | 0–499 training, 500–559 validation (from `plan.json` `train_ids` / `validation_ids`) |
| frames | 528 per episode; 295,680 rows; `meta/info.json` `splits` is `{"train": "0:560"}` — **the split comes from the plan, not from LeRobot** |
| tasks | 10 instructions (`b3_robosuite_data.plan.INSTRUCTIONS`); LeRobot numbers them by first appearance, not in the plan's sorted order, so map by string |
| `validation.json` | `episodes` 560, `train` 500, `validation` 60, `per_cell_train` 25, `per_cell_validation` 3, `substituted_slots` 0, `held_out_pairs_collected` 0 |

Independently re-verified before this plan was written: the fingerprint recomputes equal; `provenance()` (131 files) equals `provenance.json`; `plan.json` equals a fresh `make_plan()`; every episode has 528 rows and its planned instruction string; the dataset loads in the admission image.

## Global Constraints

- **The B3 dataset and everything in its provenance are frozen**: `experiments/b3_robosuite_data/**`, `experiments/robosuite_scene/*.py`, the shared files listed in `b3_robosuite_data.contract.SHARED_FILES`, `src/**`, the B3 spec and the collection plan. Editing any of them breaks the binding this package checks.
- Do not edit `experiments/xvla_b2_distractors/**`, `experiments/xvla_single_task/**`, `experiments/xvla_trial/**` or any other existing package. Override locally (see `training-a-policy` §1 and the B2 contract's own `learning_rate` docstring for the pattern).
- Image: admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` for every gate. Never the robosuite image here.
- Recipe identical to B2 except the horizon and the data: effective batch 8 (microbatch 4 × accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32 parameters with bf16 autocast, `optimizer_storage='final_only'`, seed 20260917, no resume, from the base bundle in `.cache/xvla_trial/bundle`.
- **Horizon:** `TOTAL_UPDATES = 156_000` (= 500 × 499 = 249,500 complete starts × 5.0 epochs / 8, rounded). **Checkpoints:** `[1, 31200, 62400, 93600, 124800, 156000]` (update 1 and each whole epoch).
- Output: `outputs/train/xvla_franka_stacking_b3_robosuite_v1` (+ `_admission`, `_preflight`) in the main checkout. Never overwrite or resume; a failed run restarts from zero in a new directory only with the user's consent.
- Offline metrics (loss, MAE) are never used to choose a checkpoint, stop early, or claim progress.
- Commits: user identity, no trailers, short messages; never commit outputs. The user pushes.
- The machine is shared: check `nvidia-smi` (≥ 28 GiB free) and `df -h` (≥ 33 GiB free) before `--preflight` and `--train`.

## Review Focus

1. **Validation episodes leak into training or statistics.** `test_contract` asserts the sampler yields exactly the 249,500 starts of episodes 0–499 and that the statistics use exactly rows 0–263,999.
2. **Task index mapped by number instead of by string.** `test_contract` builds a fake task table whose numbering differs from the plan's sorted order and asserts every episode still gets its planned instruction.
3. **A held-out instruction reaches training.** `test_contract` asserts the approved set is exactly the 10 trained instructions and that `approved_instruction("pick up the yellow cube and place it on top of the red cube")` raises.
4. **Fork residue.** `test_fork_hygiene` asserts no file in the package names `b2_distractor`, `xvla_b2_distractors`, `b2_distractor_20260928`, `B1_` or `50000`, and the symtable dead-name check passes.
5. **Disk or GPU taken by someone else mid-run.** The runner's resource guard runs before preflight and train (inherited); `test_runner` keeps B2's guard test with this package's numbers.

## File map

| file | change from the B2 fork source |
|---|---|
| `experiments/xvla_b3_robosuite/__init__.py` | empty |
| `experiments/xvla_b3_robosuite/contract.py` | parent binding, data reader, instructions, horizon, schedule, paths, `SOURCE_FILES` |
| `experiments/xvla_b3_robosuite/runner.py` | only what the contract change forces (names, resource numbers); the loop is untouched |
| `experiments/xvla_b3_robosuite/README.md` | new text (it is in `SOURCE_FILES`) |
| `experiments/xvla_b3_robosuite/tests/test_contract.py`, `test_runner.py`, `test_fork_hygiene.py` | forked, repointed, numbers re-derived from the contract |

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
PARENT=outputs/b3_robosuite_data/b3_robosuite_20261003
OUT=outputs/train/xvla_franka_stacking_b3_robosuite_v1
mkdir -p $PARENT .cache outputs/train                       # empty mount points inside the worktree
cpu()  { docker run --runtime runc -e NVIDIA_VISIBLE_DEVICES=void "$@"; }   # add --rm yourself for short runs; never with -d
gpu()  { docker run --runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 "$@"; }
FLAGS="--network none --read-only --user $(id -u):$(id -g) --tmpfs /tmp:rw,size=32g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/vla -v $PWD:/vla:ro -v $MAIN/$PARENT:/vla/$PARENT:ro -v $MAIN/.cache:/vla/.cache:ro \
  -w /vla --entrypoint python"
# tests:  cpu --rm $FLAGS $OLD -m unittest discover -s experiments/xvla_b3_robosuite/tests -t experiments/xvla_b3_robosuite/tests
```

`/tmp` is 32 GB because the HuggingFace arrow cache of a 295,680-row dataset lands there (FAILURES #41 is the collection-side version of this). Gate commands mount their own output directory read-write at `/vla/$OUT…` (see Tasks 4–6).

---

### Task 1: the contract

**Files:**
- Create: `experiments/xvla_b3_robosuite/__init__.py`, `contract.py` (from `experiments/xvla_b2_distractors/contract.py`)
- Create: `experiments/xvla_b3_robosuite/tests/test_contract.py` (from B2's)

**Interfaces:**
- Produces every public name of the B2 contract with the same signature (the runner and the evaluation package call them), with these values:

```python
PARENT = ROOT/'outputs/b3_robosuite_data/b3_robosuite_20261003'
DATA = PARENT/'dataset'
OUTPUT = ROOT/'outputs/train/xvla_franka_stacking_b3_robosuite_v1'   # SCRATCH, ADMISSION as in B2
SEAL = 'e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0'
TRAIN, VAL, SEED = tuple(range(500)), tuple(range(500, 560)), 20260917
TRAINING_ROWS, COMPLETE_STARTS = 500*528, 500*499          # 264_000, 249_500
VALIDATION_ROWS, TOTAL_FRAMES = 60*528, 560*528             # 31_680, 295_680
INSTRUCTIONS = b3_robosuite_data.plan.INSTRUCTIONS          # the 10 trained instructions, sorted
TOTAL_UPDATES = 156_000
def checkpoint_updates(preflight): return [1, 2] if preflight else [1, 31200, 62400, 93600, 124800, 156000]
```

- `current_binding()` for B3 (replaces B2's body):
  1. `seal.json` equals exactly the seven-field record in *The parent dataset* (fingerprint `SEAL`, the three digests, the robosuite image, 0.08, 560). Store the expected record as a constant `SEAL_RECORD`.
  2. `plan.json` passes `b3_robosuite_data.plan.validate_plan`, and `p.digest(plan) == SEAL_RECORD['plan_sha256']`.
  3. `substitutions.json` is `{}` and its digest matches.
  4. A fresh `b3_robosuite_data.contract.provenance()` equals `provenance.json`, and `p.digest(provenance) == SEAL_RECORD['provenance_sha256']` (the same digest `b3_robosuite_data.validate.seal` used).
  5. `validation.json` has the field values in *The parent dataset* table.
  6. Returns `dict(dataset_fingerprint=SEAL, seal_sha256=…, plan_sha256=…, provenance_sha256=…, validation_sha256=…, b3_provenance=<the dict>, sources=source_hashes(), assets=require_assets(), required_image=IMAGE)`.
  The full fingerprint recomputation is a separate function, `parent_fingerprint() -> str` (returns `b3_robosuite_data.validate.fingerprint(PARENT)`), called once by `--check` (Task 2), not on every binding call.
- `load_data()`: `info.json` must have `total_episodes=560, total_frames=295680, total_tasks=10, fps=30, splits={'train': '0:560'}`; episode identity, 528-frame boundaries and instruction-by-string checks as in B2 over all 560 episodes.
- `planned_instructions()`: the `instruction` of each of the 560 plan episodes, in episode order, from `PARENT/'plan.json'`.
- `learning_rate(update)`: B2's local schedule over `TOTAL_UPDATES` (500-update linear warm-up to 1e-4, cosine to the 1e-5 floor); `make_config` stretches `scheduler_decay_steps` to `TOTAL_UPDATES`.
- `recipe(preflight)`: B2's dict with `purpose='xvla_b3_robosuite_preflight' | 'xvla_b3_robosuite'` and `max_optimizer_updates=2 | TOTAL_UPDATES`.
- `SOURCE_FILES`: B2's list with the B2 data and B2 training package entries replaced by `experiments/b3_robosuite_data/{__init__,plan,contract,validate}.py` and this package's own files (contract, runner, README, three test files, `__init__`). Then make it exact: Task 3's test requires every repo module loaded by importing the contract and the runner to be listed, and nothing listed that is not loaded or this package's own.
- Drop the B1 calibration constants and everything that only existed to compare with B1.

- [ ] **Step 1: Fork the test file first.** Copy B2's `test_contract.py`, repoint every import to `experiments.xvla_b3_robosuite`, and change every literal to the values above, **derived from the contract wherever the contract has the value** (e.g. `c.TOTAL_UPDATES`, `len(c.TRAIN)*499`), never re-typed. Replace B2's parent-binding tests with B3 versions: the real parent binds; a tampered `seal.json` field, a changed `plan.json`, a non-empty `substitutions.json`, a drifted provenance entry and a wrong `validation.json` value each raise (use a temporary copy of the small JSON files, never write into the real parent). Add the three tests from Review Focus 1–3.
- [ ] **Step 2: Run it** (`cpu --rm $FLAGS $OLD -m unittest discover …`) → FAIL: the package does not exist yet.
- [ ] **Step 3: Write `contract.py`** per the interface block.
- [ ] **Step 4: Run** → `test_contract` PASS. Paste the output in the task report.
- [ ] **Step 5: Commit** — `git commit -m "Add the B3 X-VLA training contract"`.

---

### Task 2: the runner

**Files:**
- Create: `experiments/xvla_b3_robosuite/runner.py` (from B2's), `tests/test_runner.py` (from B2's)

**Interfaces:**
- Consumes the Task 1 contract. Produces the CLI `python -m experiments.xvla_b3_robosuite.runner --check | --preflight | --train`.

- [ ] **Step 1: Fork the test** with imports repointed and numbers derived from the contract. Keep B2's tests that pin behaviour: the horizon comes from `recipe()`, no hardcoded update count; CLI has no resume/override; resource guard (15 GiB preflight, 33 GiB train, 28 GiB GPU — 6 checkpoints, the same count as B2: 5 × 3.3 + 9.9 = 26.4 GB, so B2's 33 GiB still leaves 6.6 GiB); reload markers; Adam round-trip; synthetic two-update training writes `step_`-prefixed checkpoints; dispatch never launches evaluation.
- [ ] **Step 2: Run** → FAIL (module missing).
- [ ] **Step 3: Write `runner.py`**: B2's file with `from . import contract as c` resolving to this package and every B2-specific string changed. One addition: `check()` calls `c.parent_fingerprint()` before anything else, raises `ValueError('B3 parent fingerprint differs from the seal')` unless it equals `c.SEAL`, and records `parent_fingerprint_recomputed: true` in the admission report. Add a `test_runner` test with `parent_fingerprint` patched to a wrong value that asserts `check()` raises and writes no admission. Do not change the training loop, the checkpoint writer, the reload verification or the resource checks beyond their numbers.
- [ ] **Step 4: Run** → `test_runner` PASS (and `test_contract` still PASS).
- [ ] **Step 5: Commit** — `git commit -m "Add the B3 X-VLA training runner"`.

---

### Task 3: README and fork hygiene

**Files:**
- Create: `experiments/xvla_b3_robosuite/README.md`, `tests/test_fork_hygiene.py` (from B2's)

- [ ] **Step 1: Test first.** Fork B2's `test_fork_hygiene.py`: symtable dead-name check over both modules; nothing imports or patches `experiments.xvla_b2_distractors`; no residue names (Review Focus 4); the schedule is the six updates above and ends at `TOTAL_UPDATES`; `learning_rate` spans the full horizon (1e-4 at update 500, 1e-5 at `TOTAL_UPDATES`); 10 instructions expected and every one appears in the training split; the task-index map rejects a single-task table. Add `test_source_files_are_exactly_the_loaded_modules`: import `contract` and `runner` in a fresh interpreter (subprocess), collect every loaded module whose file is inside the repo, and assert that set equals the repo entries of `SOURCE_FILES` minus this package's tests and README.
- [ ] **Step 2: Run** → FAIL until the README exists and `SOURCE_FILES` is exact.
- [ ] **Step 3: Write `README.md`**: what is trained on what (dataset path, seal fingerprint), the recipe and horizon with the 5-epoch arithmetic, the gate sequence with the commands from *Running things*, costs (about 36 h; about 30 GB), "offline metrics never select a checkpoint", and "this README is in `SOURCE_FILES`: finish it before `--check`". Fix `SOURCE_FILES` until the test passes.
- [ ] **Step 4: Run the whole suite** → all three test files PASS. Then grep the package for `b2`, `B2`, `50000`, `200` and read every hit.
- [ ] **Step 5: Commit** — `git commit -m "Document the B3 training package and pin its sources"`. **From this commit on, nothing in `SOURCE_FILES` is edited until training finishes.**

---

### Task 4: `--check` (CPU admission, about 1–3 h)

- [ ] **Step 1:** Pre-launch checklist, section 1 of `.github/skills/launching-a-run/SKILL.md`, items 1–12. Record the worktree commit (`git log --oneline -1`).
- [ ] **Step 2:** Create the host directory and the worktree mount point, then launch detached:

```bash
mkdir -p $MAIN/${OUT}_admission ${OUT}_admission
cpu -d --name b3-train-check $FLAGS -v $MAIN/${OUT}_admission:/vla/${OUT}_admission:rw \
  $OLD -m experiments.xvla_b3_robosuite.runner --check
```

- [ ] **Step 3:** Poll every 10–15 minutes (`docker ps -a --filter name=b3-train-check`, `docker logs b3-train-check 2>&1 | tail -5`). On completion read `${OUT}_admission/result.json`: `status: passed`, `parent_fingerprint_recomputed: true`, `training_rows: 264000`, `complete_training_chunks: 249500`, the train/validation lists, statistics present. Paste the key fields in the task report.
- [ ] **Step 4:** On failure: read the error, load `diagnosing-a-binding-failure` if it is a binding error, fix the cause in the package (which forces re-running `--check` from a new empty admission directory — move the failed one aside with a suffix, never delete it). Never weaken a check.

### Task 5: `--preflight` (GPU, about 5 minutes)

- [ ] **Step 1:** `nvidia-smi` shows ≥ 28 GiB free; `df -h` ≥ 15 GiB; `mkdir -p $MAIN/${OUT}_preflight ${OUT}_preflight`.
- [ ] **Step 2:**

```bash
gpu --rm --name b3-train-preflight $FLAGS -v $MAIN/${OUT}_admission:/vla/${OUT}_admission:ro \
  -v $MAIN/${OUT}_preflight:/vla/${OUT}_preflight:rw $OLD -m experiments.xvla_b3_robosuite.runner --preflight
```

- [ ] **Step 3:** `${OUT}_preflight/result.json`: `status: complete`, `completed_updates: 2`, `adam_reload_fp32_exact: true`, both checkpoints with reload error ≤ 1e-5. Record the measured seconds per update; multiply by 156,000 for the real duration estimate and write it in `STATUS.md`.

### Task 6: `--train` (GPU, about 36 h)

- [ ] **Step 1:** Checklist again (items 9–14). The GPU must stay free for the whole estimated window; if `nvidia-smi` shows another user's process, do not launch — write it in `STATUS.md` and stop.
- [ ] **Step 2:** `mkdir -p $MAIN/$OUT $OUT`, then launch detached (no `--rm`, so logs survive):

```bash
gpu -d --name b3-train $FLAGS -v $MAIN/${OUT}_admission:/vla/${OUT}_admission:ro \
  -v $MAIN/${OUT}_preflight:/vla/${OUT}_preflight:ro -v $MAIN/$OUT:/vla/$OUT:rw \
  $OLD -m experiments.xvla_b3_robosuite.runner --train
```

- [ ] **Step 3:** Check at 5, 15 and 60 minutes, then every few hours: `docker logs b3-train 2>&1 | grep -E '^TRAIN|^CHECKPOINT|Traceback' | tail -3`. Confirm `step_000001` lands immediately and `step_031200` at about the predicted time. Keep `outputs/train/STATUS_b3.md` current (time, update count, seconds per update, ETA, last checkpoint, disk free).
- [ ] **Step 4:** If it dies: do not relaunch without the user. Record the traceback and the last checkpoint in `STATUS_b3.md`. *(Amended 2026-10-03: this plan assumed "there is no resume". The v1 launch could not be continued after a crash, so it was replaced by v2 in `outputs/train/..._v2`, which snapshots every 2,600 updates and has `--train --resume`, gated by `--resume-proof`; see the package README, "Resuming a failed run", and FAILURES #43. Container `b3v2-train`; output paths above read `_v2`.)*

### Task 7: completion and documentation

- [ ] **Step 1:** `result.json`: `status: complete`, `completed_updates: 156000`, six checkpoint markers with reload error ≤ 1e-5, `training_state.pt` only in `step_156000`. Run `contract.verify_checkpoints(report, OUTPUT, False)` in a fresh container.
- [ ] **Step 2:** Documentation pass: `TODO.md` (training done; next: evaluation plan 3), `docs/BENCHMARK_STATUS.md` (dated entry: wall-clock, seconds/update, disk used — **no capability claim**), `docs/OPERATIONS.md` (outputs census row), `docs/INDEX.md` if a report was added, `docs/FAILURES.md` for anything that went wrong and could recur.
- [ ] **Step 3:** Commit — `git commit -m "Record the B3 X-VLA training run"`.

---

## Self-review

- Spec section 3 coverage: package forked from B2 (Tasks 1–3); binds seal, provenance, sources, base model (Task 1); B2 recipe, from base, no resume (Global Constraints, Task 2); 156,000 updates = 5.0 epochs and the six-checkpoint schedule (Task 1); 10 trained instructions only, held-out ones left to evaluation (Task 1, Review Focus 3); offline metrics excluded (Global Constraints, Task 7). Gate sequence: Tasks 4–6.
- The split-from-plan and task-by-string facts found during verification are pinned by tests (Review Focus 1–2).
- Plan-mandated duplication a reviewer will see: the runner and most of the contract are copies of B2's, which is this repo's way of versioning a training package; residue is policed by `test_fork_hygiene`.
- Not known until it runs: the real seconds per update on 295,680-row data (B2 measured 0.84 s), hence the real duration; Task 5 measures it.
