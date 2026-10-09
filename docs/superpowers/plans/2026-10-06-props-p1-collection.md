# Props P1 — instruction sorting, collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the three region/yaw amendments, re-confirm P0, then build `experiments/props_p1_data/` and collect and seal 540 scripted episodes of one-object instruction sorting, with 80 reserved evaluation layouts.

**Architecture:** A new collection package on top of the accepted `experiments/props_scene/` (imported, not copied): the scene, the expert, the success rule and the replay check are P0's. New here are the plan (cells, held-out pairs, reserved layouts), the LeRobot recording, admission, the journal and the seal — the shape B3's collector proved.

**Tech Stack:** Python 3.10, robosuite 1.5.2 / MuJoCo 3.4.0 image, LeRobot 0.4.4, numpy. CPU containers.

**Spec:** [../specs/2026-10-05-props-sorting-design.md](../specs/2026-10-05-props-sorting-design.md) — read "Carried over from B3", the P0 section (for the scene, expert and success rule) and the whole P1 section.

**Branch / worktree:** continue on `feat/props-p0` (it holds the accepted P0 package), or branch `feat/props-p1` from it — the executor's choice, stated in the report. Load `working-on-the-robosuite-path`, `collecting-a-dataset` and `launching-a-run` first.

## Global Constraints

- Never edit: `experiments/robosuite_scene/*.py`, `experiments/b3_robosuite_data/**`, `src/**`, `data_collection/**`, `validation/**`, `experiments/paired_demonstrations/**`, `experiments/pose_snapshot_v2/**`, any sealed package. Import read-only.
- `experiments/props_scene/` is the accepted P0 package. Task 1 amends it (region, lemon yaw, recheck sampling) and nothing else; after Task 2 it is frozen for the collection's lifetime and its hashes go into the P1 provenance.
- Images: robosuite `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` for anything that builds the scene; admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` for tests that import no simulator. A long run needs `--ulimit nofile=65536:65536` and `/tmp` at 16–32 GB.
- Never `env.step()`; one `mujoco.Renderer` at a time.
- Spec values, verbatim: start region x ∈ [0.27, 0.40], y ∈ [−0.12, 0.12], **radius ≥ 0.285 m**, yaw ∈ [−90°, 90°) except **lemon ±75°**; bins `blue` / `orange`, slots (0.32, ±0.33), inner 0.22 m, walls 0.07 m; objects and masses as P0; success = lowest point below the wall top, centre inside in x and y, gripper open and not touching, 30 consecutive ticks within 900.
- Held out of training: **`milk → blue`** and **`bread → orange`**. They appear only in reserved report layouts.
- Counts (corrected 2026-10-06): **20 trained cells** (10 trained pairs × 2 arrangements) × 24 = **480 training**; 20 cells × 3 = **60 validation**; **540 admitted**; reserved **80** = 20 screen + 40 report (20 trained + 20 held-out) + 20 confirm. 24 cells exist in total; the 4 belonging to the held-out pairs are never collected.
- Recording schema, unchanged from every dataset here: `observation.state` float32 (9,), `action` float32 (8,), `observation.images.image` / `image2` 512 × 512 at 30 Hz, `task` the instruction string. `FEATURES`, `ENCODING`, `PairedDataset` come from `experiments.paired_demonstrations.media`; do not fork them.
- Seeds: `sha256("props_p1_v1:20261006:<identity>")[:4]` big-endian. Balance by construction (shuffled decks), never by sampling.
- The success rule and admission gates are never weakened to admit an episode.
- Commits: user identity, no trailers, short messages; never commit `outputs/`. Do not push.
- Outputs: `outputs/props_p1_data/props_p1_<YYYYMMDD>/` in the main checkout, mounted at `/out`.

## Review Focus

1. **The instruction's colour does not match the bin the expert fills.** `test_plan` asserts, for every episode, that the target bin body named in the episode is the one whose colour the instruction names, in both arrangements.
2. **A held-out pair leaks into training.** `test_plan` asserts no training or validation episode has `(milk, blue)` or `(bread, orange)`, and that the report tier contains both, in both arrangements.
3. **Colour predicts slot.** `test_plan` asserts each colour is the target in each slot equally often across training, so position cannot stand in for colour.
4. **A rejected attempt leaves frames in the dataset.** `test_record` asserts the dataset episode count equals the journal's admitted count after injected rejections.
5. **A start pose inside the reach limit.** `test_plan` asserts every drawn start has radius ≥ 0.285 and that lemon yaws never exceed 75°.

## File map

| file | responsibility |
|---|---|
| `experiments/props_scene/constants.py`, `benchmark.py` | Task 1 only: radius floor, lemon yaw cap, recheck sampling |
| `experiments/props_p1_data/plan.py` | cells, held-out pairs, starts, bin arrangement, reserved layouts; pure |
| `experiments/props_p1_data/contract.py` | `IMAGE`, `SPEC`, `PLAN`, `OUTPUT`, `provenance()` (this package + `props_scene` + shared + spec/plan); pure |
| `experiments/props_p1_data/episode.py` | one attempt: scene, expert, admission, the recorded frames |
| `experiments/props_p1_data/validate.py` | `admit`, `fingerprint`, `validate_dataset`, `seal` |
| `experiments/props_p1_data/record.py` | `--pilot` / `--collect` / `--resume` / `--status`, journal, resumable |
| `experiments/props_p1_data/README.md` | what, images, gate order, commands |
| `experiments/props_p1_data/tests/` | `test_plan`, `test_contract`, `test_validate`, `test_record` (admission image); `test_episode` (robosuite image) |
| `docs/PROPS_P1_COLLECTION_<YYYYMMDD>.md` | collection report (Task 7) |

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
mkdir -p "$MAIN/outputs/props_p1_data" "$MAIN/outputs/props_p0"
run() { local image=$1; shift; docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536 \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla -e NUMBA_CACHE_DIR=/tmp/numba \
  -v "$PWD:/vla:ro" -v "$MAIN/outputs/props_p1_data:/out:rw" -v "$MAIN/outputs/props_p0:/p0:rw" \
  -w /vla --entrypoint python "$image" "$@"; }
```

`test_scene` writes a render to `/p0`, so that mount must be read-write when running it.

---

### Task 1: the three amendments, and re-confirm P0

**Files:** `experiments/props_scene/constants.py`, `benchmark.py`, `tests/test_constants.py`, `tests/test_benchmark.py`.

- [ ] **Step 1: Tests first.** `test_constants`: `START_RADIUS_MIN_M == 0.285`; `YAW_MAX_DEG_BY_OBJECT["lemon"] == 75.0` and 90.0 for the rest. `test_benchmark`: every planned start has radius ≥ 0.285 and respects the per-object yaw cap; the determinism recheck selects attempts spread over **both bin arrangements and at least four objects** (the first run's recheck only hit blue-left).
- [ ] **Step 2:** run in `$OLD` → FAIL.
- [ ] **Step 3: Implement.** Draw starts by rejection within the rectangle until the radius floor is met; cap yaw per object. Recheck sampling: stride across cells rather than a fixed index step.
- [ ] **Step 4:** `test_constants`, `test_benchmark` PASS; then the **P0 re-confirmation**: the full gate (6 shards, ~27 min) on the amended region. Expectation from the last run's data: the four reach failures disappear and the lemon yaw failures disappear, so ≥ 98% with no object below 95%. **If overall < 95%, stop and report** — do not tune, do not continue to Task 2.
- [ ] **Step 5: Commit** `git commit -m "Hold props starts clear of the base and cap lemon yaw"`, and append the re-confirmation numbers to `docs/PROPS_P0_RERUN_20261005.md` as a dated note (it is the gate of record, same-day, unmerged).

### Task 2: the plan

**Files:** `experiments/props_p1_data/__init__.py`, `plan.py`, `tests/test_plan.py`.

**Interfaces:** `make_plan() -> dict` with keys `version`, `master_seed`, `rng_seeds`, `cells`, `episodes`, `train_ids`, `validation_ids`, `evaluation_holdout`, `evaluation_screen_ids`, `evaluation_report_ids`, `evaluation_confirm_ids`, `held_out_pairs`, `start_region`; `validate_plan(plan)`; `attempt_draw(episode, k)`; `resample_start(episode, r)`.

Each episode carries: `episode_index`, `split`, `object`, `bin` (colour), `blue_slot` (`left`/`right`), `cell`, `instruction`, `start` `(x, y, yaw_deg)`, `home_pose_jitter_deg`, `seed`.

- [ ] **Step 1: Write the failing tests** — the five Review Focus items, plus: 480 training and 60 validation episodes; **24 per trained cell and 3 validation per cell**; exactly 20 trained cells, each `(object, bin, blue_slot)` from the 10 trained pairs; `make_plan()` is deterministic and `validate_plan` rejects a tampered copy; reserved layouts are **20 screen / 40 report / 20 confirm**, the report tier is 20 trained + 20 held-out with both held-out pairs in both arrangements (5 each), and no reserved start coincides with a collected one; `instruction` comes from one helper and names the bin colour.
- [ ] **Step 2:** run in `$OLD` → FAIL. **Step 3:** implement (follow `b3_robosuite_data/plan.py` for the seed and deck idioms; do not import it). **Step 4:** PASS. **Step 5:** commit `"Plan the P1 instruction-sorting collection"`.

### Task 3: one episode

**Files:** `episode.py`, `tests/test_episode.py`.

**Interfaces:** `run_attempt(episode, draw) -> {"success", "reason", "frames", "stable_ticks", "final_metrics", "model", "reset", "final_qpos", "final_state"}` and the recorded frame list. Builds the scene through `props_scene.scene.build_model((object,), blue_slot)`, places the object at the drawn start, runs `props_scene.expert`, records a frame per tick in the LeRobot schema with `task` = the instruction.

- [ ] **Step 1: Tests** (robosuite image): one episode per bin colour succeeds and its frames are the right dtypes and shapes; the recorded `task` equals the plan's instruction; the expert fills the bin whose colour is named, in both arrangements (Review Focus 1, at episode level). **Steps 2–5:** FAIL → implement → PASS → commit `"Record one P1 episode"`.

### Task 4: admission and the seal

**Files:** `validate.py`, `contract.py`, `tests/test_validate.py`, `tests/test_contract.py`.

Gates, in order, all recorded: `expert_success` (P0's rule, 30 ticks), `frames`, `replay` (< 1e-3 via `props_scene.validate`), `labels` (instruction, object and bin match the plan slot), `schema` (dtypes and shapes). `fingerprint` hashes JSON canonically and other files by bytes, excluding `seal.json`. `provenance()` covers this package, `props_scene/*.py`, the shared modules, the spec and this plan.

- [ ] **Step 1: Tests** (admission image, synthetic results; replay injected): each gate fails alone and names itself; `fingerprint` ignores `seal.json`, changes on any other edit, is stable under JSON key reordering, rejects symlinks; `validate_dataset` rejects a wrong count, a cell imbalance, a held-out pair present, and an instruction disagreeing with its slot. **Steps 2–5:** FAIL → implement → PASS → commit `"Admit, validate and seal P1 episodes"`.

### Task 5: the collector

**Files:** `record.py`, `README.md`, `tests/test_record.py`.

As B3's: slots in plan order; 3 attempts per layout then up to 3 deterministic resamples; a rejection budget of 150; frames buffered and written only on admission; one journal line per attempt; `--resume` refuses a dataset/journal mismatch; `--collect` refuses an existing output and writes `plan.json`, `provenance.json` first, then the dataset, then `validation.json` and `seal.json`; `--status` prints counts and the last lines. **Do not finalize and reopen the dataset per episode** (FAILURES #41); keep one handle open and finalize once.

- [ ] **Step 1: Tests** with injected `execute`/`commit`: all slots admitted in order; a rejection retries the same slot with a new draw; resampling after three failures, recorded; `SlotExhausted` and `BudgetExhausted`; resume skips admitted slots and calls `commit` exactly once per slot; a rejected attempt never commits. **Steps 2–5:** FAIL → implement → PASS → commit `"Schedule and seal the P1 collection"`.

### Task 6: pilot, then collect

- [ ] **Step 1: Pilot** (`--pilot`, ~20 slots chosen to cover both arrangements, both held-out-adjacent pairs, all six objects, including a lemon at 70–75° and a start near the radius floor). Expect ≥ 95% admitted. Extract first/middle/last frames of three episodes and **look at them**: the right object, both bins, the instructed colour filled.
- [ ] **Step 2: Go/no-go, decided by the executor**: pilot ≥ 95% admitted, frames correct, all tests green, disk ≥ 30 GB, `nvidia-smi` irrelevant (CPU). Anything else: write `STATUS.md` and stop.
- [ ] **Step 3: Collect** detached (`props-p1-collect-<date>`), ~6 h. Poll at 5, 15, 60 minutes then hourly; keep `outputs/props_p1_data/STATUS.md` current. Never rely on `docker wait`. One `--resume` after a crash; if it stops again at the same slot, stop.
- [ ] **Step 4: Verify the seal** in a second process: recomputed fingerprint equals `seal.json`; dataset loads with `LeRobotDataset`; videos decode; `validation.json` shows 540 episodes, 20 per training cell, no held-out pair, every start inside the region.

### Task 7: report and handoff

- [ ] **Report** `docs/PROPS_P1_COLLECTION_<YYYYMMDD>.md`: the P0 re-confirmation, counts as collected, rejections by gate and object, resampled layouts, wall-clock, seal fingerprint, what the inspected frames show, and anything surprising.
- [ ] **Documentation pass:** `TODO.md`, `BENCHMARK_STATUS.md`, `OPERATIONS.md` (census), `INDEX.md`, `FAILURES.md` if anything recurrable, `CLAUDE.md` and `.github/instructions/experiments.instructions.md` (the new package is hash-bound once sealed: fork, do not edit).
- [ ] **Commit**, then write `outputs/props_p1_data/HANDOFF.md` and the same text as the final message, in the previous handoff's shape, ending with a handoff prompt for Claude. The next step after this plan is **P1 training and evaluation**, which gets its own plan.

## Self-review

- Spec P1 coverage: cells and held-out pairs (Task 2), the bins-swap shortcut design (Tasks 2–3 and Review Focus 1, 3), 480 + 60 episodes (Task 2, 6), 80 reserved layouts in three tiers (Task 2), admission (Task 4), seal and resumable collection (Tasks 4–5), the region and yaw amendments with a P0 re-confirmation before any data (Task 1).
- B3 carry-overs: object pose in the trace (P0's expert already records it), the classifier is P1-evaluation work not collection, resumable collection, `--ulimit`, large `/tmp`, no per-episode reopen.
- Not known until it runs: the admitted rate on the amended region (P0 predicts ≈ 98%), and whether any reserved layout is infeasible — the evaluation plan checks that before using them.
