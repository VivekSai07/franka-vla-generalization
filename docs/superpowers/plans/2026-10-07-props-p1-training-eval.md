# Props P1 — training and evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train X-VLA on the sealed P1 dataset for 126,900 updates (5 epochs), then screen, report and confirm it on the reserved layouts, answering: does the policy put the named object in the named bin when the bins swap sides, including for two object–bin pairs it never saw?

**Architecture:** Two new packages. `experiments/xvla_props_p1/` forks `experiments/xvla_b3_robosuite/` (contract + resumable runner + gates), binding the P1 seal instead of B3's. `experiments/xvla_props_p1_eval/` follows `experiments/xvla_b3_robosuite_eval/`, with the scene and success rule imported from the accepted `experiments/props_scene/`. Training runs in the admission image (no simulator); rollouts in the robosuite image.

**Tech Stack:** Python 3.10, LeRobot 0.4.4 (X-VLA), PyTorch 2.7.1+cu128, robosuite 1.5.2 / MuJoCo 3.4.0 for rollouts, one RTX 6000 Ada.

**Spec:** [../specs/2026-10-05-props-sorting-design.md](../specs/2026-10-05-props-sorting-design.md) — read the P1 section and "P1 training and evaluation". Both are fixed; this plan implements them.

**Branch / worktree:** continue on `feat/props-p0` (it holds the sealed P1 package), or branch from it. Load `forking-an-experiment-package`, `training-a-policy`, `launching-a-run`, `evaluating-a-checkpoint` and `working-on-the-robosuite-path`.

## The parent dataset (verified 2026-10-07 by the reviewer)

| field | value |
|---|---|
| root | `outputs/props_p1_data/props_p1_20261006/` |
| seal fingerprint | `86e3aadb678963eda8cb27d0104d6e22778d4cc25c497829e8fb47937ac53bcf` (recomputed, matches) |
| episodes / frames | 540 / 244,080; **452 frames each**; 10 tasks; fps 30 |
| splits | `train_ids` 480, `validation_ids` 60 — **from `plan.json`, not the LeRobot metadata** |
| trained cells | 20, exactly 24 training episodes each |
| held-out pairs | `milk→blue`, `bread→orange` — **0 collected**, confirmed |
| reserved layouts | 80: screen 20, report 40 (20 trained + 20 held-out), confirm 20 |

## Global Constraints

- **Frozen**: `experiments/props_p1_data/**` and `experiments/props_scene/**` (both inside the P1 seal's provenance), plus `src/**`, `data_collection/**`, `validation/**`, every sealed package. Import read-only.
- Nothing is ever written inside `outputs/props_p1_data/props_p1_20261006/` — mount it read-only.
- Images: admission `sha256:08de1e09…714abc` for training and pure tests; robosuite `sha256:54d861aa…ccf27e` for rollouts, scene restore and feasibility. Long runs need `--ulimit nofile=65536:65536` and `/tmp` at 16–32 GB (FAILURES #41, #42).
- Recipe identical to B3 except horizon and data: effective batch 8 (microbatch 4 × accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4, 500 warm-up updates, cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), clip 10, FP32 parameters with bf16 autocast, `optimizer_storage='final_only'`, no resume of a prior run, from the base bundle.
- **`TOTAL_UPDATES = 126_900`** (= 203,040 complete starts × 5 / 8). **Checkpoints `[1, 50760, 76140, 88830, 101520, 114210, 126900]`.** Resumable from snapshots from the start.
- **Rollout cap 900 ticks**; success is `props_scene.metrics` unchanged; actions bounded and the gripper thresholded exactly as `validation.rollout_diagnostics.bound_action`.
- **Candidates for screening:** the six non-trivial checkpoints. **Tie rule:** every candidate within 2 successes of the best screen count goes to the report.
- The confirm tier is touched **once**, on the winner only, after the report. It is the number quoted.
- Offline metrics never select a checkpoint or signal progress.
- Never weaken a gate, a threshold or the success rule. Commits: user identity, no trailers, short messages; never commit `outputs/`. Do not push.
- Outputs: `outputs/train/xvla_props_p1_v1{,_admission,_preflight,_resumeproof}` and `outputs/xvla_props_p1_eval/`.

## Review Focus

1. **Validation episodes leak into training or statistics.** `test_contract` asserts the sampler yields exactly the 423-start-per-episode set of the 480 train ids and that statistics cover exactly those rows.
2. **Splits taken from LeRobot metadata rather than the plan.** `test_contract` asserts the split comes from `plan.json` and that a dataset whose metadata claims one split still trains on the planned 480.
3. **A held-out instruction reaches training.** `test_contract` asserts the approved instruction set excludes the two held-out pairs' instructions, and that `approved_instruction` raises for them.
4. **The evaluation trusts an infeasible reserved layout.** `test_protocol` asserts every read-out drops layouts listed failed in `feasibility.json` and reports how many were dropped.
5. **The arrangement shortcut check computed on the wrong axis.** `test_analysis` asserts the split is by `blue_slot` (which side the blue bin is on), not by target colour, and that a synthetic position-only policy scores ~50% with a large arrangement gap.

## File map

| file | responsibility |
|---|---|
| `experiments/xvla_props_p1/contract.py` | parent binding (seal, provenance, plan), data reader, instructions, horizon, schedule, `SOURCE_FILES` |
| `experiments/xvla_props_p1/runner.py` | B3's resumable runner, repointed |
| `experiments/xvla_props_p1/README.md`, `tests/` | as B3's, numbers derived from the contract |
| `experiments/xvla_props_p1_eval/protocol.py` | tiers, trials, seeds, the 12 instructions, output dirs |
| `.../scenes.py`, `feasibility.py` | restore a reserved layout; the expert on all 80 |
| `.../rollout.py`, `evaluate.py` | one rollout; CLI `--mode smoke/screen/report/confirm --checkpoint N` |
| `.../analysis.py` | causes, read-outs, clustered intervals, the shortcut check |
| `docs/PROPS_P1_TRAINING_<date>.md`, `docs/PROPS_P1_EVALUATION_<date>.md` | reports |

## Timeline (measured, not guessed — FAILURES #51)

| stage | cost |
|---|---|
| packages + tests | implementation |
| `--check` | ~1–3 h CPU |
| `--preflight` | ~5 min GPU — **measure s/update here and recompute the ETA** |
| `--train` | **~30 h** at B3's 0.848 s/update |
| feasibility (CPU, **during training**) | ~1 h |
| screen 120 rollouts | ~3 h GPU |
| report 80 rollouts per carried candidate | ~2 h GPU each |
| confirm 20 rollouts | ~0.5 h GPU |

---

### Task 1: training contract

**Files:** `experiments/xvla_props_p1/{__init__,contract}.py`, `tests/test_contract.py` (fork B3's).

**Interfaces:** every public name of the B3 contract, same signatures, with:

```python
PARENT = ROOT/'outputs/props_p1_data/props_p1_20261006'
OUTPUT = ROOT/'outputs/train/xvla_props_p1_v1'          # _admission, _preflight, _resumeproof as B3
SEAL = '86e3aadb678963eda8cb27d0104d6e22778d4cc25c497829e8fb47937ac53bcf'
FRAMES_PER_EPISODE, CHUNK = 452, 30                      # NOT 528
TRAIN, VAL = <from plan.json train_ids / validation_ids>  # 480 / 60
COMPLETE_STARTS = 480*(452-30+1)                          # 203_040
TOTAL_UPDATES = 126_900
def checkpoint_updates(preflight): return [1, 2] if preflight else [1, 50760, 76140, 88830, 101520, 114210, 126900]
```

`current_binding()` checks: `seal.json` equals the recorded seven-field record; `plan.json` passes `props_p1_data.plan.validate_plan` and its digest matches; `substitutions.json` digest matches; a fresh `props_p1_data.contract.provenance()` equals `provenance.json`; `validation.json` field values match. `parent_fingerprint()` recomputes the full fingerprint, called once by `--check`.

- [ ] **Step 1:** fork B3's `test_contract.py`, repoint, replace every literal with the values above **derived from the contract where it has them**, and add Review Focus 1–3. **Step 2:** run in `$OLD` → FAIL. **Step 3:** write `contract.py`. **Step 4:** PASS. **Step 5:** commit `"Add the props P1 training contract"`.

### Task 2: runner, README, fork hygiene

**Files:** `runner.py`, `README.md`, `tests/test_runner.py`, `tests/test_fork_hygiene.py` (fork B3's).

Keep B3's resumable machinery unchanged (snapshot every ~2,600 updates, newest two kept, `--resume`, `--resume-proof`). Only names, paths and resource numbers change: 7 checkpoints → 6×3.3 + 9.9 ≈ 30 GB, plus ~21 GB of snapshots; require **55 GB** free before `--train` and 28 GiB GPU.

- [ ] Tests first (horizon from `recipe()`, no hardcoded fallback; CLI has no resume override; the disk guard matches this schedule; symtable dead-name check; no residue of `b3_robosuite`, `156000`, `528`). Then implement, run, commit `"Add the props P1 training runner"`. **From this commit nothing in `SOURCE_FILES` is edited until training ends.**

### Task 3: gates — `--check`, `--preflight`

- [ ] **`--check`** (CPU, detached, ~1–3 h): pre-launch checklist first (`.github/skills/launching-a-run/SKILL.md` §1). Read `result.json`: `parent_fingerprint_recomputed: true`, `training_rows` = 480×452 = 216,960, `complete_training_chunks` 203,040, the train/validation lists, statistics present.
- [ ] **`--preflight`** (GPU, ~5 min): 2 updates, Adam round-trip exact, both checkpoints reload FP32-exact. **Record the measured seconds per update, multiply by 126,900, and write the real ETA into `outputs/train/STATUS_p1.md`.** If it is materially above 0.848 s/update, say so before launching.
- [ ] **`--resume-proof`**: as B3 — a new process resumes the update-1 snapshot and reproduces update 2 exactly.

### Task 4: `--train` (~30 h)

- [ ] `nvidia-smi` free (≥ 28 GiB), `df -h` ≥ 55 GB. Launch detached, no `--rm`, `--ulimit nofile=65536:65536`.
- [ ] Check at 5, 15, 60 minutes then every few hours; keep `STATUS_p1.md` current (update count, s/update, ETA, last checkpoint, disk). Confirm `step_000001` lands immediately and `step_050760` near the predicted time.
- [ ] If it dies: `--resume` **once** from the latest snapshot; if it stops again at the same point, record and stop.

### Task 5: evaluation package (build **while training runs**, CPU only)

**Files:** `experiments/xvla_props_p1_eval/{__init__,protocol,scenes,feasibility,rollout,evaluate,analysis}.py`, `README.md`, `tests/`.

- `protocol.py`: `screen_trials()` 20, `report_trials()` 80 (40 layouts × 2 seeds), `confirm_trials()` 20; seeds `props_p1_data.plan.seed(f"evaluation:{tier}:{layout_id}:{k}")`; `eval_instruction()` approving **all 12** instructions (the two held-out ones exist only here); `output_dir(mode, candidate)`.
- `scenes.py`: restore a reserved layout from `plan.json`'s `evaluation_holdout` through `props_scene.scene`, with its bin arrangement and start pose; assert the restored state matches the plan to 1 mm / 1°.
- `feasibility.py`: the scripted expert on all 80 layouts → `feasibility.json` with `failed_layouts`.
- `rollout.py`: the loop from `xvla_b3_robosuite_eval` (read it; reproduce, do not import), with the P1 success rule, 900-tick cap, per-tick object pose, video, trace; `episode.json` carries `object`, `bin`, `blue_slot`, `tier`, `kind` (trained/held_out), `placed_bin` (which bin it ended in, or none), `cause`.
- `analysis.py`: `classify`, `readouts` (overall, by kind, by `blue_slot`, wrong-bin rate, causes, dropped-infeasible count), `clustered_interval` bootstrapping layouts.
- `evaluate.py`: refuses without `feasibility.json`, verifies the checkpoint against the training `result.json`, refuses an existing output dir, requires MuJoCo 3.4.0.

- [ ] Tests first, including Review Focus 4–5. Commit per module. **Run `feasibility.py` on CPU while training runs** and record how many of the 80 layouts the expert solves; if more than 5 fail, report before any rollout.

### Task 6: screen → report → confirm

- [ ] **Smoke** (2 trials, one trained + one held-out) on the final checkpoint: the policy-load and rollout path works; look at a video frame of each.
- [ ] **Screen**: the six non-trivial checkpoints × 20 layouts, sequentially (one GPU). Apply the tie rule; write the table into `STATUS_p1.md`.
- [ ] **Report**: each carried candidate × 80 trials.
- [ ] **Confirm**: the winner only, once, × 20 trials on the untouched tier.
- [ ] Validate the classifier on successes first; watch a video of every wrong-bin trial and one per failure cause before naming causes.

### Task 7: reports and documentation

- [ ] `docs/PROPS_P1_TRAINING_<date>.md` (wall-clock, s/update, checkpoints, no capability claim) and `docs/PROPS_P1_EVALUATION_<date>.md`: screen table and tie decision; report read-outs with clustered intervals; **the confirm number as the headline**; the four fixed read-outs with the spec's reading; causes; feasibility exclusions; limitations.
- [ ] Documentation pass: `docs/RESULTS.md`, `README.md`, `BENCHMARK_STATUS.md`, `TODO.md`, `INDEX.md`, `OPERATIONS.md` census, `FAILURES.md` for anything recurrable.
- [ ] Commit; write `HANDOFF.md` in the usual shape ending with a handoff prompt for Claude.

## Self-review

- Spec coverage: horizon and schedule from the measured dataset (Task 1); B3's recipe and resumability (Task 2); gates (Tasks 3–4); feasibility, screen, report, confirm with the tie rule and the once-only confirm tier (Tasks 5–6); the four fixed read-outs and their pre-agreed reading (Tasks 5, 7).
- The 452-frame figure — the one number that differs from B3 and drives everything — is pinned by `test_contract` and appears in the contract, not re-typed in the runner.
- Known duplication a reviewer will see: the contract, runner and rollout loop are forks, which is this repo's versioning pattern; `test_fork_hygiene` polices residue.
- Not known until it runs: the real seconds per update on 452-frame episodes (Task 3 measures it before the 30 h commitment), and whether any reserved layout is infeasible (Task 5, during training).
