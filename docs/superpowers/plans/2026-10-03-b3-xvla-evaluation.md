# B3 X-VLA evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the B3-trained X-VLA policy in the robosuite scene on B3's reserved layouts, and produce the read-outs the spec fixed in advance: overall success, held-out against trained colour pairs, distractors against none, edge against interior, wrong-cube rate, and failure causes.

**Architecture:** A new package `experiments/xvla_b3_robosuite_eval/`, robosuite image, GPU for rollouts. Pure modules (trial protocol, read-out analysis) are tested in the admission image; scene restore, the expert feasibility pass and the rollout loop run in the robosuite image. Starting scenes come from the B3 bank sealed with the dataset, never re-derived. The rollout loop is the one written for the sim2sim check, extended with per-cube tracking and the 12-instruction approval. All outputs go to `outputs/xvla_b3_robosuite_eval/`; the training output and the dataset are mounted read-only.

**Tech Stack:** Python 3.10, robosuite 1.5.2 / MuJoCo 3.4.0 image with torch + LeRobot 0.4.4 (X-VLA), OSMesa, one GPU.

**Spec:** [../specs/2026-10-03-b3-robosuite-cubes-design.md](../specs/2026-10-03-b3-robosuite-cubes-design.md), section 4. Plan 3 of 3. Depends on plan 2 for the trained checkpoints; Tasks 1–5 and the feasibility run (Task 6) can be done while plan 2 trains.

**Branch / worktree:** `feat/b3-robosuite-cubes`, worktree `.worktrees/feat-b3-robosuite-cubes`. Load `working-on-the-robosuite-path`, `evaluating-a-checkpoint` and `diagnosing-policy-failures` before Task 1.

## Global Constraints

- Frozen, never edited: everything in the B3 seal's provenance (`experiments/b3_robosuite_data/**`, `experiments/robosuite_scene/*.py`, the B3 spec, the collection plan, `src/**`, the shared files in `b3_robosuite_data.contract.SHARED_FILES`) and everything in `experiments/xvla_b3_robosuite/contract.py`'s `SOURCE_FILES`. Import them read-only.
- **Nothing is ever written under `outputs/b3_robosuite_data/b3_robosuite_20261003/`**: its fingerprint covers every file in that directory. Mount it read-only.
- Images: rollouts, scene restore and feasibility in the robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e`; pure tests in the admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
- Never `env.step()`; one `mujoco.Renderer` at a time; never `env.sim.render`.
- Rollout rules (unchanged from every earlier evaluation here): 1,200-tick cap; success = 30 consecutive ticks of `stack_metrics(...)["released_stack"]` with the unchanged window; actions bounded by the actuator ranges and the gripper thresholded at 127, exactly as `validation.rollout_diagnostics.bound_action`; 20 physics substeps per tick.
- **Candidates:** `(62400, 93600, 124800, 156000)` — epochs 2–5. Update 1 and 31,200 are not evaluated.
- **Screen:** the 20 screen layouts, `distractor` condition, one rollout seed each, every candidate: 80 trials. **Tie rule:** every candidate within 2 successes of the best screen count goes to the report.
- **Report:** each carried candidate on the 40 report layouts × {`control`, `distractor`} × 2 rollout seeds = 160 trials. The two conditions of a layout share its two seeds, so the condition contrast is paired.
- Rollout seeds: `b3_robosuite_data.plan.seed(f"evaluation:screen:{layout_id}")` and `seed(f"evaluation:report:{layout_id}:{k}")`, `k ∈ {0, 1}`. Identical across candidates.
- Instructions: the 12 ordered pairs of the four colours, text from `experiments.paired_demonstrations.plan.instruction`. The two held-out ones (`yellow→red`, `green→blue`) are allowed **only** in this package.
- Read-outs and their interpretation are fixed by spec section 4 and are not changed after any result is seen. No per-layout claims (2 seeds per layout).
- Offline metrics are never used to pick a candidate.
- Commits: user identity, no trailers, short messages; no outputs. The user pushes.
- GPU stages only when `nvidia-smi` shows the GPU free (≥ 28 GiB) and training is not running.

## Review Focus

1. **A held-out instruction leaks the policy a trained one.** `test_protocol` asserts every held-out report trial carries exactly its held-out instruction string and that `eval_instruction()` restores the shared `TASK` even when the body raises.
2. **Scene restored from the wrong bank entry or a drifted bank.** `test_scenes` asserts the restored `qpos` equals the bank array bit-for-bit and that a one-byte change in `bank.npz` (temporary copy) is rejected by its manifest hash.
3. **Wrong-cube detection fooled by the pick cube being nudged.** `test_analysis` feeds a trace where a distractor rises 2 cm (bumped) then the pick cube rises 15 cm, and asserts `first_lifted == pick`.
4. **Excluded (expert-infeasible) layouts silently counted.** `test_analysis` asserts every read-out drops trials on layouts listed in `feasibility.json` as failed and reports how many were dropped.
5. **Condition contrast computed unpaired.** `test_analysis` asserts the distractor-minus-control difference is computed from per-layout pairs and that a layout missing one condition is excluded from the contrast with a count.

## File map

| file | responsibility | image |
|---|---|---|
| `experiments/xvla_b3_robosuite_eval/__init__.py` | empty | — |
| `protocol.py` | paths, candidates, screen/report trial lists, seeds, the 12-instruction approval, output dir names | pure |
| `scenes.py` | `restore(key)` from the sealed bank with hash checks | robosuite |
| `feasibility.py` | the scripted expert on all 100 banked scenes | robosuite, CPU |
| `rollout.py` | one policy rollout: loop, video, trace, per-cube tracking | robosuite, GPU |
| `evaluate.py` | CLI: `--mode smoke|screen|report --checkpoint N` | robosuite, GPU |
| `analysis.py` | first-cause classification, read-outs, layout-clustered intervals | pure |
| `README.md` | what, how, costs | — |
| `tests/test_protocol.py`, `test_analysis.py` | pure tests | admission |
| `tests/test_scenes.py`, `test_rollout.py` | restore and a 50-tick rollout with a stub policy | robosuite |
| `docs/B3_XVLA_EVALUATION_<YYYYMMDD>.md` | the report (Task 8) | — |

Shared names:

```python
# protocol.py
PARENT, BANK_NPZ, BANK_JSON, PLAN_JSON          # under outputs/b3_robosuite_data/b3_robosuite_20261003
EVAL_ROOT = ROOT/'outputs/xvla_b3_robosuite_eval'
CANDIDATES = (62400, 93600, 124800, 156000)
ALL_INSTRUCTIONS: tuple[str, ...]                # 12, sorted
def screen_trials() -> list[dict]               # 20: ordinal, layout_id, condition, seed, pick, place, instruction, kind, edge, colours
def report_trials() -> list[dict]               # 160, same keys, plus seed_index
def eval_instruction(instruction) -> contextmanager
def output_dir(mode: str, candidate: int) -> Path   # EVAL_ROOT/f'{mode}_step_{candidate:06d}'
# scenes.py
def restore(key: str) -> tuple                  # (model, data); key = f'{layout_id}:{condition}'
# rollout.py
def rollout(policy, pre, post, model, data, renderer, trial, folder) -> dict   # episode.json contents
# analysis.py
def classify(trace: dict, pick: str, success: bool) -> str   # 'success'|'wrong_cube'|'never_lifted'|'lifted_not_carried'|'released_off_target'
def first_lifted(trace: dict, threshold_m=0.03) -> str | None
def readouts(rows: list[dict], excluded_layouts: set) -> dict
def clustered_interval(rows, key, n=10_000, seed=0) -> tuple[float, float]
```

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
PARENT=outputs/b3_robosuite_data/b3_robosuite_20261003
TRAIN=outputs/train/xvla_franka_stacking_b3_robosuite_v2   # v1 was superseded by the resumable v2 (FAILURES #43)
EVAL=outputs/xvla_b3_robosuite_eval
mkdir -p $PARENT $TRAIN $EVAL .cache $MAIN/$EVAL       # mount points in the worktree; host output root
FLAGS="--network none --read-only --user $(id -u):$(id -g) --tmpfs /tmp:rw,size=16g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/vla -e NUMBA_CACHE_DIR=/tmp/numba \
  -v $PWD:/vla:ro -v $MAIN/$PARENT:/vla/$PARENT:ro -v $MAIN/$TRAIN:/vla/$TRAIN:ro \
  -v $MAIN/.cache:/vla/.cache:ro -v $MAIN/$EVAL:/vla/$EVAL:rw -w /vla --entrypoint python"
# CPU: docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void $FLAGS <image> ...
# GPU: docker run -d --name <name> --runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 $FLAGS $NEW ...
```

---

### Task 1: protocol

**Files:** Create `__init__.py`, `protocol.py`, `tests/test_protocol.py`.

**Interfaces:** produces the `protocol.py` names above. Reads `PLAN_JSON` (the sealed plan) for the reserved layouts — `evaluation_holdout`, `evaluation_screen_ids`, `evaluation_report_ids`.

- [ ] **Step 1: Write the failing tests** (admission image):

```python
    def test_screen(self):
        trials = screen_trials()
        self.assertEqual(len(trials), 20)
        self.assertEqual({t["condition"] for t in trials}, {"distractor"})
        self.assertEqual({t["kind"] for t in trials}, {"trained"})
        self.assertEqual([t["ordinal"] for t in trials], list(range(1, 21)))

    def test_report(self):
        trials = report_trials()
        self.assertEqual(len(trials), 160)
        self.assertEqual(Counter(t["kind"] for t in trials), {"trained": 80, "held_out": 80})
        self.assertEqual(Counter(t["condition"] for t in trials), {"control": 80, "distractor": 80})
        self.assertEqual(Counter(t["edge"] != "interior" for t in trials)[True], 40)
        by_layout_seed = defaultdict(list)                          # both conditions share the seed
        for t in trials: by_layout_seed[(t["layout_id"], t["seed_index"])].append(t)
        for pair in by_layout_seed.values():
            self.assertEqual(sorted(t["condition"] for t in pair), ["control", "distractor"])
            self.assertEqual(len({t["seed"] for t in pair}), 1)

    def test_held_out_instructions_only_on_held_out_layouts(self):
        for t in report_trials():
            held = (t["pick"], t["place"]) in HELD_OUT_PAIRS
            self.assertEqual(held, t["kind"] == "held_out")
            self.assertEqual(t["instruction"], instruction(t["pick"], t["place"]))

    def test_colours_present(self):         # control: pick+place only; distractor: all four
    def test_seeds_are_stable_distinct_and_match_the_rule(self):
    def test_eval_instruction_accepts_12_rejects_others_and_restores(self):  # restore also on exception
    def test_output_dir(self): self.assertEqual(output_dir("screen", 62400).name, "screen_step_062400")
    def test_no_robosuite_import(self):     # ast-parse protocol.py
```

- [ ] **Step 2: Run** in `$OLD` → FAIL. **Step 3: Implement.** `eval_instruction` follows the training contract's `approved_instruction` pattern (swap `experiments.xvla_single_task.contract.TASK`, restore in `finally`) but accepts `ALL_INSTRUCTIONS`. **Step 4: Run** → PASS. **Step 5: Commit** `git commit -m "Add the B3 evaluation protocol"`.

---

### Task 2: scene restore

**Files:** Create `scenes.py`, `tests/test_scenes.py`.

**Interfaces:** `restore(key) -> (model, data)`. Uses `b3_robosuite_data.scenes.build_model(colours)` and the bank's per-key `colours`; restores `initial_state` with `data_collection.benchmark.restore_state`; checks `qpos` equals the bank's `qpos` exactly and the manifest's scene hash (recompute it the way `b3_robosuite_data.scenes.make_bank` did — read that function, do not import its private helper if it is private; reimplement the documented hash and test it against the stored value).

- [ ] **Step 1: Tests** (robosuite image): three keys (a screen distractor key, a report control key, a report held-out distractor key) restore with exact `qpos`; the model contains exactly the bank's colours; every cube sits within 1 mm of its planned xy; Review Focus 2 (tampered copy of `bank.npz` in `/tmp`).
- [ ] **Step 2–4:** FAIL → implement → PASS (`$NEW`). **Step 5: Commit** `git commit -m "Restore B3 reserved scenes from the sealed bank"`.

---

### Task 3: feasibility pass

**Files:** Create `feasibility.py` (CLI `python -m experiments.xvla_b3_robosuite_eval.feasibility --out /vla/$EVAL/feasibility`).

**Behaviour:** for each of the 100 bank keys: `restore(key)`; run `b3_robosuite_data.expert.B3Expert(colours).run(initial_state, episode, sink)` once, where `episode` is built from the reserved layout (`pick`, `place`, `instruction`, `home_pose_jitter_deg`, `hover_z_offset`) with `seed = plan.seed(f"evaluation:feasibility:{key}")`; record `success`, `reason`, `final_xy_m`, `final_z_m`. Write `feasibility.json` = `{"results": {key: …}, "failed_keys": [...], "failed_layouts": [...]}` (a layout fails if any of its keys fails). One process, about 100 × 50 s ≈ 1.5 h CPU. Refuses an existing `--out`.

- [ ] **Step 1: Test** (`test_scenes.py`, robosuite image): `run_one(key)` on one screen key returns `success` true and `final_xy_m < 0.02`. **Step 2–4:** FAIL → implement → PASS. **Step 5: Commit** `git commit -m "Check every reserved B3 scene with the scripted expert"`.

---

### Task 4: the rollout

**Files:** Create `rollout.py`, `tests/test_rollout.py`.

**Interfaces:** `rollout(policy, pre, post, model, data, renderer, trial, folder) -> dict`. Start from a copy of `experiments/robosuite_sim2sim/rollout.py::rollout` (read it), with these changes:

- the instruction context is `protocol.eval_instruction(trial["instruction"])` instead of B2's `approved_instruction`, and observations go through `experiments.xvla_single_task.contract.observation` exactly as there;
- the trace also records, every tick, `cube_xyz` = positions of every cube present (array shaped `(ticks, n_colours, 3)`, with `colours` saved beside it), so analysis can tell which cube moved first;
- `episode.json` gains `pick`, `place`, `kind`, `edge`, `condition`, `layout_id`, `seed`, `colours`, `first_lifted` (from `analysis.first_lifted`), `cause` (from `analysis.classify`), plus everything sim2sim already wrote;
- the video is written as before (both cameras side by side).

- [ ] **Step 1: Test** (robosuite image, CPU is enough): a stub policy object with `reset()` and `select_action()` returning a fixed in-range 8-vector, identity `pre`/`post`, one restored report key, the loop capped to 50 ticks through a module-level `MAX_TICKS` the test patches: `episode.json` exists with the new keys, `trace.npz` has `cube_xyz` of shape `(50, n, 3)`, `cameras.mp4` decodes to 51 frames. **Step 2–4:** FAIL → implement → PASS. **Step 5: Commit** `git commit -m "Roll out a policy on a B3 reserved scene"`.

---

### Task 5: analysis and the CLI

**Files:** Create `analysis.py`, `evaluate.py`, `README.md`, `tests/test_analysis.py`.

**`analysis.py`** (pure numpy):

- `first_lifted(trace)`: the first cube whose z rises more than `threshold_m` above its tick-0 z **and stays above it for 15 consecutive ticks** (a bump is not a lift); `None` if none.
- `classify(trace, pick, success)`: `success`; else `wrong_cube` if `first_lifted` is a colour other than `pick`; else `never_lifted` if the pick cube never rose 3 cm; else `released_off_target` if the gripper reopened after the pick was lifted; else `lifted_not_carried`. Validate on successes first: every success must classify as `success` with `first_lifted == pick` (the test enforces this on synthetic traces; Task 8 checks it on real ones).
- `readouts(rows, excluded_layouts)`: overall; by `kind` (trained / held-out); by `condition`; paired `distractor − control` per layout; by `edge` (interior / inner / outer); wrong-cube rate overall and by `kind`; counts of each `cause`; number of trials dropped as infeasible. Each rate with `clustered_interval`.
- `clustered_interval(rows, key, n=10_000, seed=0)`: 95% percentile bootstrap resampling **layouts** (with all their trials), `numpy.random.default_rng(seed)`.

**`evaluate.py`**: `--mode smoke|screen|report`, `--checkpoint` one of `CANDIDATES`. Before any rollout: refuse unless `EVAL_ROOT/feasibility/feasibility.json` exists; verify the checkpoint with the training contract (`xvla_b3_robosuite.contract.verify_checkpoints(result, OUTPUT, False)` on the training `result.json`) and its `config.json` (`type xvla`, `chunk_size 30`, `n_action_steps 10`, `n_obs_steps 1`, `dtype float32`); refuse an existing output dir; require `mujoco.__version__ == "3.4.0"`. Load the policy the way `xvla_b2_distractors_eval.evaluate.load_counted` does (read it; reproduce, do not import that package) and `make_pre_post_processors(policy.config, pretrained_path=checkpoint)`; `torch.use_deterministic_algorithms(True)`. Group trials by colour set and open one renderer per group (OSMesa: never two at once). After every trial rewrite `result.json` (trials so far + `readouts`). `smoke` = the first screen trial and the first held-out report trial (2 trials). Print one `EVAL ` line per trial.

- [ ] **Step 1: Tests** (`$OLD`): Review Focus 3–5; classification of five synthetic traces, one per cause; `clustered_interval` returns a narrower interval for 40 layouts than for 4 with the same rate; `readouts` on a hand-built 8-row table returns the exact expected numbers. **Step 2–4:** FAIL → implement → PASS. `README.md`: stages, commands, costs, "outputs only under `outputs/xvla_b3_robosuite_eval/`". **Step 5: Commit** `git commit -m "Analyse and run B3 evaluations"`.

---

### Task 6: feasibility run (CPU, can run while training)

- [ ] `docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void $FLAGS $NEW -m experiments.xvla_b3_robosuite_eval.feasibility --out /vla/$EVAL/feasibility` (detached with a name if long). Expect about 1.5 h; it must not slow training (CPU only, one process).
- [ ] Record in `STATUS_b3.md`: how many of 100 keys succeeded; list failed layouts. If more than 5 layouts fail, stop and tell the user before any GPU stage: that would say the reserved set is harder than the training set for the expert itself.
- [ ] Commit nothing (outputs are not committed); note the result in the task report.

### Task 7: smoke, screen, report (GPU, after training finishes)

- [ ] **Smoke:** `--mode smoke --checkpoint 156000`; both trials finish (any outcome), `cameras.mp4` plays, wall-clock per trial is plausible (tens of seconds to a few minutes — `evaluating-a-checkpoint` §1: an `rc=0` in seconds means nothing ran). Look at one video frame per trial.
- [ ] **Screen:** the four candidates, one detached container each, run **one after another** (one GPU). About 1.5–4.5 h per candidate. Read each `result.json`; apply the tie rule; write the screen table into `STATUS_b3.md` with the carried candidate(s).
- [ ] **Report:** each carried candidate, 160 trials, about 3–9 h each. Poll; never rely on `docker wait`.

### Task 8: report and documentation

- [ ] **Check the classifier on the real data first:** every success classifies as `success` with `first_lifted == pick`; read the videos of at least one trial per failure cause and of every `wrong_cube` trial before writing a cause into the report.
- [ ] **Report** `docs/B3_XVLA_EVALUATION_<YYYYMMDD>.md`: the screen table and tie decision; the report read-outs with clustered intervals; the interpretation per spec section 4 (state which of its pre-fixed readings applies, with the numbers); the cause table; feasibility exclusions; limitations (2 seeds per layout; robosuite scene only; no contract on the robosuite scene package; MuJoCo 3.4.0 not comparable with earlier results); what was ruled out.
- [ ] **Documentation pass:** `docs/RESULTS.md` (B3 section with caveats), `README.md` headline, `docs/BENCHMARK_STATUS.md`, `TODO.md` (B3 done; next: the props project), `docs/INDEX.md`, `docs/FAILURES.md` for anything recurrable, the `evaluating-a-checkpoint` skill if a new lesson came out.
- [ ] **Commit** `git commit -m "Report the B3 X-VLA evaluation"`. Then tell the user the branch is ready to merge (merging is theirs).

---

## Self-review

- Spec section 4 coverage: feasibility first (Tasks 3, 6), screen with the tie rule (Task 7), report 40 × 2 × 2 = 160 (Tasks 1, 7), read-outs fixed in advance incl. wrong-cube rate and causes validated on successes (Task 5, 8), interpretation rules quoted from the spec (Task 8), no per-layout claims (Global Constraints).
- Names match between tasks: `protocol.screen_trials/report_trials/eval_instruction/output_dir`, `scenes.restore`, `rollout.rollout`, `analysis.first_lifted/classify/readouts/clustered_interval`.
- The sealed dataset directory is mounted read-only and nothing writes inside it; feasibility and rollouts write under `outputs/xvla_b3_robosuite_eval/`.
- Plan-mandated duplication a reviewer will see: the rollout loop is copied from `robosuite_sim2sim` (an unbound package, so importing it would let a later edit there change B3's protocol silently) and the policy loader is reproduced from the B2 eval package.
