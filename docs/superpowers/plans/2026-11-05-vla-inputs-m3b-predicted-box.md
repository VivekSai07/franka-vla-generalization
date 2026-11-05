# M3b — the box from pixels instead of the oracle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** answer Q1–Q4 of the [M3b spec](../specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md): how much of A3's gain over A2 survives a
box predicted from pixels (arm A5), what box error A3 tolerates (arms A6–A9, corrupted oracle), whether A3 follows the box or the language
(A9), and where a predicted box fails. **No policy is trained.** Decision rule on R = (S − 523)/48 in spec D10.

**Architecture:** three new packages, nothing frozen edited. `experiments/vla_inputs_arm_xvla_force_box_noisy/` (A6–A9: A3's frozen adapter with a
corrupted box); `experiments/vla_inputs_box_detector/` (the pixel-only predictor: data cache, model, contract, trainer, gate);
`experiments/vla_inputs_arm_xvla_force_box_pred/` (A5: A3's frozen adapter with the detector's box); plus `experiments/vla_inputs_m3b_analysis/` and
`scripts/vla_inputs_m3b_eval.sh`. Rollouts are M0's frozen evaluator, in the existing campaign root.

**Tech stack:** admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (detector data, training, CPU tests),
robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (replay, parity, rollouts; run with `-e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba`).

**Depends on:** M3 done (A3 trained, evaluated at 101,520 = 571/573; checkpoints in `outputs/train/xvla_p2_force_box_v1/checkpoints/`), S0 sealed, M0
core `c4f6c303f2ae`, A2's comparison stage and A3's, both in `outputs/vla_inputs_eval/m0_20261030`.

## Global constraints

- **Frozen, never edited:** `experiments/xvla_p2_force/**`, `experiments/xvla_p2_force_box/**`, `experiments/vla_inputs_arm_xvla_force_box/**`,
  `experiments/vla_inputs_arm_xvla_force/**`, `experiments/vla_inputs_arm_xvla/**`, `experiments/vla_inputs_signals/**`,
  `experiments/vla_inputs_eval/**` (core hash must stay `c4f6c303f2ae`), `experiments/xvla_props_p2/**`, `experiments/xvla_single_task/contract.py`,
  everything in P2's `provenance()`. Import read-only; A3's arm closure hash `150d3970b3b4` is re-verified by M0 whenever A3 is touched.
  The A3 comparison and A2 stages under `outputs/vla_inputs_eval/m0_20261030/arms/{A2,A3}/` are read-only inputs; never rewrite them.
- **Branch and worktrees:** work on `feat/vla-inputs-m3b` in `.worktrees/feat-vla-inputs-m3b` (skill `branching-for-an-experiment-axis`; do not reuse
  `feat/vla-inputs-m3`). Detector training and every rollout run from **detached worktrees** at the gated, merged commit (`.worktrees/m3b-run`,
  `.worktrees/m3b-eval`); never `checkout`/`switch`/`reset` a mounted directory. Merge into `main` with `--no-ff` before anything runs from it.
- **Fixed before any run they govern (write them into the contract or constants module, with a test that pins them):** the corruption parameters
  (D3), the detector gate thresholds and the frame stride k (D4–D5), the presence threshold 0.5, and the decision thresholds (D10).
- **Resource guards:** 65 GiB free disk (the detector cache counts), 28 GiB free GPU memory, thermal pause 83 → 75 °C; none relaxed. Evaluation: 6 workers
  × 7 cores, one checkpoint per worker. Poll with `docker ps` and the stage logs, never `docker wait`.
- **Tests:** admission image, `-m unittest discover -s <pkg>/tests -t <pkg>/tests`, mounting `outputs/props_p2_data` and `outputs/vla_inputs_signals`
  read-only; rollout-side tests in the robosuite image. Every new test is confirmed to fail against a deliberate mutation.
- **Stop and report** (do not weaken anything) on any failed gate, any change to the core hash, a busy GPU or less than 65 GiB free.
- Commits as VivekSai07 <viveksaisurya07@gmail.com>, short messages, no AI trailers; nothing pushed; nothing deleted under `outputs/` or `datasets/`.

---

### Task 0: preconditions and orientation

- [ ] `docker ps`, `nvidia-smi`, `df -h /`, `git worktree list`, `git log --oneline -5`. Record the numbers in the session notes.
- [ ] Load skills `branching-for-an-experiment-axis`, `forking-an-experiment-package`, `launching-a-run`, `working-on-the-robosuite-path`,
  `evaluating-a-checkpoint`, `diagnosing-policy-failures`, `test-driven-development`, `verification-before-completion`.
- [ ] Read `docs/VLA_INPUTS_M3_20261105.md`, the M3b spec (all of it), `experiments/xvla_p2_force_box/{overlay.py,eval_adapter.py,eval_compare.py}`,
  `scripts/vla_inputs_m3_eval.sh`, `experiments/vla_inputs_eval/{adapters.py,rollout.py}`.
- [ ] Create the branch and worktree from `main`. Confirm `contract.EXTRA_CORE` and `package_files` in `vla_inputs_eval/contract.py` do **not** list
  `scripts/vla_inputs_eval.sh` (if they do, the driver is frozen and the detector mount below needs another route; stop and ask).

### Task 1: corruption arms A6–A9 (CPU)

**Files:** create `experiments/vla_inputs_arm_xvla_force_box_noisy/{__init__.py,corruption.py,adapter.py,README.md}` and `tests/`.

**Interfaces:** `corruption.CONDITIONS = {"jitter_small": ..., "jitter_large": ..., "dropout": ..., "swap": ...}` with the spec D3 constants pinned in one place;
`corruption.apply(condition, box, tick, distractor_box=None) -> (2, 5) float64`, deterministic from sha256 of (condition, box bytes, tick), output always a valid box
(ordered, minimum side 2 px, clipped to [0, 1], `in_frame` 0 means all zeros); `adapter.XVLAJitterSmall`, `XVLAJitterLarge`, `XVLADropout`, `XVLASwap` subclasses of A3's
`XVLAForceBoxAdapter` overriding only the box taken in `act`. The swap state builder reads the distractor from the model: bodies named `prop_*` other than the
target, the one nearest the target by `data.xpos` distance; its box is `signals.target_box(model, data, <name>)`.
- [ ] **Step 1: failing tests (admission image where possible):** each condition's output is valid for random, edge-touching, tiny and `in_frame` 0 boxes; determinism
  (same inputs, same output; different tick, different noise); σ is what the constants say over 10,000 draws; dropout rate about 0.5 per `act` on the gripper camera only,
  front camera byte-equal to input; swap returns the distractor's box on both cameras and requires one; a condition with its constant set to zero (or dropout 0) is
  the identity.
- [ ] **Step 2 (robosuite image):** on replayed sealed episodes 0, 18, 271, 539 (S0 `replay.frames`): with a condition's corruption off, the drawn frames and the policy's processed
  inputs equal A3's byte for byte (spec D7 i); the swap distractor is a prop other than the target and is in the scene for every layout of the M0 plan (all 573 + 3 dropped
  comparison layouts: check `scenes.restore` once per layout, CPU, no policy).
- [ ] **Step 3–4:** forking-skill pass (symtable dead-name scan; grep for `force_box` residue), implement, pass, mutate (σ halved; clip removed; seeded by trial order; dropout on the
  front camera; wrong distractor; target box drawn in A9). Each mutation must fail a test.
- [ ] **Step 5:** README (non-zero size, states what each condition does and that forces stay privileged); commit `M3b: corruption arms`.

### Task 2: analysis module (CPU)

**Files:** create `experiments/vla_inputs_m3b_analysis/{__init__.py,analyse.py,README.md}` and `tests/`.

**Interfaces:** `recovery(a5_rows, a2_rows, a3_rows, seed=0, resamples=10000)` returns R, the interval, S, floor, ceiling, gap (ratio bootstrap, layouts resampled stratified by
object, numerator and denominator together); `compare_to(arm, control)` exact McNemar and the paired effect through M0's `stats`; `holm(...)` over A6–A9 against A3;
failure classes via `analysis.readouts`; `detector_error(log_dir, trace_dirs)` per-tick IoU against the oracle, share of ticks with no box in each camera, replan-to-replan jitter;
a CLI `--root ... analyse`. It refuses unless every arm's `prepared.json` carries core `c4f6c303f2ae`, the layout ids and dropped layouts agree, and A2's and A3's stages exist and are complete.
- [ ] **Step 1: failing tests** on synthetic stage fixtures: hand-computed R (S = 559 gives exactly 0.75 against 523/571), exact McNemar b/c/p, bootstrap on a case with a known answer
  (all-equal rows give R undefined and a clear error), Holm values, refusals on core, layout and drop mismatch, IoU of known boxes.
- [ ] **Step 2–4:** fail, implement, pass, mutate (floor and ceiling swapped; unstratified bootstrap; Holm over five; wrong control). Commit `M3b: analysis`.

### Task 3: evaluation driver (CPU)

**Files:** create `scripts/vla_inputs_m3b_eval.sh` (a copy of `scripts/vla_inputs_m3_eval.sh`: steps `prepare <arm>`, `smoke <arm>`, `comparison <arm>`, `analyse`; arms A5–A9 map to
their adapter identifiers; `CK` = A3's checkpoint directory, checkpoint 101,520 the single and matched-phase checkpoint); extend `scripts/vla_inputs_eval.sh` minimally with an
`EXTRA_MOUNTS` environment variable (extra `-v` pairs) for the detector directory, with a test against a fake `docker` as the existing driver tests do.
- [ ] Tests (fake `docker`): the printed commands for each step, the arm-to-adapter map, refusal without `prepare`, `M0_NO_RUN=1` prints and runs nothing.
- [ ] Commit `M3b: evaluation driver`. **Merge into `main` with `--no-ff` before the GPU work below**, so the gated commit is on `main`; create `.worktrees/m3b-eval` detached at it.

### Task 4: night A — corrupted-oracle arms A6–A9 (GPU)

- [ ] Pre-launch checklist (`launching-a-run` section 1); `docker ps`, `nvidia-smi`, `df -h /`. Refuse a busy GPU.
- [ ] **Live equality gate (spec D7 iv):** `prepare` A6 with the corruption constants off is not allowed as an arm; instead run, from the eval worktree, a scratch campaign copy
  (a copy of `plan/` and `feasibility/`, A0 and A3 prepared records as references) and re-run comparison ordinals 1 and 291 with a **zero-corruption** adapter; both traces equal A3's recorded
  `trace.npz` arrays bit for bit. Stop on any difference.
- [ ] `prepare` A6, A7, A8, A9 (checkpoint 101,520 from `outputs/train/xvla_p2_force_box_v1/checkpoints`, phase 101,520) → `smoke` each (2 trials) → `comparison` each, one arm after the other
  (6 workers × 7 cores, about 2.7 h each). Check after each arm that the stage is complete, and that the shared core is still `c4f6c303f2ae`.
- [ ] Status entry in `docs/BENCHMARK_STATUS.md` at launch and when done (the numbers, nothing interpreted beyond the prediction); `TODO.md`.

### Task 5: the detector (CPU extraction, GPU training)

**Files:** create `experiments/vla_inputs_box_detector/{__init__.py,contract.py,data.py,model.py,runner.py,gate.py,README.md}` and `tests/`.

**Interfaces (the contract binds, like the policy packages):** the P2 seal and the S0 seal and the plan split (episodes 0–479 train, 480–539 validation) from the sealed records; `SOURCE_FILES`
with `test_fork_hygiene`-style equality; the constants **fixed before training**: frame stride k, input size, the gate thresholds (spec D5), presence threshold 0.5, augmentation ranges. `data.extract()`
decodes the sealed videos of the training episodes (LeRobot's own decoder, as the policy training reads them) at stride k into a uint8 cache under `outputs/vla_inputs_box_detector/cache_v1/` with a manifest
of (episode, frame, source hash); it measures decode speed and cache size first and **refuses** below the disk guard. `model.py`: a class-conditioned two-camera box regressor with presence (at most 50 M
parameters, no downloads). `runner.py --check | --train | --gate`. `gate.py`: the D5 measurements on decoded validation frames, on raw renders of replayed validation episodes (robosuite image), and on synthetic
out-of-view frames.
- [ ] **Step 1: failing tests:** the geometric augmentation transforms boxes exactly (hand-computed translations, clipping, absence), the instruction parser maps all ten trained instructions and
  rejects others, labels equal S0 `target_box` for chosen (episode, frame), a training batch never contains validation or any M0-tier layout, the loss ignores nothing silently (absent boxes train
  presence only), the gate fails for a deliberately bad predictor and for a predictor that always says "present".
- [ ] **Step 2:** measure decode throughput on one episode; choose k so extraction takes a few hours on at most 24 cores and the cache stays under about 20 GB; write k into the contract, commit, and only then extract.
- [ ] **Step 3:** `--check` (CPU: seals, split, a batch round trip), extract, `--train` from the detached `.worktrees/m3b-run` (claimed output, per-epoch checkpoints, resumable, validation IoU logged every epoch,
  the checkpoint chosen by validation IoU only). Poll `docker ps` and `result.json`.
- [ ] **Step 4: the gate (spec D5)** on the chosen checkpoint. **If it fails: stop, report the numbers, fix the detector using only training and validation data, re-run; never loosen a threshold.**
- [ ] Commit `M3b: detector`. Record the gate numbers in `docs/BENCHMARK_STATUS.md`.

### Task 6: arm A5 (predicted box)

**Files:** create `experiments/vla_inputs_arm_xvla_force_box_pred/{__init__.py,adapter.py,README.md}` and `tests/`.

**Interfaces:** `XVLAPredictedBoxAdapter(XVLAForceBoxAdapter)`: loads the detector from `outputs/vla_inputs_box_detector/<run>/final/` (an `INPUT_DIRS` entry, so `prepare` binds every file), runs it in `act` on
`obs["images"]` and the instruction, draws with the frozen `overlay.draw_frame`, logs per `act` a JSON line (tick, predicted box, the oracle box kept in a separate diagnostic attribute) to the trial's
`predicted_box_log.jsonl` beside `trace.npz` if the folder is derivable, otherwise to a per-process log keyed by `reset` count and matched to traces by the oracle box sequence (decide at build time, test the match on
two trials). Presence < 0.5 → nothing drawn.
- [ ] **Step 1: failing tests (robosuite image):** a stand-in detector returning the oracle box makes A5 equal A3 byte for byte (D7 ii); **leakage:** altering the diagnostic oracle changes nothing drawn, and a
  mutated adapter that draws the oracle fails this test (D7 iii); the detector sees raw uint8/255 frames exactly as the gate measured them; the log round-trips; the 17-value state is A3's.
- [ ] **Step 2–4:** forking pass, implement, pass, mutate (oracle drawn; stale prediction reused; cameras swapped; presence threshold ignored).
- [ ] **Step 5:** commit `M3b: predicted-box arm`; merge `--no-ff` into `main`; new detached `.worktrees/m3b-eval` commit for the run.

### Task 7: night B — A5, then the report

- [ ] Pre-launch checklist; `prepare` A5 (the detector directory mounted through `EXTRA_MOUNTS`, bound by the arm manifest) → `smoke` → `comparison` (6 workers × 7 cores, about 2.7 h plus the detector cost per tick).
- [ ] `analyse`: R with its interval, A5 against A2 (primary) and A3, A6–A9 against A3 with Holm, failure classes, the detector's per-tick error from the logs.
- [ ] `docs/VLA_INPUTS_M3B_<date>.md`: the result against the spec's prediction and the D10 branch it falls in, per-class counts against the predicted ranges, Q1–Q4 answered one by one (or stated as not
  answered), the caveats (forces privileged, one seed, closed-set detector, the A3-never-saw-noise confound). Docs pass in the same turn (`RESULTS.md`, `BENCHMARK_STATUS.md`, `TODO.md`, `README.md`, `CLAUDE.md`, `INDEX.md`,
  the programme and M3b specs, `FAILURES.md` for any mistake). Commit.
- [ ] State which D10 branch follows (M3b closed / M3b-2 / M4a next) and **ask the user**; do not start it. A download bundle for M3b is the user's call.
