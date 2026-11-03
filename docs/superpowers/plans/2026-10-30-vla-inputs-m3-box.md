# M3 — the target box drawn on both cameras, on top of M2's force state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Rewritten 2026-11-03** with the redesigned spec (numeric 19-value state against A0 replaced by a drawn box on A2's 17-value
state against A2; reasons in the spec's "What changed"). Nothing of the 2026-10-30 plan had been built.

**Goal:** X-VLA trained exactly as M2's run (arm A2) except that both camera frames carry the target's box drawn as a
magenta band, evaluated as arm A3 by M0 and tested against A2.

**Architecture:** `experiments/xvla_p2_force_box/`, a fork of `experiments/xvla_p2_force/` that imports M2's `forces` module
(state unchanged) and adds `overlay.py` (pure drawing) and `boxes.py` (sealed sidecar `target_box` reader); the loader draws in
`JoinedDataset.__getitem__`; a zero-equivalence gate against M2's runner; a rollout adapter whose state builder also returns
the live box for `act` to draw; `eval_compare.py` for the A3-against-A2 test.

**Tech stack:** admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (training, CPU tests),
robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (parity, rollouts).

**Spec:** [../specs/2026-10-30-vla-inputs-m3-box-design.md](../specs/2026-10-30-vla-inputs-m3-box-design.md) (D1–D11).
**Depends on:** S0 sealed, M0 frozen core `c4f6c303f2ae`, A2 trained and evaluated on C (done 2026-11-03).

## Global constraints

- **Frozen, never edited:** `experiments/xvla_p2_force/**` (A2's training binding and arm), `experiments/vla_inputs_signals/**`
  (S0 seal), `experiments/vla_inputs_eval/**` (M0 core), `experiments/xvla_props_p2/**`, `experiments/xvla_single_task/contract.py`,
  everything in P2's `provenance()`. Where behaviour must differ, fork into the new package.
- **Branch and worktree:** work on `feat/vla-inputs-m3` in `.worktrees/feat-vla-inputs-m3` (skill
  `branching-for-an-experiment-axis`). Train from a **detached worktree** at the gated commit (`.worktrees/m3-run`), as M2 did
  from `.worktrees/m2-run`. Never `checkout`/`switch`/`reset` in a mounted directory.
- **Recipe:** derived from `experiments.xvla_p2_force.contract.recipe()` and asserted equal in a test (apart from output paths and
  the purpose string), never retyped. State order and transform: M2's, via `experiments.xvla_p2_force.forces`.
- **Overlay (spec D3):** magenta `(1.0, 0.0, 1.0)`, 4 px band outside the half-open pixel box
  `[floor(512·u_min), ceil(512·u_max)) × [floor(512·v_min), ceil(512·v_max))`, clipped to 512×512, nothing drawn when `in_frame` is
  0, applied to float32 CHW frames in [0, 1] before LeRobot's processors; box values used as float64, exactly as stored or computed.
- **Resource guards as M2:** 65 GiB free disk, 28 GiB free GPU, thermal pause 83 → 75 °C; not relaxed.
- **Tests:** `-m unittest discover -s experiments/xvla_p2_force_box/tests -t experiments/xvla_p2_force_box/tests` in the
  admission image, mounting `outputs/props_p2_data` and `outputs/vla_inputs_signals` read-only; `eval_tests/` in the robosuite
  image with `-e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba`. Every new test is confirmed to fail against a deliberate mutation.
- Commits by the user's identity only, no AI trailers; merges `--no-ff`; nothing pushed without the user.

---

### Task 0: preconditions

- [ ] `docker ps` (nothing may be mounting the worktree you create), `nvidia-smi` (other users' jobs), `df -h /` (≥ 65 GiB + M2's
  margin), `git worktree list`. Record the numbers in the session notes.
- [ ] Create branch and worktree from `main`; load skills `forking-an-experiment-package`, `training-a-policy`,
  `launching-a-run`, `working-on-the-robosuite-path`.

### Task 1: fork

**Files:** copy `experiments/xvla_p2_force/` (without `eval_adapter.py`, `eval_tests/`) → `experiments/xvla_p2_force_box/`; replace
the copied `forces.py` by an import of `experiments.xvla_p2_force.forces`; modify `contract.py` (purpose `xvla_p2_force_box`,
`OUTPUT = outputs/train/xvla_p2_force_box_v1`, `REFERENCE_FILES` = M2's `__init__.py`, `contract.py`, `forces.py`, `runner.py`),
`runner.py` (`REFERENCE_RUNNER = 'experiments.xvla_p2_force.runner'`), `README.md`; adapt tests.

- [ ] **Step 1: failing tests:** recipe equals M2's value by value (and so P2's); the joined 17-value state for (0, 0), (18, k)
  with a clipped value, and (539, 451) equals M2's `joined_state` byte for byte; statistics equal M2's admission statistics
  byte for byte (state 17, action 8); `SOURCE_FILES` equals the modules importing contract and runner load (fork-hygiene test);
  `eval_adapter.py`, `eval_compare.py`, `eval_tests/` are absent from `SOURCE_FILES`.
- [ ] **Step 2:** forking-skill pass: grep the fork for `xvla_p2_force` (only the deliberate `forces` import and the reference
  runner may remain) and `xvla_props_p2`; symtable dead-name scan; claim tokens say `xvla_p2_force_box`.
- [ ] **Step 3–4:** implement, pass, mutate (a copied `forces.py` with one constant changed; wrong reference runner).
- [ ] **Step 5:** commit `M3: fork M2 for the drawn box`.

### Task 2: the overlay

**Files:** create `experiments/xvla_p2_force_box/overlay.py`, `tests/test_overlay.py`.

**Interfaces:** `THICKNESS = 4`, `COLOUR = (1.0, 0.0, 1.0)`, `SIZE = 512`;
`pixel_box(box5) -> tuple[int, int, int, int] | None` (None iff `in_frame` == 0; raises on values outside [0, 1], non-finite,
`in_frame` not in {0, 1}, or an empty box with `in_frame` 1); `band(box5) -> np.ndarray bool (512, 512)`;
`draw(image, box5) -> torch.Tensor` (float32 (3, 512, 512) in [0, 1], returns a copy, input untouched);
`draw_frame(frame, target_box) -> dict` (front → `observation.images.image`, gripper → `observation.images.image2`, `target_box`
shape (2, 5)).
- [ ] **Step 1: failing tests:** hand-computed pixel sets for an interior box, a box touching each image edge, a box covering the
  whole image (band empty), a one-pixel box; `in_frame` 0 draws nothing; band pixels exactly magenta, every other pixel equal to
  the input byte for byte; no pixel inside the pixel box changes; the input tensor is not mutated; dtype/shape/range checks;
  camera mapping.
- [ ] **Step 2–4:** fail, implement, pass, mutate (band inward; thickness 3; u/v swapped; floor/ceil swapped; cameras swapped;
  drawn when `in_frame` 0; in-place drawing). Each mutation must fail a test.
- [ ] **Step 5:** commit `M3: overlay`.

### Task 3: sidecar boxes, loader and input check

**Files:** create `boxes.py`, `tests/test_boxes_join.py`; modify `contract.py` (`JoinedDataset`, `check_model_inputs`), `runner.py`
(`check()` records the colour measurement and PNGs).

**Interfaces:** `boxes.table()` verifies the S0 seal (as `forces.table()` does) and returns `target_box` float64 (540, 452, 2, 5)
indexed by (episode, frame); `boxes.target_box(episode, frame) -> (2, 5)`. `JoinedDataset.__getitem__` = M2's join, then
`overlay.draw_frame(row, boxes.target_box(episode, frame))`, behind a single module-level hook `contract.OVERLAY` (the
zero-equivalence gate swaps it for the identity). `check_model_inputs(policy, batch, raw_state, rows, frames)`: M2's checks, plus
for every sample the pre-processor frames are exactly magenta on `overlay.band` of its sidecar box. `--check` additionally writes
`colour_check.json` (spec D4: fraction of near-magenta pixels, L∞ ≤ 0.1, on undrawn frames 0, 225, 451 of every 10th training
episode, both cameras; **fails above 0.01%**) and `overlay_samples/*.png` (spec D9).
- [ ] **Step 1: failing tests:** seal refusal on a tampered sidecar copy; box for (0, 0) equals the sidecar file's row; a shifted
  or duplicated row is rejected (M2's join rules); the joined row's frames equal `draw_frame(undrawn, box)`; the input check fails
  on an undrawn frame, a frame drawn with the wrong episode's box, and a camera swap; `in_frame` is 1 on every training row
  (asserted, S0's finding).
- [ ] **Step 2–4:** fail, implement, pass, mutate.
- [ ] **Step 5:** README of `xvla_p2_force_box` (non-zero size, states what differs from M2 and what is bound); commit
  `M3: sidecar boxes, loader, input check`.

### Task 4: zero-equivalence (GPU)

**Files:** modify `runner.py` (`--zero-equivalence`), `tests/test_zero_equivalence.py`.

- [ ] Pure test: the zero hook is the identity and is the only thing swapped.
- [ ] On GPU (after Task 6's `--check` and `--preflight`): 30 updates of M2's runner (`experiments.xvla_p2_force.runner`, imported
  read-only) against 30 of this runner with `OVERLAY` = identity, same seed: metrics, weights and Adam moments byte-equal,
  `zero_equivalence.json` written. Mutation (overlay left on, through a wrapper, no bound file edited) reported unequal.
- [ ] Commit `M3: zero-equivalence gate`.

### Task 5: rollout side, parity and the A3-against-A2 test

**Files:** create `experiments/xvla_p2_force_box/eval_adapter.py`, `eval_compare.py`, `eval_tests/test_box_parity.py`,
`eval_tests/test_eval_compare.py`; create `experiments/vla_inputs_arm_xvla_force_box/` (`__init__.py`, `adapter.py`
re-exporting, `README.md`, `tests/test_closure.py` as `vla_inputs_arm_xvla_force`'s); create `scripts/vla_inputs_m3_eval.sh`
(a copy of `scripts/vla_inputs_m2_eval.sh` for arm A3, its adapter and checkpoint directory, plus a `compare_a2` step).

**Interfaces:** `XVLAForceBoxAdapter(XVLAForceAdapter)`, `name = "xvla_force_box"`; its instance attribute `state_builder(model,
data, target)` returns M2's `state_force(...)` and stores `signals.target_box(model, data, target)` for this tick; `act(obs)`
takes the stored box (raises if none was stored since the last `act`, so a stale box can never be drawn), draws with
`overlay.draw_frame` after `uint8 / 255`, then does exactly what `XVLAForceAdapter.act` does. `INPUT_DIRS` as M2's.
`eval_compare.compare(root, checkpoint, control_checkpoint, seed=0)`: loads A3's and A2's comparison stages through M0's own
loaders (read-only import), refuses unless both prepared records carry core `c4f6c303f2ae`, both arm closures verify, layout ids
and dropped layouts are identical; exact McNemar and the object-stratified paired bootstrap from `vla_inputs_eval.stats`;
amended Holm over {A1 vs A0, A2 vs A0, A3 vs A2, A4 vs A0}, unrun at p = 1, reading A2's p from its existing compare file; writes
`arms/A3/compare_A2_<ckpt>.json`.
- [ ] **Step 1: failing tests (robosuite image):** on sealed episodes 0, 18, 271, 539 replayed with S0's `replay.frames`: live
  `target_box` equals the sidecar row byte for byte at every frame; the adapter's drawn frames differ from the undrawn render
  exactly on `overlay.band`, where they are magenta; the processed `image_input` for four frames equals the training processor's
  output for the same drawn frames; M2's 17-value state parity re-run. `act` without a fresh `state_builder` call raises.
  `eval_compare` on synthetic stage fixtures gives hand-computed b, c, p and Holm values and refuses on core, layout or drop
  mismatch.
- [ ] **Step 2–4:** fail, implement, pass, mutate (box from the previous tick; cameras swapped; drawn before the `/255`; control
  arm A0 instead of A2; Holm over the old family).
- [ ] M0 shared-core hash still `c4f6c303f2ae` (closure test against A0's `prepared.json`).
- [ ] Commit `M3: rollout adapter, parity, A3-against-A2 test`. Merge into `main` with `--no-ff` before training starts, so the
  gated commit is on `main`.

### Task 6: gates and training

- [ ] Pre-launch checklist (`launching-a-run` section 1): `docker ps`, `nvidia-smi`, `df -h /`, `--ulimit`, mounts (P2 data and S0
  sidecar read-only; output bind mount fresh and empty).
- [ ] `git worktree add --detach .worktrees/m3-run <gated commit>`; from it: `--check` (colour check passes; PNGs written) →
  **show the PNGs to the user and get approval (spec D9)** → `--preflight` → `--resume-proof` → `--equivalence` →
  `--zero-equivalence` (Task 4).
- [ ] `--train` without a deadline, under a supervisor adapted from M2's `outputs/train/xvla_p2_force_supervisor.sh` (resume
  from the newest verified snapshot after a death, at most 5 times). Poll with `docker ps` and the run's `status.json`, never
  `docker wait`.
- [ ] Status entries in `docs/BENCHMARK_STATUS.md` at launch and at each night boundary; `TODO.md` updated.

### Task 7: evaluation as A3

- [ ] `scripts/vla_inputs_m3_eval.sh`: `prepare --arm A3` (six candidates, matched-phase 101,520) → smoke → `selection` → `select`
  → `comparison` (winner + 101,520, deduplicated, 6 workers) → `compare` (A3 against A0, M0's own, secondary) → `compare_a2`
  (A3 winner against A2 88,830, primary; A3 101,520 against A2 101,520, matched phase).
- [ ] `docs/VLA_INPUTS_M3_<date>.md`: primary result against the pre-registered criterion; amended and M0-native Holm; the
  exploratory readouts of spec "Pre-registered criterion"; how often the gripper-camera box left the frame at rollout; the
  prediction (0 to +4) against the result. Docs pass (`RESULTS.md`, `BENCHMARK_STATUS.md`, `TODO.md`, programme spec, README);
  commit.
- [ ] Record whether M3b's start condition is met, with the numbers.
