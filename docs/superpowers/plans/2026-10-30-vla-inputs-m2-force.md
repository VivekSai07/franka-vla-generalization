# M2 — force/torque in the robot state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** X-VLA trained exactly as P2's run except that the state carries 8 more values (wrist wrench 6, finger forces 2),
evaluated as arm A2 by M0.

**Architecture:** `experiments/xvla_p2_force/`, a fork of `experiments/xvla_props_p2/` whose data loader joins the sealed S0
sidecar; a local model-input check for 17 values; a zero-equivalence gate against P2's runner; a state builder in M0's evaluator
that calls S0's `signals.py`.

**Tech stack:** admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (training), robosuite
image for rollouts.

**Spec:** [../specs/2026-10-30-vla-inputs-m2-force-design.md](../specs/2026-10-30-vla-inputs-m2-force-design.md).
**Depends on:** S0 sealed (sidecar + report), M0 Tasks 1–6 and A0 on C.

## Global constraints

- Recipe identical to `experiments/xvla_props_p2/contract.py` `recipe()`: 126,900 updates, microbatch 4 × accumulation 2,
  chunk 30, 10 executed actions, the LR schedule, `CHECKPOINT_UPDATES = (1, 50760, 76140, 88830, 101520, 114210, 126900)`,
  snapshots every 2,600, the same sampler seed. Values are **derived from the P2 contract and asserted equal in a test**, never
  retyped.
- State order: `qpos[:9]`, `wrist_wrench` (Fx, Fy, Fz, Tx, Ty, Tz), `finger_force` (left, right).
- Clip ranges: the 0.1/99.9 percentiles in S0's sealed `report.json`, read at run time after verifying the S0 seal.
- Do not edit `experiments/xvla_single_task/contract.py`, `experiments/xvla_props_p2/**`, or anything in S0 after its seal.
- Resource guards as P2: 65 GiB free disk, 28 GiB free GPU; neither relaxed.
- Test commands: admission image, `-m unittest discover -s experiments/xvla_p2_force/tests -t experiments/xvla_p2_force/tests`,
  mounting `outputs/props_p2_data` and `outputs/vla_inputs_signals` read-only.

---

### Task 1: fork and sidecar join

**Files:** copy `experiments/xvla_props_p2/` → `experiments/xvla_p2_force/`; modify `contract.py`; create
`tests/test_sidecar_join.py`; adapt the copied tests.

**Interfaces:** `contract.SIDECAR` (S0 root and seal record), `contract.extra_state(episode_index, frame_index) -> np.ndarray (8,)`
(clipped, float32), `contract.OBS = {"observation.state": (17,), ...}`, `contract.collate_rows` producing 17-value states.
- [ ] **Step 1: failing tests:** the joined row for (episode 0, frame 0) equals `concat(sealed state, clipped sidecar values)`;
  the loader refuses a sidecar whose seal does not verify; every training row has a sidecar row (452 × 480); clip ranges equal
  S0's `report.json` values; the recipe equals P2's (`xvla_p2_force.contract.recipe(False) == xvla_props_p2.contract.recipe(False)`
  apart from output paths and the purpose string).
- [ ] **Step 2:** forking-skill pass: grep `xvla_props_p2` in the fork (only the deliberate reference import for the
  equivalence gate may remain), symtable dead-name scan.
- [ ] **Step 3–4:** implement, pass, mutate (unclipped values; validation rows in statistics; wrong column order).
- [ ] **Step 5:** commit `M2: fork, sidecar join`.

### Task 2: statistics and the model-input check

**Files:** modify `contract.py`; create `tests/test_inputs.py`.

**Interfaces:** `contract.selected_stats(...)` over 17 state values; `contract.check_model_inputs(policy, batch, raw_state)` — a
local copy of the shared check with `REAL_STATE_DIM = 17`: proprio shape (n, 20), `proprio[:, :17]` equals the normalised
state, `proprio[:, 17:]` is zero, domain id 0, image and token shapes as P2.
- [ ] **Step 1: failing tests:** the check passes on a real batch; it fails if a 17-value state is truncated to 9; statistics
  shapes are (17,) for state, (8,) for action; the first 9 entries of the state statistics equal P2's admission statistics
  byte for byte.
- [ ] **Step 2–4:** fail, implement, pass, mutate.
- [ ] **Step 5:** commit `M2: statistics and input check`.

### Task 3: zero-equivalence gate

**Files:** modify `runner.py`; create `tests/test_zero_equivalence.py`.

**Interfaces:** `runner.py --zero-equivalence`: in one process, run 30 updates of P2's loop (`experiments.xvla_props_p2.runner`,
imported read-only) and 30 updates of this loop with the 8 extra values forced to 0 after normalisation, from the same seed;
compare metrics, weights and Adam moments byte for byte; write `zero_equivalence.json`.
- [ ] **Step 1: failing test (pure part):** the forcing helper zeroes exactly columns 9–16 after normalisation.
- [ ] **Step 2–4:** implement; on GPU, run `--zero-equivalence`: equal. Mutation: leave one extra column un-zeroed → the gate
  must report unequal (run once, record, restore).
- [ ] **Step 5:** commit `M2: zero-equivalence gate`.

### Task 4: train/rollout parity

**Files:** modify `experiments/vla_inputs_eval/adapters.py` (add `state_force`); create
`experiments/vla_inputs_eval/tests/test_force_parity.py`.

**Interfaces:** `adapters.state_force(model, data, target) -> np.ndarray (17,)` = `qpos[:9]`, `signals.wrist_wrench`,
`signals.finger_force`, clipped with the same S0 ranges.
- [ ] **Step 1: failing test (robosuite image):** for episodes 0, 271, 539 replayed with S0's `replay.frames`, `state_force` at
  every frame, cast to float32, equals `xvla_p2_force.contract`'s joined training row byte for byte.
- [ ] **Step 2–4:** fail, implement, pass, mutate (unclipped; reordered).
- [ ] **Step 5:** README of `xvla_p2_force` written (non-zero size checked); commit `M2: rollout state builder, parity`.

### Task 5: gates and training

- [ ] Pre-launch checklist; `docker ps`; `nvidia-smi`; `df -h .` (65 GiB guard).
- [ ] `--check` → `--preflight` → `--resume-proof` → `--equivalence` (inherited fast-vs-serial loop) → `--zero-equivalence` →
  `--train --stop-at <deadline>` nights from a detached worktree.
- [ ] Night status entries in `docs/BENCHMARK_STATUS.md`.

### Task 6: evaluation as A2

- [ ] `prepare --arm A2` (six candidates) → smoke → S → `select` → C with the winner → `compare --arm A2`.
- [ ] `docs/VLA_INPUTS_M2_<date>.md`: primary result, `never_lifted` and milk (exploratory), force traces of failing milk trials
  for A0 and A2, against the spec's prediction; docs pass; commit.
