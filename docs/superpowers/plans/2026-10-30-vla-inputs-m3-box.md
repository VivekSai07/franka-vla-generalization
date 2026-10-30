# M3 — the target box in the robot state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** X-VLA trained exactly as P2's run except that the state carries the target's box in both cameras (10 values),
evaluated as arm A3 by M0.

**Architecture:** `experiments/xvla_p2_box/`, forked from `experiments/xvla_p2_force/` **after A2's evaluation is complete**;
only the selected sidecar arrays, the state size (19), the clip rule (none) and the local input check change.

**Tech stack:** as M2.

**Spec:** [../specs/2026-10-30-vla-inputs-m3-box-design.md](../specs/2026-10-30-vla-inputs-m3-box-design.md).
**Depends on:** S0 sealed, M0, A2 evaluated (so M2's package is never edited).

## Global constraints

- Everything in the M2 plan's global constraints, with: state order `qpos[:9]`, front camera
  `(u_min, v_min, u_max, v_max, visible)`, gripper camera `(u_min, v_min, u_max, v_max, visible)`; no clipping.
- `experiments/xvla_p2_force/**` is frozen once A2's training started.

---

### Task 1: fork

**Files:** copy `experiments/xvla_p2_force/` → `experiments/xvla_p2_box/`; modify `contract.py`, `runner.py` (purpose strings,
output paths only), `README.md`; adapt tests.

**Interfaces:** `contract.extra_state(episode_index, frame_index) -> np.ndarray (10,)` from S0's `target_box`; `OBS` state (19,);
`check_model_inputs` with `REAL_STATE_DIM = 19`.
- [ ] **Step 1: failing tests:** joined row = sealed state + `target_box.reshape(10)` for (episode 0, frame 0) and a frame where
  `gripper_cam` sees no target (visible 0, four zeros); recipe equals P2's; statistics shapes (19,); first 9 statistics equal
  P2's byte for byte; the input check fails on a truncated state.
- [ ] **Step 2:** forking-skill pass: grep for `xvla_p2_force` and `force` residue (sidecar keys, clip helper), symtable scan.
- [ ] **Step 3–4:** implement, pass, mutate (camera order swapped; `visible` dropped; clip applied).
- [ ] **Step 5:** commit `M3: fork for the target box`.

### Task 2: zero-equivalence and parity

**Files:** modify `runner.py`; modify `experiments/vla_inputs_eval/adapters.py` (add `state_box`); create
`tests/test_zero_equivalence.py`, `experiments/vla_inputs_eval/tests/test_box_parity.py`.

- [ ] `--zero-equivalence` against P2's runner, columns 9–18 forced to 0 after normalisation: equal over 30 updates; mutation run
  recorded.
- [ ] `state_box(model, data, target)` equals the joined training rows byte for byte on episodes 0, 271, 539 (S0 replay).
- [ ] README written (non-zero size); commit `M3: gates`.

### Task 3: gates, training, evaluation

- [ ] Pre-launch checklist; `--check` → `--preflight` → `--resume-proof` → `--equivalence` → `--zero-equivalence` → `--train`
  nights.
- [ ] `prepare --arm A3` → smoke → S → `select` → C → `compare --arm A3`.
- [ ] `docs/VLA_INPUTS_M3_<date>.md`: primary result; `wrong_object` (paired) and held-out-pair success (exploratory); A3 against
  A2 (descriptive); the spec's prediction and the "inconclusive below ~8 points" reading; docs pass; commit.
- [ ] Record whether M4a's start condition (M4 spec D1) is met, with the numbers.
