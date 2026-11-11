# G1 — domain-randomized dataset and X-VLA training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** a sealed 3,000-episode randomized dataset with live force signals, X-VLA trained on it with A2's inputs and recipe, evaluated on G0.
**Spec:** [../specs/2026-11-06-g1-randomized-data-design.md](../specs/2026-11-06-g1-randomized-data-design.md) (D1–D10).
**Depends on:** G0 Tasks 1–3 (pools, engine, expert) merged; G0's evaluator and A2's G0 result before G1 is evaluated (not before it is built).
**Status 2026-11-10:** G0 is done (`main` `9356869`); its engine is merged and **bound by G0's core** (`gen_scene`, `gen_eval` frozen).
Read the spec's revision R1-R4 first: distractors 0-5, D3 expressed through `gen_scene.Condition(pool="train")` only, the per-object expert gate, the new prediction.

## Global constraints

As the [G0 plan](2026-11-06-g0-generalization-benchmark.md), plus: the collector uses **train pools only** (asserted on every episode); the G0 test
pools are never read by `gen_g1_data` (a test fails if they are); the dataset is sealed before any training reads it and never edited after.
Skills: `collecting-a-dataset`, `forking-an-experiment-package`, `training-a-policy`, `launching-a-run`.

---

### Task 1: collection plan (CPU)

**Files:** `experiments/gen_g1_data/{plan.py,contract.py,README.md}` (fork of `props_p2_data`), tests.
- [ ] 3,000 slots: P2's 10 trained instruction pairs (milk→blue, bread→orange held out), blue side alternating, a `gen_scene.Condition(pool="train")`
  per slot sampled from the D3 ranges as amended by R1-R2, with a fixed master seed (the sampler lives in `gen_g1_data`, never in `gen_scene`); 2,700 train / 300 validation; namespace `gen:g1`, disjoint from every G0 tier and every earlier plan.
- [ ] Tests: ranges respected; train pools only; disjointness; plan digest pinned. Commit `G1: collection plan`.

### Task 2: recorder, live signals, expert gate (CPU, robosuite image)

**Files:** `experiments/gen_g1_data/{episode.py,record.py,merge.py,validate.py}` (forks), tests.
- [ ] Per frame: images (both cameras), 9-value state, action, and S0 signals computed live with the imported S0 functions (wrist wrench, finger
  forces, target box), stored per episode; LeRobot dataset as P2's, with the signals in the per-episode npz.
- [ ] Parity test: for the nominal condition, a recorded episode's signals equal S0's sidecar computation for the same replayed state byte for byte.
- [ ] **Expert gate (D4, R3):** a 96-episode pilot across the ranges; ≥ 95% overall, ≥ 90% per table-height quintile, per distractor count and per object; contact
  sheets of 24 pilot episodes → **show the user, wait for approval.** If the gate fails: stop, report, fix the expert with re-gating; never lower a threshold.
- [ ] Commit `G1: recorder and gate`.

### Task 3: collection, merge, seal (CPU)

- [ ] Check disk (≥ 65 GiB after the expected 17 GB). 16 shard containers (`--collect ... --shard K`, cores pinned, `LP_NUM_THREADS` as P3);
  `--status` while running; merge, validate, seal; expert re-test on a sample, determinism check twice. Report `docs/GEN_G1_COLLECTION_<date>.md`.

### Task 4: training package and gates (admission image, GPU for gates)

**Files:** `experiments/xvla_g1_rand/` (fork of `xvla_p2_force`: dataset binding, sidecar source = the sealed G1 signals, constants derived from G1
training rows and pinned), `experiments/vla_inputs_arm_xvla_g1/` (arm adapter, A2's adapter with G1's constants), tests.
- [ ] Equality gate (D7): the G1 runner on P2's data and S0 with A2's constants equals M2's runner byte for byte over 30 updates.
- [ ] `--check` → `--preflight` → `--resume-proof` → `--equivalence` → the equality gate, from a detached worktree `.worktrees/g1-run`.
- [ ] Train-rollout parity: the arm's live state equals the training rows on replayed G1 episodes (nominal and two randomized conditions).
- [ ] `--train` with `--stop-at` windows around other users' jobs; supervisor as M2's; status entries.

### Task 5: evaluation and report

- [ ] `gen_eval` prepare G1 (six candidates, matched phase 101,520) → smoke → selection tier (all conditions) → select → benchmark tier at the
  winner and 101,520 → compare against A2 per axis (Holm over five, D10).
- [ ] M0 comparison tier (nominal, 573) at the winner, against A2 (secondary).
- [ ] Report `docs/GEN_G1_<date>.md` against the prediction; docs pass; commit. State whether G2's start condition holds (an axis 5 level not held)
  and **ask the user** before starting G2.
