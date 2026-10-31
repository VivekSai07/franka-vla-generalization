# Overnight pilots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** a one-command overnight queue that trains continuation pilots from A0, evaluates them on a 192-layout pilot tier,
and leaves a morning summary with promote/drop decisions.

**Architecture:** training package `experiments/xvla_p2_pilot/` (admission image; fork of `xvla_props_p2`'s runner with a
continuation recipe and data-defined variants), arm adapter `experiments/xvla_p2_pilot/eval_adapter.py` (outside
`SOURCE_FILES`), pilot plan generator `experiments/vla_inputs_pilot_plan/` (outside M0's frozen core), night driver
`scripts/vla_inputs_pilot_night.sh`.

**Spec:** [../specs/2026-10-31-vla-inputs-overnight-pilots-design.md](../specs/2026-10-31-vla-inputs-overnight-pilots-design.md).
**Depends on:** S0 sealed; M0 core frozen; M2's extra-state transform (`xvla_p2_force`, built by session 1) for the force
variant; the GPU free of M1 training for Task 6.

## Global constraints

- `experiments/vla_inputs_signals/` (sealed), `experiments/vla_inputs_eval/` (frozen core) and `experiments/xvla_props_p2/**`
  are never edited.
- Continuation recipe constants live in one `recipe()`: 10,000 updates, warm-up 200 to 1e-5 constant, effective batch 8, P2's
  optimizer settings, fixed sample seed. Derived, never retyped (training-a-policy skill).
- Promotion rule: Δ ≥ +5 points on the paired pilot tier. Fixed here; never changed after a pilot result.
- No offline metric decides anything.
- Merges with `git merge --no-ff`; commits in the user's identity; no push.

---

### Task 1: can M0's frozen core run a second campaign? (check before building)

- [ ] Read `experiments/vla_inputs_eval/{evaluate,feasibility,protocol,plan}.py`. Confirm that `plan`/`feasibility`/`prepare`/
  `worker`/`collect`/`compare` accept a campaign root whose `plan/plan.json` was written by another generator with the same
  schema, that `comparison` can hold the 192 pilot layouts and `selection` can be empty, and that nothing pins the M0 plan
  digest. Write the finding into this plan. If any of it fails, stop and propose a versioned wrapper (no core edit).

### Task 2: pilot layout plan

**Files:** `experiments/vla_inputs_pilot_plan/{__init__,plan}.py`, `tests/test_plan.py`.
- [ ] Failing tests: 48 cells × 4 = 192 in `comparison`, 2 smoke layouts, empty `selection`; namespace `vla_inputs:pilot`;
  disjoint (seed, start pose, distractor poses) from the sealed P2 plan, the P2 re-test plan and both M0 tiers; deterministic;
  schema accepted by M0's `feasibility.load_plan`.
- [ ] Implement with P2's samplers exactly as M0's `plan.py` does (import, never copy-edit the core); pass; mutation-check the
  disjointness filter. Commit `Pilots: layout plan`.

### Task 3: continuation runner

**Files:** fork `experiments/xvla_props_p2/` → `experiments/xvla_p2_pilot/` (`contract.py`, `runner.py`, `README.md`, tests);
`experiments/xvla_p2_pilot/variants/{control,force,box}.json`.
- [ ] Failing tests: base weights are the bundle's 101,520 (`model.safetensors` hash equals its marker); optimizer state is
  fresh; learning rate is 0→1e-5 over 200 updates then constant; horizon 10,000 from `recipe()`; the sample order for a fixed
  seed is identical across variants; a variant file outside the schema is rejected; the variant's sha256 lands in the run
  record; `variants/*.json` and `eval_adapter.py` are absent from `SOURCE_FILES`.
- [ ] State builders: `control` (9 P2 values, extras zero), `force` (M2's clip + standardise transform imported read-only from
  `xvla_p2_force`, hash recorded), `box` (S0 `target_box`, raw). Local model-input check as M2/M3.
- [ ] Equality gate `--zero-equivalence --variant V`: 30 updates of `V` with extras forced to zero byte-equal to `control`.
- [ ] Only the final weights are kept (no snapshots; a crashed pilot is re-run, the run is deterministic). Disk per pilot ~3.3 GB.
- [ ] Fork hygiene (residue grep, symtable). Commit `Pilots: continuation runner`.

### Task 4: arm adapter

**Files:** `experiments/xvla_p2_pilot/eval_adapter.py`, `experiments/xvla_p2_pilot/eval_tests/`.
- [ ] Failing tests: the adapter loads a pilot checkpoint and the variant's state builder; for 3 S0 episodes replayed through
  S0's `replay.frames`, the rollout state equals the training rows byte for byte (normalised and padded); the M0 core hash is
  unchanged after adding the adapter. Commit `Pilots: arm adapter`.

### Task 5: night driver and morning summary

**Files:** `scripts/vla_inputs_pilot_night.sh`, `experiments/xvla_p2_pilot/summary.py`, tests.
- [ ] `vla_inputs_pilot_night.sh <night_root> <variant>...`: (once per recipe version) control continuation + its evaluation +
  A0 on the pilot tier; then per variant: zero-equivalence gate → continuation → `prepare` → pilot-tier evaluation (6 workers,
  one checkpoint per worker) → `compare` against the control. Sequential; polls `docker ps`; resumes an interrupted
  evaluation; a failed gate skips that variant and is reported, never retried silently.
- [ ] `summary.py`: per variant Δ, the decision under the fixed rule, McNemar p, mechanism counts, wall-clock, any failure;
  writes `summary.json` and `SUMMARY.md`. Tests on fabricated rows only for arithmetic; the rule tested at Δ = 4.9 and 5.0.
- [ ] Commit `Pilots: night driver`.

### Task 6: dry night (after M1 training ends)

- [ ] Pilot plan → feasibility (CPU) → smoke (2 trials) for control and A0 → a 300-update continuation of `force` end to end
  through the driver, to time each stage and confirm the summary. Then pilot night 1 for real: control, A0, `force`, `box`.
- [ ] Morning: write `docs/VLA_INPUTS_PILOTS_<date>.md` (decisions, Δ, mechanism counts, wall-clock), status log, TODO.

### Task 7: calibration

- [ ] After full M2 is evaluated on M0's comparison tier, compare with the force pilot: sign of Δ, interval overlap. Record in
  the pilot report; apply the spec's calibration rule (pilots may only promote, not drop, if they disagree).
