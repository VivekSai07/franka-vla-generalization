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

- [x] **Done 2026-10-31 (CPU, admission image): yes, with no core edit.** Read `plan`, `feasibility`, `protocol`, `contract`,
  `evaluate`, `adapters`, `scenes`, `stats`; then ran a scratch check (a repository copy and the core's own fake-arm harness)
  on a pilot-shaped plan built from M0's layouts (192 comparison layouts, 4 per cell, an **empty** selection tier, 2 smoke
  layouts) through `load_manifests`, `prepare` (A0, A1), `worker`/`collect` on smoke and comparison, an empty selection
  stage, `select` and `compare`: all passed; `compare` reported 192 layouts and role "primary and matched-phase"; the core hash
  in this worktree is `c4f6c303f2ae…`, equal to A0's recorded one. What it needs, all outside the core:
  1. `plan/plan.json` + `plan/plan.sha256` (`props_p2_data.plan.digest`) in the new campaign root, with `tiers` holding `smoke`,
     `selection` (empty) and `comparison` (192), the layout schema of M0's `plan.py`, and layout `tier` values `smoke` /
     `comparison`; `contract.digest(plan)` must equal `p2.digest(plan)` (tested). Nothing pins M0's plan digest or namespace.
  2. **Adapters must live under `experiments/vla_inputs_arm_*`** (`adapters._IDENTIFIER`), so each pilot arm package is a thin
     module there that imports `xvla_p2_pilot.eval_adapter`, as M2's `vla_inputs_arm_xvla_force` does; it declares the variant
     JSON files in `EXTRA_SOURCE` so they are in the arm hash.
  3. **`compare` requires `arms/<arm>/selection.json`**, which `select` writes from collected selection stages. With one
     checkpoint and an empty selection tier the driver runs, per arm, an empty `worker` pass (it verifies the checkpoint and
     runs no trial), `collect`, `select` (winner = the only checkpoint, `tied` lists it) and declares that checkpoint as the
     matched-phase one at `prepare`. Ceremony, not selection: the report says so.
  4. **Arm names are `A0`..`A9` per campaign root** and `compare` is always against `A0`. In a pilot root: `A0` = the shared
     continuation control, `A9` = the untouched bundle checkpoint 101,520 (context), `A1`..`A8` = variants.
  5. Selection-tier `kind` checks and `final` (Holm over A1-A4) are never used by pilots.

### Task 2: pilot layout plan

**Files:** `experiments/vla_inputs_pilot_plan/{__init__,plan}.py`, `tests/test_plan.py`.
- [x] Failing tests: 48 cells × 4 = 192 in `comparison`, 2 smoke layouts, empty `selection`; namespace `vla_inputs:pilot`;
  disjoint (seed, start pose, distractor poses) from the sealed P2 plan, the P2 re-test plan and both M0 tiers; deterministic;
  schema accepted by M0's `feasibility.load_plan`.
- [x] Implement with P2's samplers exactly as M0's `plan.py` does (import, never copy-edit the core); pass; mutation-check the
  disjointness filter. Commit `Pilots: layout plan`.

**Done 2026-10-31** (commit `Pilots: layout plan`; 16 tests, admission image, CPU; the sealed P2 root mounted read-only). The generator
`experiments/vla_inputs_pilot_plan/plan.py` imports M0's `plan` read-only (`_deck`, `cell_name`, `layout_id`, `check_disjoint`,
`forbidden_values`) and P2's samplers; its forbidden set **adds M0's whole plan (both tiers)** to P2's and the re-test's. Shape: 192
`comparison` layouts (48 cells x 4; 32 held-out), an empty `selection` tier and **2 extra `smoke` layouts** (trained cells, two and one
distractors; not members of another tier, because the pilot has no selection tier). Mutation checks: with the disjointness filter
disabled, with M0 dropped from the forbidden set and with M0's namespace, 8 tests failed. The Task 1 scratch check is now a permanent
test on the real pilot plan (`CoreServesThisPlan`: M0's own fake-arm harness through prepare, worker, collect, an empty selection stage,
select and compare, with one layout dropped for every arm in a second case; and the core hash equals A0's `c4f6c303f2ae`).

### Task 3: continuation runner

**Files:** fork `experiments/xvla_props_p2/` → `experiments/xvla_p2_pilot/` (`contract.py`, `runner.py`, `README.md`, tests);
`experiments/xvla_p2_pilot/variants/{control,force,box}.json`.
- [x] Failing tests: base weights are the bundle's 101,520 (`model.safetensors` hash equals its marker); optimizer state is
  fresh; learning rate is 0→1e-5 over 200 updates then constant; horizon 10,000 from `recipe()`; the sample order for a fixed
  seed is identical across variants; a variant file outside the schema is rejected; the variant's sha256 lands in the run
  record; `variants/*.json` and `eval_adapter.py` are absent from `SOURCE_FILES`.
- [x] State builders: `control` (9 P2 values, extras zero), `force` (M2's clip + standardise transform imported read-only from
  `xvla_p2_force`, hash recorded), `box` (S0 `target_box`, raw). Local model-input check as M2/M3.
- [x] Equality gate `--zero-equivalence --variant V`: 30 updates of `V` with extras forced to zero byte-equal to `control`.
- [x] Only the final weights are kept (no snapshots; a crashed pilot is re-run, the run is deterministic). Disk per pilot ~3.3 GB.
- [x] Fork hygiene (residue grep, symtable). Commit `Pilots: continuation runner`.

**Done 2026-10-31, CPU only** (commit `Pilots: continuation runner`; 80 tests, admission image; 8 targeted mutations each failed at
least one test: zero flag ignored, warm-up 100, stale optimizer accepted, another sampler, raw extras unchecked, optimizer state saved,
base weights as the start, variant hash left out of the marker). What differs from the plan text:
- **Fork source is M2, not P2's runner.** `xvla_p2_pilot` imports M2's (`xvla_p2_force`) constants, split, sampler (`sample_batches`,
  so the order is identical by construction), dataset checks and `forces.py` read-only, and copies M2's loop plumbing
  (`one_update`, the fused checks, `Prefetcher`, the thermal guard, the byte-exact reload) by AST segment; `tests/test_runner.py`
  asserts the 14 copied definitions equal M2's runner's. M2's runner gained `--zero-equivalence` on main meanwhile (P2's loop as the
  reference); the copied functions were not touched by it, and the branch was merged with `main` before the final runs.
- **The state is 19 values for every variant** (9 joints, 10 extras padded with zeros): control extras zero, `force` M2's 8, `box` S0's
  `target_box` (front camera then gripper camera, raw, 10). The model pads to 20 anyway, so control equals P2's input and force equals
  M2's byte for byte. The box table is pinned (`BOX_TABLE_SHA256` `2475c91e...`; `in_frame` is 1 on every sealed frame).
- **The start is pinned file by file** (A0's recorded hashes for the 6 small files, `59d02d0a...` for the weights) and the action
  statistics the processors use must equal the ones saved in the start checkpoint (checked by `--check` and `--train`).
- **Admission is inline**, not a stored proof: `--check`, `--zero-equivalence` and `--train` each run the full CPU admission of the variant
  (both fingerprints recomputed, binding, rows, processor and model-input check). `--check` adds the strict load of the start weights
  and a fresh optimizer. **Real runs on the real data, no GPU:** `--check` passed for `control`, `force` and `box`.
- **`--zero-equivalence`** compares the variant with every extra forced to zero against the control, in this runner's loop, for 30
  updates (metrics, weights, Adam moments). It does not re-prove M2's loop equals P2's (M2's own gate does; `one_update` is M2's by AST).
- **`--updates N`** (`--train` only) runs a prefix of the same schedule as a dry run, checkpoint `step_<N>`, purpose `xvla_p2_pilot_dry`.
- No snapshots, no resume, no optimizer state on disk: one checkpoint (`step_010000`, 3.3 GB), a crashed pilot is re-run in a new directory.
- **Not exercised without a GPU:** a real update, the checkpoint save and byte-exact reload, the thermal pause, the disk and GPU guards on a
  real device. The synthetic-policy tests run the real `train` and `zero_equivalence` with a one-layer model and patched GPU calls.

### Task 4: arm adapter

**Files:** `experiments/xvla_p2_pilot/eval_adapter.py`, `experiments/xvla_p2_pilot/eval_tests/`.
- [x] Failing tests: the adapter loads a pilot checkpoint and the variant's state builder; for 3 S0 episodes replayed through
  S0's `replay.frames`, the rollout state equals the training rows byte for byte (normalised and padded); the M0 core hash is
  unchanged after adding the adapter. Commit `Pilots: arm adapter`.

**Done 2026-10-31** (commit `Pilots: arm adapter`; 19 stub and parity tests in the robosuite image, 6 closure tests; 5 targeted mutations
failed tests). **One adapter class serves every variant:** a pilot checkpoint's marker names its variant and the variant file's sha256;
`XVLAPilotAdapter` loads that file, refuses a changed one and takes the state builder from it (`LIVE`: `control`, `force` = M2's own
`state_force` then two zeros, `box` = S0's `target_box`). So the arm closure does not list variant files (adding a variant never changes a
prepared arm) and a checkpoint cannot be evaluated with the wrong builder. Identifier:
`experiments.vla_inputs_arm_xvla_pilot.adapter:XVLAPilotAdapter` (thin arm package, as M2's). **Parity:** over replayed sealed episodes
0, 18 and 271 (all 452 frames each) the live state equals `contract.joined_state` byte for byte for all three builders, and the
normalised, padded `proprio` the policy builds equals the training batch's. The shared core hash is unchanged (`c4f6c303f2ae`).

### Task 5: night driver and morning summary

**Files:** `scripts/vla_inputs_pilot_night.sh`, `experiments/xvla_p2_pilot/summary.py`, tests.
- [x] `vla_inputs_pilot_night.sh <night_root> <variant>...`: (once per recipe version) control continuation + its evaluation +
  A0 on the pilot tier; then per variant: zero-equivalence gate → continuation → `prepare` → pilot-tier evaluation (6 workers,
  one checkpoint per worker) → `compare` against the control. Sequential; polls `docker ps`; resumes an interrupted
  evaluation; a failed gate skips that variant and is reported, never retried silently.
- [x] `summary.py`: per variant Δ, the decision under the fixed rule, McNemar p, mechanism counts, wall-clock, any failure;
  writes `summary.json` and `SUMMARY.md`. Tests on fabricated rows only for arithmetic; the rule tested at Δ = 4.9 and 5.0.
- [x] Commit `Pilots: night driver`.

**Done 2026-10-31** (commit `Pilots: night driver`; 56 rollout-side tests incl. 20 driver tests against a fake `docker` and 17 summary
tests; 12 targeted mutations failed tests; the one survivor, a redundant control-failure guard, led to a stricter test).
- **The night root is the campaign root** of M0's frozen core (`plan/`, `feasibility/`, `arms/`), with `training/<variant>[_zeroeq]/`,
  `events.jsonl`, `night.json`, `summary.json`, `SUMMARY.md` beside them, so the pilot checkpoints are visible through the evaluator's
  `/out` mount. Arms: `A0` the shared control (the continuation), `A1`..`A8` the variants in the order given, `A9` the untouched bundle
  checkpoint (context; not needed for any decision). The M0 driver is reused unmodified for feasibility, stage, select and compare; the
  night driver has its own `prepare` (the phase argument), training and summary commands.
- **Safety:** the driver refuses to start while anything uses the GPU (`nvidia-smi` compute apps; `PILOT_ALLOW_BUSY=1` overrides), mounts
  the worktree read-only, polls `docker ps` (never `docker wait`), keeps a failed training directory and never retries it, resumes an
  interrupted evaluation, skips finished steps on a second invocation. Run it from a **dedicated worktree** (it creates the empty
  mountpoint directories inside it), never the checkout M2 trains from.
- **`summary.py`:** Δ = exact integer arithmetic (`promote iff 20*(new-control) >= layouts`), tested at 4.9 and 5.0 points and at the real
  count (9 of 192 drops, 10 promotes); McNemar p, discordant counts, the paired effect interval, all mechanism counts, wall clock per phase,
  failures; a variant whose gate, training or evaluation did not complete gets **no decision** and the reason. The gripper-camera target
  loss named in the spec is not a field the evaluator records, so it is not reported. Every drop is marked provisional until Task 7.

### Task 6: dry night (after M1 training ends)

- [x] Pilot plan → feasibility (CPU) → smoke (2 trials) for control and A0 → a 300-update continuation of `force` end to end
  through the driver, to time each stage and confirm the summary. Then pilot night 1 for real: control, A0, `force`, `box`.
  **Done 2026-11-02 with changes:** the first dry night failed on a file list (FAILURES #68), the re-run passed; night 1 ran
  control and `force` only (A9 skipped, `box` never run).
- [x] Morning: write `docs/VLA_INPUTS_PILOTS_<date>.md` (decisions, Δ, mechanism counts, wall-clock), status log, TODO.
  [VLA_INPUTS_PILOTS_20261102.md](../../VLA_INPUTS_PILOTS_20261102.md) (written 2026-11-03).

**Pre-launch notes (2026-10-31):** the driver is built and tested on the CPU; the GPU is held by M2's training (started 17:02, about
18-20 h), so nothing in this task has run, and the driver refuses to start while the GPU is in use. When it is free: create a dedicated
worktree for the night (`git worktree add .worktrees/pilot-run <commit>`; never the checkout M2 trains from), run the dry night first
(`UPDATES=300 PILOT_SMOKE=1 WT=<that worktree> scripts/vla_inputs_pilot_night.sh <night_root> force`: the 2 smoke layouts of every arm, then the pilot tier), read `SUMMARY.md` and the wall-clock lines to size
the real night, then the real one. The first GPU evidence for the pilot runner is the zero-equivalence gate and this dry run; expect to find
at least one GPU-only defect there. Disk: about 3.3 GB per trained pilot, so a night of control plus two variants is about 10 GB plus the
evaluation logs.

### Task 7: calibration

- [x] After full M2 is evaluated on M0's comparison tier, compare with the force pilot: sign of Δ, interval overlap. Record in
  the pilot report; apply the spec's calibration rule (pilots may only promote, not drop, if they disagree). **Done 2026-11-03:**
  same sign, intervals [−1.6, +7.9] and [8.9, 16.4] do not overlap; the rule applies.
