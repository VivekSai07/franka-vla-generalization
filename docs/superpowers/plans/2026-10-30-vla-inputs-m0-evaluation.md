# M0 — common evaluation protocol and the control arm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** a policy-agnostic evaluator on 656 fresh P2-style layouts (80 selection, 576 comparison), the paired statistics, and
the control arm A0 (X-VLA 101,520 from the P2 bundle) measured on the comparison tier.

**Architecture:** new package `experiments/vla_inputs_eval/` (shared core frozen and hash-bound before A0; arm adapters separately bound): `plan.py` (layouts), `scenes.py` (restore, forked),
`adapters.py` (frozen interface and lazy loader; implementations in arm packages), `rollout.py` (forked from P2's, plus S0 signals in the
trace), `stats.py`, `evaluate.py` (stages), driver `scripts/vla_inputs_eval.sh`. P2's success rule and classifier are imported
unchanged from `experiments/xvla_props_p2_eval/analysis.py`.

**Tech stack:** robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` for rollouts (GPU),
admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` for pure steps.

**Spec:** [../specs/2026-10-30-vla-inputs-m0-evaluation-design.md](../specs/2026-10-30-vla-inputs-m0-evaluation-design.md).
**Depends on:** S0 Tasks 2–3 before equivalence development; S0 signal source finalized and sealed before preparing A0.

## Global constraints

- Worktree `.worktrees/feat-vla-inputs`, branch `feat/vla-inputs`. `docker ps` before editing anything a run might read.
- Tiers: smoke 2 (⊂ S), **S = 80** (40 trained cells × 2), **C = 576** (48 cells × 12; user decision 2026-10-30). One rollout seed per layout, shared by
  every arm. Never re-run a completed trial.
- Cells: objects `can, milk, bread, lemon, bottle, cereal` × bins `blue, orange` × blue-bin side `left, right` × distractors
  `1, 2`. Held-out pairs: `milk→blue`, `bread→orange` (C only).
- Identity namespace `vla_inputs:m0:<tier>:<cell>:<i>`.
- Selection: candidates = the six checkpoints after update 1; winner = most S successes; tie → later checkpoint.
- Primary per arm vs A0: exact two-sided McNemar on C, α = 0.05; effect with a paired bootstrap stratified by object (10,000,
  seed 0); Holm for the fixed A1–A4 family reported alongside (unrun arms p=1); original power is unadjusted.
- Workers pinned (`--cpuset-cpus`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `LP_NUM_THREADS` = cores per worker), sized by the peak
  GPU memory measured in smoke (FAILURES #57). Poll `docker ps`.
- OSMesa only. A0 never under EGL.
- Test commands: as in the S0 plan (robosuite image) for simulator tests; for pure tests the admission image with
  `-m unittest discover -s experiments/vla_inputs_eval/tests -t experiments/vla_inputs_eval/tests`. GPU tests add
  `--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8` and mount `outputs/downloads_20261017/props_p2` read-only.

---

### Task 1: layout plan

**Files:** create `experiments/vla_inputs_eval/__init__.py`, `plan.py`, `tests/test_plan.py`.

**Interfaces:**
- Consumes (read-only): P2's samplers in `experiments/props_p2_data/plan.py` (start region, yaw caps, spacing, hand clearance,
  distractor decks; the P2 re-test used them through `p2._deck` and the start sampler — reuse the same entry points).
- Produces: `plan.make_plan() -> dict` with `layouts: {layout_id: {tier, cell, object, bin, blue_slot, distractor_count,
  distractors, start, home_pose_jitter_deg, seed, held_out: bool}}` and `tiers: {"smoke": [...], "selection": [...],
  "comparison": [...]}`; `plan.write(out)` (writes `plan.json` and its sha256; no overwrite).

- [ ] **Step 1: failing tests:** counts per tier and per cell (S: 2 per trained cell, none held-out; C: 12 per cell, 576 total);
  smoke ⊂ S; deterministic (two calls equal byte for byte after `json.dumps(sort_keys=True)`); every seed distinct; **disjoint
  from P2** (no seed, start pose or distractor pose equal to any layout in the sealed P2 `plan.json`, episodes and
  `evaluation_holdout`) and from the P2 re-test plan (`experiments.xvla_props_p2_retest.plan`); distractor identities in each
  cell drawn from shuffled decks (every option before any repeat), as P2.
- [ ] **Step 2–4:** fail, implement, pass; mutations: drop the disjointness filter (a test must fail on a forced collision
  fixture built from a real P2 layout), off-by-one in a cell count.
- [ ] **Step 5:** commit `M0: layout plan`.

### Task 2: restore and feasibility

**Files:** create `scenes.py`, `feasibility.py`, `tests/test_scenes.py`.

**Interfaces:**
- Produces: `scenes.restore(layout: dict) -> (model, data, names)` (target first; asserts 1 mm / 1° like P2's);
  `feasibility.run_shard(out, k, shards)` (unrendered expert per layout; jsonl rows `{layout_id, feasible, reason}`);
  `feasibility.merge(out) -> {"feasible": [...], "dropped": [...]}`.
- [ ] **Step 1: failing tests:** restoring one P2 reserved layout through `scenes.restore` gives a full state
  (`mj_getState`, `mjSTATE_FULLPHYSICS`) byte-equal to `experiments.xvla_props_p2_eval.scenes.restore(layout_id)`; every S ∪ C
  layout restores; feasibility of 2 layouts runs and writes 2 rows.
- [ ] **Step 2–4:** fail, implement, pass, mutate (yaw tolerance check removed → a test with a rotated fixture fails).
- [ ] **Step 5:** commit `M0: restore and feasibility`.

### Task 3: statistics

**Files:** create `stats.py`, `tests/test_stats.py`.

**Interfaces:** `stats.mcnemar(b: int, c: int) -> float` (exact two-sided); `stats.paired_effect(rows, n=10_000, seed=0) ->
{"estimate", "lo", "hi"}` (bootstrap over layouts stratified by object); `stats.holm(pvalues: dict) -> dict`;
`stats.select(successes_by_checkpoint: dict[int, int]) -> int` (max, tie → later).

- [ ] **Step 1: failing tests:** `mcnemar(15, 8)` ≈ 0.21 (the recorded P2 101,520 vs 126,900 result, report tier);
  `mcnemar(0, 0) == 1.0`; symmetric; bootstrap on a fixture where every layout has d = +1 gives (1, 1, 1); stratification keeps
  per-object counts; `select({101520: 19, 126900: 19}) == 126900`; Holm on a three-value fixture computed by hand.
- [ ] **Step 2–4:** fail, implement, pass, mutate (one-sided p, unstratified resampling, tie → earlier).
- [ ] **Step 5:** commit `M0: statistics`.

### Task 4: adapters

**Files:** create `adapters.py`, `tests/test_adapters.py`.

**Interfaces:**
- `class PolicyAdapter(Protocol): name: str; def reset(self) -> None; def act(self, obs: dict) -> np.ndarray  # (8,)`
- `obs` keys: `images` (two uint8 512 × 512 × 3, `front_cam` then `gripper_cam`), `state` (np.ndarray, as built by the arm's
  state builder), `instruction` (str).
- `StateBuilder = Callable[[MjModel, MjData, str], np.ndarray]`; `state_p2(model, data, target) -> data.qpos[:9]` (float64
  copy; cast as P2's rollout does).
- `XVLAAdapter(checkpoint: Path, state_builder)`: loads as `xvla_props_p2_eval.evaluate.load_policy` does (config from the
  checkpoint, weights via the same loader, asserts chunk 30 / 10 actions), builds `pre`/`post` with
  `make_pre_post_processors(policy.config)` and the checkpoint's statistics, runs `select_action` under `torch.inference_mode`
  and bf16 autocast exactly as P2's rollout, and returns the 8 physical commands.
- [ ] **Step 1: failing tests (GPU):** the adapter's action on a fixed observation equals P2's code path
  (`pre(observation(frame))` → `post(policy.select_action(...))`) byte for byte; `act` returns shape (8,).
- [ ] **Step 2–4:** fail, implement, pass, mutate.
- [ ] **Step 5:** commit `M0: policy adapters`.

### Task 5: rollout and the equivalence gate

**Files:** create `rollout.py`, `tests/test_rollout_equivalence.py`.

**Interfaces:**
- `rollout.run(adapter, state_builder, model, data, names, renderer, trial, folder) -> dict` — the loop of
  `experiments/xvla_props_p2_eval/rollout.py` with the adapter in place of the inline policy call, plus, right after
  `synchronize_observation`, `signals.wrist_wrench`, `signals.finger_force`, `signals.target_box` appended to the trace
  (keys `wrist_wrench`, `finger_force`, `target_box`). Same `cameras.mp4`, `trace.npz` (P2's keys + the three), `episode.json`.
- [ ] **Step 1: failing test (GPU, robosuite image):** the equivalence gate. From the bundle's recorded report trials of
  101,520, take **the two lowest-ordinal successes and the two lowest-ordinal failures** (rule fixed here, before running);
  run them through `rollout.run` with `XVLAAdapter(bundle step_101520, state_p2)`; assert every P2 trace array and the
  `episode.json` fields `outcome`, `cause`, `ticks`, `placed_bin`, `first_lifted` byte-equal to the recorded ones.
- [ ] **Step 2–4:** fail, implement, pass. Mutation: compute the signals on `data` instead of a scratch copy → must fail if the
  computation writes anything the rollout reads (if it does not fail, document why the mutation is equivalent).
- [ ] **Step 5:** commit `M0: rollout with signals, equivalence gate`.

### Task 6: stages and driver

**Files:** create `evaluate.py`, `README.md`, `tests/test_evaluate.py`, `scripts/vla_inputs_eval.sh`.

**Interfaces:** `evaluate.py prepare --arm A --checkpoints ...` (hash every checkpoint against its marker; write
`arms/<arm>/prepared.json` with separate shared-core and arm-specific source closures, immutable layout/feasibility
manifests, runtime digest, processor/statistics/weights hashes, and S0 provenance; core equality with A0 is mandatory); `worker --arm A --tier T --checkpoint N --shard k --shards n` (interleaved jobs, one
folder per trial, skip existing complete trials); `collect --arm A --tier T`; `select --arm A` (S only); `compare --arm A`
(C, against A0; shared-core hashes must match A0 and each arm must match its own adapter/runtime manifest); `final`.
- [ ] **Step 1: failing tests (pure):** a resumed worker skips complete trials and redoes incomplete ones; a worker only ever
  loads the one checkpoint it was launched with (no in-process switch; FAILURES #57); `compare` refuses on a
  module-hash mismatch; `select` reads only S; `compare` reads only C; held-out layouts never reach `select`.
- [ ] **Step 2–4:** fail, implement, pass, mutate. Fork hygiene: grep for `xvla_props_p2_eval` outside the deliberate read-only
  imports; symtable dead-name scan.
- [ ] **Step 5:** commit `M0: stages and driver`.

### Task 6b: audit and freeze the protocol before A0

- [ ] Freeze the shared adapter interface/loader; implementations are lazy imports from separately bound modules. Test with
  an X-VLA-only environment and a fake future adapter that adding an arm does not alter shared-core hashes.
- [ ] Mutation tests: modifying shared rollout, analysis, signal source or layout manifest rejects comparison; modifying an
  arm adapter after prepare rejects that arm; a different declared adapter with identical core is accepted.
- [ ] Require complete feasible pairs before comparison. Infrastructure interruption resumes missing trials; nonfinite
  policy output counts as failure. Record exclusions by cell and define the resulting population.
- [ ] Recompute power before policy outcomes at actual feasible N, including alpha .0125 as the conservative first Holm
  threshold for four tests; report assumptions and Monte Carlo error. Do not claim the old .05 table is adjusted power.
- [ ] Add the secondary fixed 4.0-epoch comparison per arm, deduplicating the S winner when identical; budget up to 576 extra
  trials. Never use C to choose between them. Freeze recipes/analysis before C; subsequent C-informed redesign needs fresh
  confirmation layouts. Name results as conditional on the trained seed.
- [ ] Record ties explicitly. The later-checkpoint rule is an administrative choice, not evidence that it is better.
- [ ] Verify S0 signal source is finalized/bound before A0; no later edits to the shared signal tap.

### Task 7: run A0

- [ ] Pre-launch checklist ([launching-a-run](../../../.github/skills/launching-a-run/SKILL.md) section 1).
- [ ] `plan` → `feasibility` (CPU, sharded) → merge; record dropped layouts.
- [ ] `verify_bundle.py` on the P2 bundle; `prepare --arm A0 --checkpoints 101520`.
- [ ] Smoke: measure peak memory including checkpoint switches and loading, not only two steady-state trials (use a separate resource probe or selection-tier jobs; never rerun completed C trials).
  Worker count must fit actual available GPU/RAM/CPU with headroom. Evidence (spec, What is established): 8 non-switching
  X-VLA workers ran at ~4.2 GB each in P3; switching workers reached 7.3 GiB in P2 and one died. **Never let a worker switch
  checkpoint:** one checkpoint per worker pool (A0: 8 workers; selection: one worker per candidate).
  Schedule GPU rollouts separately from M1 training unless combined measured peaks fit. Budget video storage.
- [ ] A0 on C (576 minus dropped). Poll. `collect`.
- [ ] Write `docs/VLA_INPUTS_M0_<date>.md` (plan, feasibility, A0 per object, per cell, failure causes, signals recorded), update
  `docs/RESULTS.md`, `docs/BENCHMARK_STATUS.md`, `TODO.md`, `docs/INDEX.md`; commit `Docs: M0 control arm`.
