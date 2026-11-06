# G0 — generalization benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** the frozen G0 evaluator and the shared randomization engine, then A2 and A0 (A3 descriptive) measured on five axes paired with nominal.
**Spec:** [../specs/2026-11-06-g0-generalization-benchmark-design.md](../specs/2026-11-06-g0-generalization-benchmark-design.md) (D1–D14).
**Programme:** [../specs/2026-11-06-generalization-programme-design.md](../specs/2026-11-06-generalization-programme-design.md).

**Tech stack:** robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (scene, expert, feasibility,
rollouts; run with `-e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba`), admission image
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (pure tests, analysis). Host python has no numpy.

## Global constraints

- **Frozen, never edited (import read-only, fork where behaviour differs):** `experiments/props_scene/**`, `experiments/robosuite_scene/**`,
  `experiments/props_p2_data/**`, `experiments/vla_inputs_signals/**`, `experiments/vla_inputs_eval/**` (M0 core `c4f6c303f2ae`),
  `experiments/xvla_props_p2/**`, `experiments/xvla_p2_force/**`, `experiments/xvla_p2_force_box/**`, every `experiments/vla_inputs_arm_*`
  package that has been prepared (A0, A2, A3, A5–A9), `experiments/xvla_single_task/contract.py`, and everything in any seal's `provenance()`.
  Check membership with the `diagnosing-a-binding-failure` and `forking-an-experiment-package` skills before touching a shared helper.
- **Branch and worktrees:** work on `feat/generalization` (this branch; worktree `.worktrees/feat-generalization`). Merge into `main` with
  `--no-ff` before anything runs from it; rollouts run from a detached worktree `.worktrees/g0-eval` at that merge. Never
  `checkout`/`switch`/`reset` a mounted directory.
- **Fixed in code before the first policy trial, each with a pinning test:** every level's parameters (D5), the pools' split rule and manifest
  digest (D6), tick caps (D7), the disturbance triggers, the `held` definition (D13).
- **Guards not relaxed:** 65 GiB free disk, an idle GPU (no other compute process), thermal pause 83 → 75 °C, 6 workers × 7 cores, one checkpoint
  per worker. Poll `docker ps` and stage logs; never `docker wait`.
- Tests first; every new test confirmed to fail against a deliberate mutation. Docs pass in the same turn as any finding. Commits as
  VivekSai07 <viveksaisurya07@gmail.com>, short messages, no AI trailers; nothing pushed; nothing deleted under `outputs/` or `datasets/`.

---

### Task 1: texture assets and pools (host + CPU)

**Files:** create `experiments/gen_scene/{__init__.py,pools.py,assets_manifest.json,README.md}`, `experiments/gen_scene/tests/test_pools.py`;
a host-side fetch script `scripts/gen_fetch_textures.sh` (never run inside a container; containers stay `--network none`).
- [ ] Choose about 60 table-like and 60 floor/wall CC0 textures (ambientCG or Poly Haven, 1K PNG colour maps only), download on the host into
  `.cache/gen_textures/`, record name, source URL, licence and sha256 in `assets_manifest.json`; commit the manifest only.
- [ ] `pools.split()`: a fixed rule (sha256 of the file name, sorted, every 5th to test) giving a **train pool** and a **test pool** per surface;
  primitive pools A (train) and B (test) defined in code with disjoint shape families and colour sets.
- [ ] Tests: manifest hashes match the files; the split is deterministic and disjoint; a mutation moving one file across pools fails.
- [ ] Commit `G0: texture pools`.

### Task 2: the randomization engine (CPU, robosuite image)

**Files:** `experiments/gen_scene/{conditions.py,compose.py}`, tests.

**Interfaces:** `Condition` (frozen dataclass, JSON round-trip, `digest()`), `NOMINAL`; `compose.build(objects, blue_slot, condition) -> (env, model)`
cached by (objects, slot, condition digest) with a bounded cache like M0's; lights and colours applied on the compiled `MjModel`, textures,
table offset (table, bins and the drop height move together), bin shift and primitive distractors applied to the scene XML before compiling;
`compose.reset(model, poses, home_jitter_deg)` equal to `props_scene.scene.reset` for nominal.
- [ ] **Gate D9 (i):** for three sealed P2 episodes, the nominal condition's model renders the initial state byte-equal to `props_scene` (both
  cameras, 512 px, OSMesa).
- [ ] Tests: each level's parameters are applied (read back from the model); table offset moves every on-table body by exactly the offset;
  a primitive from pool B never appears with a train condition; contact sheets render for every level.
- [ ] Mutations: lights scaled twice; table moved without the bins; a test texture used in a train condition. Commit `G0: randomization engine`.

### Task 3: the forked expert (CPU)

**Files:** `experiments/gen_scene/expert.py` (fork of `props_scene/expert.py`, reading bin positions and table height from the model instead of
constants), tests.
- [ ] **Gate D9 (ii):** under nominal, the fork reproduces the qpos traces of three sealed P2 episodes (S0's `replay` gives the sealed initial
  states and actions) byte for byte.
- [ ] Expert success on 48 attempts per spatial level and per distractor level (unrendered); report rates. They feed feasibility; nothing is tuned
  to raise them except fixing a demonstrated expert bug, which is then re-gated.
- [ ] Commit `G0: expert reading the scene`.

### Task 4: the evaluator core (CPU)

**Files:** `experiments/gen_eval/{contract.py,plan.py,feasibility.py,scenes.py,rollout.py,protocol.py,evaluate.py,stats.py,README.md}` forked from
`experiments/vla_inputs_eval/`, plus `tests/`.
- [ ] `plan.py`: namespace `gen:g0`; the 144-layout benchmark tier, the 24-layout selection tier, 2 smoke layouts; disjointness from P2's plan,
  the re-test, M0's plan and the pilot plan (seeds and poses) asserted in a test; the stratified level rotation (D4) and every trial's condition.
- [ ] `rollout.py`: M0's loop plus the condition and the disturbance hooks (`move` trigger at the first tick the hand is within 6 cm horizontally
  of the target before any close command; `drop` trigger at the first tick the target is 5 cm above its start, gripper forced open for 10 ticks);
  `grasp_lift`, `recovered`, `disturbance_fired`, `disturbance_tick` recorded; tick cap per D7.
- [ ] `feasibility.py`: D8, sharded, unrendered, per condition; dropped trials recorded by cell and level.
- [ ] `evaluate.py`: prepare (binds the core, the arm closure, the texture directory and checkpoints), worker, collect, select, compare (per axis
  and level, D13), final; refuses mismatched cores or plans as M0 does.
- [ ] **Gate D9 (iii):** with the nominal condition, `gen_eval` re-runs A2's recorded M0 comparison trials 1 and 291 (their M0 layouts given
  directly) bit for bit in every trace array, **under M0's worker runtime** (deterministic algorithms, the same CPU thread count).
- [ ] Tests and mutations for the trigger logic (fires once, only before a close for `move`, only after a lift for `drop`), the level rotation, the
  `held` rule, the bootstrap; commit `G0: evaluator`.

### Task 5: driver, feasibility, visual check

- [ ] `scripts/gen_eval.sh` (fork of `scripts/vla_inputs_eval.sh` and `vla_inputs_m3_eval.sh`): `plan`, `feasibility`, `prepare <arm>`, `smoke`,
  `selection`, `benchmark`, `compare`, `final`; tests against a fake `docker`.
- [ ] Run `plan` and `feasibility` (CPU, sharded). Report feasible counts per axis and level.
- [ ] Contact sheet of every level (D10) → **show the user, wait for approval.**
- [ ] Merge into `main` with `--no-ff`; `git worktree add --detach .worktrees/g0-eval <merge>`.

### Task 6: G0 night (GPU)

- [ ] Pre-launch checklist (`launching-a-run`); `prepare` A2 (88,830) and A0 (101,520); `smoke`; benchmark tier A2 then A0 (6 workers × 7 cores).
  A3 (101,520) only if the user says so after A2/A0.
- [ ] `compare` per arm; `final`.
- [ ] Report `docs/GEN_G0_<date>.md`: every axis and level, paired drops with intervals, failure classes, `grasp_lift`, `recovered`, against the
  prediction (per axis and class). Docs pass (`RESULTS.md`, `BENCHMARK_STATUS.md`, `TODO.md`, `README.md`, `CLAUDE.md`, `INDEX.md`, the
  programme spec). Commit.
