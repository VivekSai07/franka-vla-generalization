# Props P3 — collection Implementation Plan

> **For agentic workers:** written 2026-10-19, success rule and layout counts revised 2026-10-20 (user accepted the revisions), to be
> executed in a new session after the user has confirmed the spec's decisions D1–D7. Steps use checkbox (`- [ ]`) syntax for tracking. Training and evaluation get their own plans after the seal.

**Goal:** Build `experiments/props_p3_data/`, pass the expert re-test and a sharded pilot, then collect, merge and seal 544 P3
episodes (fixed prompt, one object, food/drinks → blue/orange) with parallel CPU-rendered shards.

**Architecture:** A fork of `experiments/props_p2_data/` that removes the distractor machinery and keeps the sharded scheduler,
bounded model cache, verified merge and seal. New: the P3 plan (8 trained cells, 4 held-out cells, 288 reserved layouts, fixed
prompt), a task-label replacement in the episode sink, validators that accept one shared task string. `props_scene` is imported
read-only and unchanged.

**Tech Stack:** Python 3.10, robosuite 1.5.2 / MuJoCo 3.4.0 image, LeRobot 0.4.4, numpy.

**Spec:** [../specs/2026-10-19-props-p3-food-drinks-design.md](../specs/2026-10-19-props-p3-food-drinks-design.md).

## Task 0 (before any code): confirm and verify

- [x] **0.1** Confirm D1–D7 with the user, one at a time if any is contested; edit the spec with a dated line for each change
  **before** coding. The success thresholds are pre-registration: they are fixed now and not moved after any rollout.
  *(done 2026-10-20: all confirmed unchanged, recorded in the spec.)*
- [x] **0.2** Check the state the draft assumes (it was written without running any of this): `git worktree list`, `docker ps`
  (a live container may hash a tree you are about to edit), `git log --oneline -5`, whether `feat/props-p2-followup` and
  `feat/egl-rendering` were merged, free disk (`df -h /home`; the P2 collection needed the sealed root 3.2 GB plus 3.1 GB of
  shards). *(done 2026-10-20: the follow-up is on `origin/main`; `feat/egl-rendering` is not merged; 76 GB free.)*
- [x] **0.3** Create the branch with a worktree, never a checkout in a mounted tree:
  `git worktree add -b feat/props-p3 .worktrees/feat-props-p3 <base>` (see the `branching-for-an-experiment-axis` skill).

## Build decisions (2026-10-20, made while building; before the first `--collect`)

This plan is hash-bound by the seal's `provenance()` (FAILURES #52): it is not edited after `--collect` starts, so task
progress and results from Task 8 on live in the dated collection report, not here.

- **Expert: `props_scene.expert.PropsExpert` unchanged** (the spec says "the P0/P1 expert unchanged"). P2's 0.19 m carry
  existed to clear a standing distractor and has no purpose with one object. The episode wrapper is P1's single-object one,
  plus P2's bounded model cache (12 possible object-and-arrangement models per worker, at most 4 kept).
- **16 shards of 34 slots**, each pinned to 3 cores with `LP_NUM_THREADS=3` (544 does not divide by P2's 12). The thread cap
  was proven pixel-neutral for this package: 12 rendered gate attempts, capped at 3 against default threads, in two
  processes, byte-equal (traces and per-frame image digests).
- **No control tier** among the reserved layouts (nothing to control for without distractors): screen 24, report 240,
  confirm 24. The expert gate is 288 = 6 objects × 2 arrangements × 24, bars as P0/P2 (95% overall, 90% per object, 93% per slot).
- **The shared-prompt checks, re-derived (Review Focus 4), scoped:** the `labels` gate and `validate_dataset` require exactly
  `sort the item` in every frame, episode and the task table (P2 matched ten instructions by text); the bin must be the
  object's container and no held-out object may be a target, both checked on the plan and on the merged dataset. The training
  contract's equivalents (`INSTRUCTIONS`, `planned_instructions`, `task_index_map`, `approved_instruction`) belong to the training
  plan and are not touched here.
- Fork hygiene is tested (`test_contract`: no P1/P2 package reference, a `symtable` unbound-global scan, no distractor residue,
  a non-empty README), and every test was mutation-checked.

## Global Constraints

- Never edit: `experiments/props_scene/**`, `experiments/props_p1_data/**`, `experiments/props_p2_data/**`,
  `experiments/robosuite_scene/**`, `src/**`, `data_collection/**`, `experiments/paired_demonstrations/**`,
  `experiments/pose_snapshot_v2/**`, the P1 and P2 specs and plans. Import `props_scene` read-only.
- P3 must not import `experiments.props_p1_data` or `experiments.props_p2_data` (its provenance stays independent): copy the
  files you need and rename. Run the `forking-an-experiment-package` checks, including the `symtable` dead-name scan.
- Images: robosuite `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` for everything that builds the
  scene; admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` for tests that import no
  simulator. **Not** the EGL image. Flags as P2: `--read-only --network none --tmpfs /tmp:rw,size=16g
  --ulimit nofile=65536:65536 -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1
  -e PYTHONPATH=/vla`; pin cores per worker (`--cpuset-cpus`, `LP_NUM_THREADS`).
- Counts: 480 train (8 cells × 60), 64 validation (8 × 8), 544 admitted; reserved 288 = screen 24 + report 240 (80 trained + 160 held-out) + confirm 24, one rollout seed per layout.
- Success rule and P0/P1 gates never weakened; if a bar fails, find why, do not move the bar.
- Commits: user identity, no AI or co-author trailers, short messages; never commit `outputs/`; do not push (no credentials on
  this machine; the user pushes). Docs pass after every finding.

## Review Focus (each cost real time in P1/P2)

1. **A held-out object is collected** (test_plan, validate_dataset): bottle and cereal must not appear as a target in any
   training or validation slot.
2. **Container predicts object, or side predicts container.** Per-colour slot balance; every trained cell has the same count;
   the bin colour follows from the object exactly (drinks orange, food blue) and the slot is balanced independently.
3. **The expert's own instruction text reaches the dataset.** `props_scene.expert.task_text` writes "put the X in the Y bin"
   into every frame. The P3 sink must replace it with the fixed prompt, and a test must fail if any frame carries anything else
   (confirm the test fails against the unfixed sink).
4. **One shared task string breaks inherited assumptions.** P2 matches tasks to instructions by text and checks a bijection with
   10 instructions: `props_p2_data/validate.py` (instruction equality per episode, the `task_table` check around lines 44 and
   145), and in training `xvla_props_p2/contract.py` (`INSTRUCTIONS`, `planned_instructions`, `task_index_map`,
   `approved_instruction`, which sets `single_task.TASK`). Re-derive each check from P3's data model, scoped to what changed;
   do not relax it generally.
5. **A rejected attempt leaves frames in a shard, or a shard skips/duplicates a slot**; **the merge reorders, drops or alters
   an episode** (as P2).
6. **Forked residue**: no `props_p1_data` / `props_p2_data` import, no "distractor" machinery left half-removed, no P2 constants
   (`P2_`, 120 layouts, spacing) surviving.
7. **A rendering assumption.** OSMesa only; `LP_NUM_THREADS` must be proven pixel-neutral again for the worker count used
   (two determinism runs, capped against default, byte-equal), as in P2.

## File map (fork of `props_p2_data`)

| file | responsibility |
|---|---|
| `plan.py` | cells, held-out cells, balanced slot/arrangement decks, resample/attempt draws, 288 reserved layouts; pure |
| `contract.py` | `IMAGE`, `OUTPUT`, `provenance()` over this package + `props_scene` + shared + `src` + spec/plan; pure |
| `episode.py` | one attempt, bounded expert cache, **task-label replacement in the sink** |
| `validate.py` | `admit` (P1 gates, no distractor gate), `fingerprint`, `validate_dataset` (one shared task), `seal` |
| `record.py` | sharded scheduler; `--pilot/--collect/--resume/--status`; `--merge`; `--gate` |
| `merge.py` | `aggregate_datasets` over shards + verification against the shards |
| `README.md` | what, images, gate order, commands (hash-bound: write it **before** `--check`, confirm its size on disk) |
| `tests/` | `test_plan`, `test_contract`, `test_validate`, `test_record` (admission image); `test_merge`, `test_episode` (robosuite image) |

## Tasks

- [x] **1. Plan** — `plan.py` + `test_plan.py` (Review Focus 1–2). Admission image.
- [x] **2. Contract** — `contract.py` + `test_contract.py` (Focus 6; provenance stable; spec and plan bound).
- [x] **3. Episode, validate** — `episode.py`, `validate.py`; `test_validate.py` (admission image, fake replay),
  `test_episode.py` (robosuite image: the recorded task equals the fixed prompt, the expert's text never appears, bounded cache
  rebuild is bit-identical) (Focus 3–4).
- [x] **4. Sharded scheduler** — `record.py` + `test_record.py` (shard slot blocks partition 0..543 exactly; resume per shard)
  (Focus 5).
- [x] **5. Merge** — `merge.py` + `test_merge.py` (two tiny synthetic shards) (Focus 5).
- [x] **6. Expert re-test** — `record --gate`: 288 attempts, no rendering, bars in the spec; rendered attempts re-run in a second
  process for byte-equality; `LP_NUM_THREADS` pixel check (Focus 7). Report numbers; stop if a bar fails.
- [x] **7. Pilot** — 2 shards × 6 slots → merge → validate on a scratch root. Watch the first episodes.
- [ ] **8. Launch** — shards `--collect`, poll actively (`docker wait` has failed silently here); `--merge`; seal; independent
  re-verification in a second process.
- [ ] **9. Report + docs pass** — `docs/PROPS_P3_COLLECTION_<date>.md`, TODO, BENCHMARK_STATUS, INDEX, RESULTS if numbers
  change; mutation-check every new test (break the code, see it fail).

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka        # or the P3 worktree root
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
run() { local image=$1; shift; docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536 \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/vla -v "$MAIN:/vla:ro" -v "$MAIN/outputs/props_p3_data:/out:rw" -w /vla \
  --entrypoint python "$image" "$@"; }
run $OLD -m unittest discover -s experiments/props_p3_data/tests -t experiments/props_p3_data/tests -p 'test_[pcvr]*.py'
run $NEW -m unittest discover -s experiments/props_p3_data/tests -t experiments/props_p3_data/tests -p 'test_[me]*.py'
```

Output root: `outputs/props_p3_data/props_p3_<YYYYMMDD>/` (shards under `shards/`).

## Afterwards

Training (`experiments/xvla_props_p3/`, fork of `xvla_props_p2`: `--check`, `--preflight`, `--resume-proof`, `--equivalence`,
then `--train` in nights) and evaluation (`experiments/xvla_props_p3_eval/`, fork of `xvla_props_p2_eval`, classifier validated on
successes first, workers pinned) are separate plans. The P2 diagnostics code
(`experiments/xvla_props_p2_eval/diagnostics/`: replay, hand rise, closest approach, contact sheets) is reusable and not
hash-bound.
