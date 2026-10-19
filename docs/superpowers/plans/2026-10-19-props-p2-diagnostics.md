# Props P2 failure diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the 12 never-lifted, 7 wrong-object and the milk failures of the P2 evaluation from replayed traces, with no new rollouts.

**Architecture:** Replay each recorded trial's `applied_action` through the restored scene, unrendered, recording hand pose and contacts per tick (Gate 0: the replay must equal the recorded `state` and `props_xyz` exactly). A sealed-expert forward-kinematics pass defines the grasp envelope per object. Pure modules turn the per-tick arrays into a first-cause taxonomy and layout-level statistics.

**Tech Stack:** Python 3.10, MuJoCo 3.4.0 (robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e`), numpy, matplotlib; pure modules and tests also run in the admission image (`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`, no scipy: do not import it).

## Global Constraints

Spec: [2026-10-19-props-p2-diagnostics-design.md](../specs/2026-10-19-props-p2-diagnostics-design.md) (committed `5b88080`). These apply to every task:

- No GPU, no new rollouts, no change to anything hash-bound (`experiments/props_p2_data/**`, `props_scene/*.py`, `robosuite_scene/*.py`, `src/`, the P2 spec and collection plan). New code lives in `experiments/xvla_props_p2_eval/diagnostics/` (not bound: the evaluation package has no contract). Do not edit `experiments/xvla_props_p2_eval/*.py`.
- Closed run: at least 5 consecutive ticks with applied gripper command < 127.5. First closure: first tick of the first closed run. Grasp closure: first tick of the closed run containing the lift tick (lifted trials), else the first closure.
- Held: both fingers in contact with the object on at least 10 consecutive ticks after the first closure. Lifted: `analysis.lift_tick` (2 cm, 15 ticks).
- Envelope: per object, at the expert's grasp closure, per hand-frame axis p1 to p99 over that object's expert episodes where it is the target; yaw error between finger axis and object yaw modulo 90 degrees, p1 to p99. Defined from the expert before any policy failure is read.
- Gate 0: replay equals recorded `state` and `props_xyz` exactly on all 180 trials of 101,520 (and every other trial replayed). Gate 1: at least 0.80 of 101,520's successes have their grasp closure inside the expert envelope; otherwise stop and ask the user.
- Taxonomy order (first cause): `wrong_object`, `lifted_then_failed`, `never_closed`, `closed_on_distractor`, `closed_off_target`, `closed_not_held`, `held_not_lifted`.
- At most six statistical tests; label anything not backed by physics or a replay measurement exploratory.
- Evidence before claims; run tests in the stated image and read the output. New tests must fail against a deliberately broken implementation (mutation check) before the task is closed.
- Commits are authored solely by the user: no AI or co-author trailers, short messages.

## File Structure

| file | responsibility | image |
|---|---|---|
| `diagnostics/__init__.py` | package marker | both |
| `diagnostics/grasp.py` | pure: closed runs, closures, hand-frame transform, yaw error, envelope, held | both |
| `diagnostics/taxonomy.py` | pure: first-cause classification of one trial from its measures | both |
| `diagnostics/stats.py` | pure: layout-concentration permutation, Mann-Whitney permutation test | both |
| `diagnostics/replay.py` | replay one trial; per-tick hand pose, finger positions, contacts; Gate 0 | robosuite |
| `diagnostics/reference.py` | expert grasp envelope by forward kinematics over the 540 sealed episodes | robosuite |
| `diagnostics/run.py` | CLI: `gate0`, `reference`, `measure`, `analyze` (shardable) | robosuite |
| `diagnostics/contact_sheet.py` | PNG frame strips of a trial's video for the manual check | robosuite |
| `diagnostics/tests/test_*.py` | tests for each module | per module |
| `docs/PROPS_P2_DIAGNOSTICS_<date>.md` | the report | n/a |

Test command (admission image, pure modules), from the repo root:

```bash
docker run --rm --runtime runc --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$PWD/outputs/xvla_props_p2_eval:/vla/outputs/xvla_props_p2_eval:ro" \
  -w /vla --entrypoint python sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m unittest discover -s experiments/xvla_props_p2_eval/diagnostics/tests -t experiments/xvla_props_p2_eval/diagnostics/tests
```

The robosuite-image variant adds `--ulimit nofile=65536:65536 -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=4 -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache` and the robosuite image digest.

---

### Task 1: `grasp.py` (pure)

**Files:** Create `diagnostics/__init__.py`, `diagnostics/grasp.py`, `diagnostics/tests/test_grasp.py`.

**Interfaces:**
- Produces:
  - `closed_runs(command, min_run=5, threshold=127.5) -> list[tuple[int,int]]` : `(first_tick, last_tick)` inclusive of every run of at least `min_run` consecutive ticks with `command < threshold`.
  - `first_closure(command) -> int | None`; `grasp_closure(command, lift_tick) -> int | None` (closed run containing `lift_tick`, else first closure; `None` if no run).
  - `to_hand_frame(hand_pos, hand_rot, point) -> np.ndarray(3)` with `hand_rot` a flat 9 vector row-major `R` (hand-to-world): `R.T @ (point - hand_pos)`.
  - `yaw_error_deg(finger_axis_xy, object_yaw_deg) -> float` in `[-45, 45)`: finger axis yaw minus object yaw, modulo 90.
  - `envelope(samples) -> dict` with `lo`, `hi` (4-vectors: x, y, z hand-frame and yaw error): p1 and p99 of `samples` (N x 4).
  - `inside(envelope, sample) -> np.ndarray(4, dtype=bool)` per component, inclusive.
  - `held_ticks(left_contact, right_contact, start, min_run=10) -> bool`: both fingers touch on `min_run` consecutive ticks at or after `start`.

- [ ] **Step 1: Write `test_grasp.py` first.** Cases (write them as `unittest.TestCase` methods with these literal inputs and expectations):
  - `closed_runs([255]*3+[0]*4+[255]*2+[0]*5+[255], 5)` returns `[(9, 13)]` (the 4-tick run is chatter).
  - `first_closure` of an all-255 command is `None`; `grasp_closure([255,0,0,0,0,0,255,0,0,0,0,0,0], lift_tick=9)` returns 7, and with `lift_tick=None` returns 1.
  - `to_hand_frame` with identity rotation, hand at (1,2,3), point (1.5,2,3.25) gives (0.5,0,0.25); with a 90 degree rotation about z (`R = [[0,-1,0],[1,0,0],[0,0,1]]`) and point (1,3,3) the result is (1,0,0).
  - `yaw_error_deg((1,0), 0) == 0`, `yaw_error_deg((0,1), 0) == 0` (90 degrees is the same grasp), `yaw_error_deg((1,0), 30) == -30`, `yaw_error_deg((1,0), 50) == 40`.
  - `envelope` of the integers 0..100 in every column: `lo == 1`, `hi == 99`; `inside` is inclusive at both ends and False one ulp outside.
  - `held_ticks`: left and right both True on ticks 20..29 gives True for `start=15`, False for `start=21`, False if only left is True.

- [ ] **Step 2: Run it, confirm it fails** (`ModuleNotFoundError: grasp`).
- [ ] **Step 3: Implement `grasp.py`** to the interfaces above (numpy only).
- [ ] **Step 4: Run, confirm every test passes.** Mutation check: change `<` to `<=` in `closed_runs` and the threshold test must fail; restore.
- [ ] **Step 5: Commit** `git add experiments/xvla_props_p2_eval/diagnostics && git commit -m "P2 diagnostics: grasp definitions"`.

### Task 2: `taxonomy.py` (pure)

**Files:** Create `diagnostics/taxonomy.py`, `diagnostics/tests/test_taxonomy.py`.

**Interfaces:**
- Consumes: the evaluation row fields `cause`, `task_success`, `first_lifted`, `object`.
- Produces: `classify(row, m) -> str`, with `m` a dict: `closed` (bool: any closed run), `target_in_env` (bool at first closure), `distractor_in_env` (list[bool] at first closure), `held` (bool), `lifted` (bool). Returns one of `success`, `wrong_object`, `lifted_then_failed`, `never_closed`, `closed_on_distractor`, `closed_off_target`, `closed_not_held`, `held_not_lifted`. `CLASSES` is the tuple of the seven failure classes in spec order.

- [ ] **Step 1: Test.** Literal cases: success returns `success`; `cause='wrong_object'` returns `wrong_object` whatever `m` says (including `closed_on_distractor` measures); `cause='timeout_carrying'` with `lifted=True` returns `lifted_then_failed`; `cause='never_lifted'` with `closed=False` returns `never_closed`; closed, `target_in_env=False`, `distractor_in_env=[True]` returns `closed_on_distractor`; same with `distractor_in_env=[False]` returns `closed_off_target`; `target_in_env=True`, `held=False` returns `closed_not_held`; `held=True` returns `held_not_lifted`; `target_in_env=True` and `distractor_in_env=[True]` returns `closed_not_held` (the target is in envelope, so it is not a distractor grasp); an unknown cause raises `ValueError`.
- [ ] **Step 2: Run, confirm FAIL. Step 3: Implement. Step 4: Run, PASS; mutation: swap rows 4 and 5, the distractor test must fail; restore. Step 5: Commit** `P2 diagnostics: taxonomy`.

### Task 3: `stats.py` (pure)

**Files:** Create `diagnostics/stats.py`, `diagnostics/tests/test_stats.py`.

**Interfaces:**
- Produces: `concentration(failures_by_layout: dict[str, int]) -> int` (sum of squared counts); `concentration_p(failed: dict[str, list[bool]], n=10000, seed=0) -> float` (permute the failure labels over all trials keeping each layout's trial count, observed statistic against permutations, one-sided p = (count >= observed + 1) / (n + 1)); `mann_whitney_p(pos, neg, n=20000, seed=0) -> float` (two-sided permutation p on the U statistic of the pooled values, deterministic for a seed).

- [ ] **Step 1: Test.** (a) `concentration({'a':3,'b':0,'c':1}) == 10`. (b) 10 layouts x 2 trials with all 6 failures in 3 layouts (2,2,2): `concentration_p < 0.05`; with failures spread one per layout over 6 layouts: `p > 0.5`. (c) `mann_whitney_p([1,2,3],[10,11,12,13]) < 0.1` and `mann_whitney_p([5,6,7],[5,6,7]) > 0.9`; identical output on two calls (seed). (d) Raises `ValueError` on an empty group.
- [ ] **Steps 2-4:** fail, implement, pass; mutation: make `concentration_p` permute within layouts and (b)'s first case must fail. **Step 5: Commit** `P2 diagnostics: layout statistics`.

### Task 4: `replay.py` (robosuite image) and Gate 0

**Files:** Create `diagnostics/replay.py`, `diagnostics/tests/test_replay.py`.

**Interfaces:**
- Consumes: `scenes.restore(layout_id) -> (model, data, names)` (names target first), `data_collection.benchmark.{synchronize_observation, apply_action, step_control}`, `src.robots.franka_panda.config.{ACTIVE_JOINTS, ARM_DOF}`.
- Produces: `replay_trial(folder: Path) -> dict` where `folder` is a trial folder holding `episode.json` and `trace.npz`. Returned arrays, T ticks, n props in `names` order: `names` (list), `hand_pos` (T,3), `hand_rot` (T,9), `left_pos`, `right_pos` (T,3), `props_xyz` (T,n,3), `props_yaw_deg` (T,n), `left_touch`, `right_touch` (T,n) bool (any substep that tick), `touched` (T,n) bool (prop in contact with any body other than world/table that tick), `command` (T,) the applied gripper command, `gate0` (bool: `state` and `props_xyz` equal the recorded arrays exactly), `first_mismatch` (int or None).

Replay loop (one tick; the physics is exactly `step_control` unrolled, as in `props_p2_data.episode.distractor_contacts`):

```python
synchronize_observation(model, data)
rec = (data.qpos[:9].copy(), np.stack([data.xpos[i] for i in prop_ids]))   # compared to trace state[t], props_xyz[t]
hand_pos, hand_rot = data.xpos[hand].copy(), data.xmat[hand].copy()          # body robot0_right_hand
action = trace['applied_action'][t]
apply_action(model, data, action, ACTIVE_JOINTS)
synchronize_observation(model, data)
for _ in range(20):                                                          # step_control asserts 20 substeps at 30 fps
    data.qfrc_applied[:ARM_DOF] = data.qfrc_bias[:ARM_DOF]
    mujoco.mj_step(model, data)
    for i in range(data.ncon):                                               # fold finger/prop and prop/other contacts into this tick's flags
        ...
synchronize_observation(model, data)
```

- [ ] **Step 1: Test (`test_replay.py`, skipped unless MuJoCo 3.4.0).** (a) Gate 0: `replay_trial` on one success and one failure of `outputs/xvla_props_p2_eval/report_p2_step_101520` returns `gate0 is True`. (b) Positive control on the success: `left_touch` and `right_touch` of the target (column 0) are both True on at least 30 ticks, and `touched` is False for every distractor column on every tick (the 540 sealed episodes touch no distractor). (c) Negative control: corrupt a copy of `trace.npz` (`applied_action[100, 0] += 0.01`), save it to a temp folder with the same `episode.json`, and `gate0` is False with `first_mismatch` greater than 100. (d) Shapes: `hand_pos` is `(T, 3)` with `T == len(trace['state'])`.
- [ ] **Step 2:** run in the robosuite image, confirm FAIL (no module). **Step 3:** implement `replay_trial` (build the prop ids from `names`; finger ids `left_finger`, `right_finger`; allowed partner bodies `world` and `table`; `props_yaw_deg = degrees(atan2(xmat[1,0], xmat[0,0]))` per prop). **Step 4:** run, PASS; mutation: apply the action before the `rec` capture and (a) must fail. **Step 5: Commit** `P2 diagnostics: trial replay and Gate 0`.

### Task 5: `reference.py` (expert envelope, robosuite image)

**Files:** Create `diagnostics/reference.py`, `diagnostics/tests/test_reference.py`.

**Interfaces:**
- Consumes: sealed episodes `outputs/props_p2_data/props_p2_20261013/episodes/episode_NNN.npz` (keys `qpos` (452,30), `ctrl` (452,8), `object`, `distractors`, `blue_slot`); `scene.build_model(objects, blue_slot) -> (env, model)`; `grasp.*`.
- Produces: `expert_grasp_sample(npz_path) -> np.ndarray(4)` (hand-frame x, y, z of the target and the yaw error at the expert's grasp closure, by `data.qpos[:] = qpos[t]; mj_forward`; the grasp closure is the first closure of `ctrl[:, 7]`); `build_reference(episodes_dir) -> dict[object, dict]` with `lo`, `hi`, `n` per object, written to `outputs/xvla_props_p2_eval/diagnostics/reference.json` by `run.py reference`.

- [ ] **Step 1: Test.** (a) For `episode_000.npz` the sample is finite, the target z in the hand frame is between -0.15 and 0.15 m (the object is between the fingers), and the sample is identical on two calls. (b) `build_reference` on the first 12 episodes returns `lo <= hi` per axis for every object that appears and `n` equal to that object's episode count. (c) At the grasp closure the hand-frame distance from the target centre to the finger midpoint is below 0.08 m on 12/12 episodes (the expert is gripping the target; a closure tick that fails this means the closure definition is wrong).
- [ ] **Steps 2-4:** fail, implement, pass. **Step 5: Commit** `P2 diagnostics: expert grasp envelope`.

### Task 6: `run.py` and the analysis (robosuite image)

**Files:** Create `diagnostics/run.py`, `diagnostics/tests/test_run.py`.

**Interfaces:**
- Consumes: Tasks 1-5; evaluation `result.json` files under `outputs/xvla_props_p2_eval/<mode>_<policy>_step_<N>/`.
- Produces CLI (`python -m experiments.xvla_props_p2_eval.diagnostics.run <cmd>`):
  - `gate0 --shard K --shards N`: replays every trial of the runs listed in `RUNS` and writes `outputs/xvla_props_p2_eval/diagnostics/measures/<run>/trial_NNN_<layout>.json` (the compact per-trial measures, not the per-tick arrays): `gate0`, `lift_tick`, `first_closure`, `grasp_closure`, `closed`, `target_in_env`, `distractor_in_env`, `held`, `lifted`, `closest_approach_m`, `target_in_hand_at_closure`, `yaw_error_deg`, `class`. `RUNS = [('screen','p2',101520), ('report','p2',101520), ('control','p2',101520), ('confirm','p2',101520), ('report','p2',126900), ('report','p1',126900)]`.
  - `reference`: writes `reference.json`.
  - `analyze`: Gate 0 (all measured trials true) and Gate 1 (fraction of 101,520 successes with grasp closure inside the envelope >= 0.80; print and exit non-zero otherwise), then the tables of the spec: taxonomy by run, never-lifted detail, layout concentration (`stats.concentration_p`), the wrong-object layout test (`stats.mann_whitney_p` on each layout's minimum target-distractor distance), the milk table. Writes `diagnostics/analysis.json` and prints Markdown tables.

- [ ] **Step 1: Test (`test_run.py`).** On three fixture trials in a temporary evaluation root (copy one success, one `never_lifted`, one `wrong_object` trial folder from the real outputs): `gate0` writes three measure files, each with `gate0 == True`; the success has `class == 'success'`; the never-lifted trial's `class` is one of the five never-lifted classes; the wrong-object trial's `class == 'wrong_object'`; `analyze` on a reference built from 12 episodes exits through the Gate 1 check with a printed fraction (not a crash).
- [ ] **Steps 2-4:** fail, implement, pass. **Step 5: Commit** `P2 diagnostics: run and analysis`.
- [ ] **Step 6: Run for real.** Reference: `run reference`. Gate 0 over all 420 trials, sharded over 6 containers pinned to 5 cores each (`--cpuset-cpus`, `OMP_NUM_THREADS=5`); every container must print `GATE0_SHARD_DONE`. Then `analyze`. **If Gate 0 fails on any trial, stop: find out why before reading any result.** **If Gate 1 is below 0.80, stop and ask the user.**

### Task 7: Video check

**Files:** Create `diagnostics/contact_sheet.py`.

**Interfaces:** `sheet(folder: Path, ticks: list[int], out: Path)` decodes `cameras.mp4` (PyAV `av`, both 512x512 views side by side per frame) and writes one PNG strip with the requested ticks.

- [ ] **Step 1:** implement; run on one trial and open the PNG (`view_image`) to confirm frames and ticks line up (the frame count equals `ticks + 1`).
- [ ] **Step 2:** for each class that has failures, build strips at (first closure - 15, first closure, closure + 30, last tick) for the two oddest trials (largest and smallest `closest_approach_m` within the class), view them, and write down any trial whose class the video contradicts. A contradiction fixes the *definition* in the spec (a dated edit with the reason), not the table.
- [ ] **Step 3: Commit** `P2 diagnostics: contact sheets`.

### Task 8: Report and documentation pass

**Files:** Create `docs/PROPS_P2_DIAGNOSTICS_<date>.md`; modify `docs/RESULTS.md`, `docs/BENCHMARK_STATUS.md`, `docs/INDEX.md`, `TODO.md`; `docs/FAILURES.md` if a mistake recurs.

- [ ] **Step 1:** Write the report: headline, Gate 0 and Gate 1 numbers, taxonomy tables per run, the never-lifted detail, the wrong-object test and its exploratory label, the milk table, checks that did not hold up, limitations (milk: 34 trials, cell and tier confounded; 7 wrong-object events), and directions labelled as hypotheses (the "what would change the next step" table of the spec).
- [ ] **Step 2:** Grep the live docs for the P2 claims this changes (`grep -rn "never_lifted\|wrong-object\|milk→blue" README.md docs/*.md TODO.md`) and update every hit, including the TODO "Open" line.
- [ ] **Step 3:** Run both test suites (admission and robosuite) and read the output; commit `P2 diagnostics report and docs`. Do not push.

## Self-Review (done against the spec)

- Spec coverage: questions 1-3 are Task 6 analyses; Gate 0 is Task 4 and Task 6 step 6; Gate 1 is Task 6; definitions are Tasks 1 and 5; the taxonomy is Task 2; counting tests is Global Constraints and Tasks 3 and 6; video check is Task 7; outputs are Task 8; contrasts (126,900, P1) are in `RUNS`.
- Interfaces: `closed_runs`, `first_closure`, `grasp_closure`, `to_hand_frame`, `yaw_error_deg`, `envelope`, `inside`, `held_ticks` (Task 1) are the only names Tasks 5 and 6 use; `classify` and `CLASSES` (Task 2) and `concentration_p`, `mann_whitney_p` (Task 3) are used unchanged in Task 6; `replay_trial`'s return keys (Task 4) are the ones Task 6 reads.
- One spec detail made explicit here: the "fingertip midpoint" is the mean of the `left_finger` and `right_finger` body positions.
