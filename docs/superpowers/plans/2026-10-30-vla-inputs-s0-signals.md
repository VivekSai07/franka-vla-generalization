# S0 — extra-signal sidecar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** a sealed per-frame sidecar (wrist wrench, finger forces, target and prop boxes) for all 540 sealed P2 episodes,
recorded by exact action replay, plus the `signals.py` module the evaluator reuses.

**Architecture:** pure signal functions over `(model, data)` computed on a scratch `MjData` copy; an action-replay loop that
checks every frame against the sealed trace; a sharded builder, a contract with `provenance()`, and a seal.

**Tech stack:** robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (MuJoCo 3.4.0),
numpy, MuJoCo `MjSpec`, `mujoco.Renderer` segmentation (OSMesa).

**Spec:** [../specs/2026-10-30-vla-inputs-s0-signals-design.md](../specs/2026-10-30-vla-inputs-s0-signals-design.md).

## Global constraints

- Branch `feat/vla-inputs` in worktree `.worktrees/feat-vla-inputs` (create once, from `main`, per
  [branching-for-an-experiment-axis](../../../.github/skills/branching-for-an-experiment-axis/SKILL.md)); `docker ps` first.
- Never edit any file in P2's `provenance()` (142 files, not 76; list in `outputs/props_p2_data/props_p2_20261013/provenance.json`).
- Mount the sealed P2 root **read-only**. Never write under `datasets/` or the P2 root.
- Sealed P2 seal fingerprint: `b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`.
- No tolerance in G1: byte equality only.
- Every new test is mutation-checked: break the code it guards, see it fail, restore.
- Test command (robosuite image, from the worktree root):

```bash
docker run --rm --runtime runc --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,size=8g -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa \
  -e PYTHONPATH=/vla -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -w /vla \
  --entrypoint python sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e \
  -m unittest discover -s experiments/vla_inputs_signals/tests -t experiments/vla_inputs_signals/tests
```

(`outputs/` is gitignored and absent in a fresh worktree: symlink or bind-mount the main checkout's `outputs/props_p2_data`.)

---

### Task 1: action replay with exact checks

**Files:** create `experiments/vla_inputs_signals/__init__.py`, `replay.py`, `tests/test_replay.py`.

**Interfaces:**
- Produces: `replay.load_episode(index: int) -> dict` (plan episode, sealed npz arrays, sealed `action` and `observation.state`
  rows from the dataset parquet, float32); `replay.frames(index: int, on_frame: Callable[[int, MjModel, MjData], None]) -> dict`
  that builds the model with `props_p2_data.episode.expert_for(names_of(e), e["blue_slot"]).model`, `restore_state` from the
  sealed `initial_state`, and per tick: `synchronize_observation` → assert `data.qpos` byte-equal to sealed `qpos[t]` and
  `props_pose` equal → `on_frame(t, model, data)` → `apply_action(model, data, action[t], ACTIVE_JOINTS)` →
  `step_control(model, data, 1/30, ARM_DOF)`; at the end asserts `qpos == final_qpos`; returns `{"frames": 452, "index": i}`.
  Raises `ReplayMismatch(index, tick, field)` on the first difference.

- [ ] **Step 1: failing tests.** `test_three_episodes_replay_exactly` (indices 0, 271, 539: no exception, 452 frames, sealed
  `observation.state` equals `qpos[:, :9].astype(float32)`); `test_corrupted_action_is_caught` (change a non-saturated command enough to alter dynamics, expect replay rejection;
  do not assume a one-ULP change must survive actuator clipping and change qpos at precisely tick 101); `test_on_frame_sees_the_observation_instant`
  (the callback's `data.qpos` equals sealed `qpos[t]` for every t).
- [ ] **Step 2:** run the test command; expect import failure.
- [ ] **Step 3:** implement `replay.py` (the probe of 2026-10-30 is the reference loop; it matched 452/452 frames on all three).
- [ ] **Step 4:** run; expect pass. Mutation: drop the `final_qpos` check → no test fails? Then add `test_final_state_checked`
  (truncate actions by one, expect `ReplayMismatch(field="final_qpos")`).
- [ ] **Step 5:** commit `S0: exact action replay`.

### Task 2: wrench and finger forces

**Files:** create `signals.py`, `tests/test_forces.py`.

**Interfaces:**
- Produces: `signals.wrist_wrench(model, data) -> np.ndarray (6,)` (force N, torque N·m, hand frame `robot0_right_hand`, MuJoCo
  force/torque-sensor convention); `signals.finger_force(model, data) -> np.ndarray (2,)` (left, right; sum of normal
  components of `mj_contactForce` over contacts between the finger body and any geom whose body is not part of the robot);
  `signals.scratch(model, data) -> MjData` (a `copy.copy(data)` copy: the Python bindings have no `mj_copyData`). Neither function writes to `data`.

- [ ] **Step 1: failing tests.** `test_wrench_matches_mujoco_sensor` (G3): build a sensor model from the same XML with `MjSpec`,
  add a site at the hand body origin and `force` + `torque` sensors on it; for every 10th frame of episodes 0, 271, 539 (via
  `replay.frames`), copy the full integration state (`mjSTATE_INTEGRATION`, includes warm start) into the sensor model's data,
  `mj_forward` from the same pre-forward integration state/warm start used for the scratch reference path, compare `sensordata` with `wrist_wrench` to 1e-9 absolute. Controlled static known-load fixtures test sensor sign, torque reference-point translation and gravity contribution;
  controlled no-contact and known-contact fixtures test each finger and robot/external filtering. Real trajectory forces
  and moving-pose load differences are descriptive, not universal pass/fail laws. An elevated object may have been released.
  The sealed NPZ has **no** `phase_index` (verified 2026-10-30: keys are `qpos, ctrl, props_pose, initial_state, final_qpos,
  object, bin, blue_slot, start, distractors, distractor_poses`; `Committer` never saves it). Use the documented
  pre-close / closed-and-lifted / after-release inference from actions and height; do not search for recorded phases.
  `test_signals_do_not_write_data`: `qpos`, `qvel`,
  `qacc_warmstart`, `efc_force` unchanged after calling both.
- [ ] **Step 2:** run; expect failure.
- [ ] **Step 3:** implement (on a scratch copy: `mj_rnePostConstraint`, then the force sensor's transform of `cfrc_int` to the
  site frame; `mj_contactForce` for each contact).
- [ ] **Step 4:** run; pass. Mutations that must fail a test: world frame instead of hand frame; sign flip; including
  robot-robot contacts; computing on `data` instead of the scratch copy.
- [ ] **Step 5:** commit `S0: wrench and finger forces`.

### Task 3: projected boxes

**Files:** modify `signals.py`; create `tests/test_boxes.py`.

**Interfaces:**
- Produces: `signals.box(model, data, body_name: str, camera: str) -> np.ndarray (5,)` = `(u_min, v_min, u_max, v_max, in_frame)`,
  coordinates / 512 clipped to [0, 1], all 0 when not in_frame; `signals.target_box(model, data, target) -> (2, 5)` (front_cam,
  gripper_cam); `signals.props_box(model, data, names) -> (len(names), 2, 5)`; `CAMERAS = ("front_cam", "gripper_cam")`.

- [ ] **Step 1: failing tests.** `test_axis_point_projects_to_centre` (a vertex on the camera's optical axis → (0.5, 0.5));
  `test_segmentation_inside_box` (G4: 60 frames, 6 per episode from 10 episodes chosen by `range(0, 540, 54)`, both cameras:
  render segmentation with `mujoco.Renderer(model, 512, 512)` + `enable_segmentation_rendering()`, take the pixels whose geom
  belongs to `prop_<target>`, assert all inside the box expanded by 1 px; record IoU in the test log). Mutations that must
  fail: flip v; swap u/v; fovy of the other camera; using only the collision geom when the visual geoms are larger.
- [ ] Add analytic tight-box fixtures, near-plane edge intersections, behind-camera/out-of-frame/full-occlusion cases,
  compiled mesh transforms and geom-group checks. A constant full-image-box mutation must fail; containment alone is not
  correctness. `in_frame` means projected geometric validity, not visible pixels. Clip geometry before perspective division.
- [ ] **Step 2–4:** fail, implement, pass, mutate.
- [ ] **Step 5:** commit `S0: projected boxes`.

### Task 4: builder, contract, seal

**Files:** create `sidecar.py`, `contract.py`, `README.md`, `tests/test_sidecar.py`, `tests/test_fork_hygiene.py`.

**Interfaces:**
- Produces: `sidecar.build_shard(out, k, shards)` (episodes `k::shards`, each via `replay.frames` with `on_frame` computing
  `wrist_wrench`, `finger_force`, `target_box`, `props_box`; writes `episodes/episode_NNN.npz` with those arrays (452, …),
  float64 with explicit episode/frame identity; no overwrite); `sidecar.merge(out)` (all 540 present, `index.json` sha256 per file); `sidecar.report(out)` (per object
  × segment statistics, segments as in Task 2, training-row 0.1/99.9 percentiles of the 8 force values, written to `report.json`);
  `sidecar.seal(out)` / `sidecar.verify(out)`; `contract.SOURCE_FILES`, `contract.provenance()`,
  `contract.P2_SEAL = "b1d642b3…f4d898"` (full value), `contract.SIDECAR_ROOT`.
- [ ] **Step 1: failing tests:** no-overwrite; merge refuses a missing episode; seal changes when one byte of one sidecar file
  changes; `verify` in a fresh process recomputes the fingerprint; `test_non_perturbation` (G2: episode 0 replayed with and
  without the signal callback, identical full integration state and subsequent `qpos` every frame); fork hygiene (symtable dead-name scan; no import of any
  `props_p2_data` module except read-only use in `replay.py`).
- [ ] **Step 2–4:** fail, implement, pass, mutate.
- [ ] Seal report/statistics, frame identities, model/runtime and full source closure as well as arrays; reject tampering
  and missing/duplicate/shifted rows. Exact qpos validates kinematics, not independently observed historical wrench.
- [ ] **Step 5:** README written (what, gates, how to run) **before** the run; it is in `SOURCE_FILES`. Confirm its size on disk is
  not zero (FAILURES #56). Commit `S0: sidecar builder, contract, seal`.

### Task 5: run, seal, report

- [ ] `docker ps`, `df -h .`, `git log --oneline -1` in the worktree.
- [ ] Build: 6 shards, pinned (`--cpuset-cpus`, `OMP_NUM_THREADS`, `LP_NUM_THREADS`), output
  `outputs/vla_inputs_signals/p2_signals_<date>/`. Poll `docker ps`; do not trust `docker wait`.
- [ ] G1 is enforced inside every replay; a `ReplayMismatch` anywhere stops S0 (record the episode and tick; no sidecar).
- [ ] Merge, report, seal; `verify` in a second container.
- [ ] Write `docs/VLA_INPUTS_S0_<date>.md` (gates, counts, the descriptive report, the M2 clipping percentiles), update
  `docs/BENCHMARK_STATUS.md`, `TODO.md`, `docs/INDEX.md`; commit `Docs: S0 sidecar sealed`.
