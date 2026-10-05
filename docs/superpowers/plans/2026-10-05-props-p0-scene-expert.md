# Props P0 — two-bin scene and expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A robosuite scene with two distinct bins and six real-life objects, and a scripted expert that puts any object into either bin from a random start pose, passing the P0 gate — then a go/no-go report and a handoff prompt for the reviewer.

**Architecture:** New package `experiments/props_scene/`. robosuite composes the scene (our arena subclass with two static bins, robosuite objects, Panda on the minimal mount); one function rewrites the compiled XML to the repo's robot contract; plain MuJoCo steps it. A new expert plans per-object grasp yaw and height from the object's geometry. A benchmark runner executes the 300-attempt gate. No policy, no LeRobot dataset in P0.

**Tech Stack:** Python 3.10, robosuite 1.5.2 / MuJoCo 3.4.0 image, pinocchio IK, numpy. CPU containers only.

**Spec:** [../specs/2026-10-05-props-sorting-design.md](../specs/2026-10-05-props-sorting-design.md) — read "What the user wants", "Carried over from B3" and the whole P0 section.

**Branch / worktree:** `feat/props-p0` from `main`, worktree `.worktrees/feat-props-p0`. Load the skills `working-on-the-robosuite-path`, `branching-for-an-experiment-axis` and `launching-a-run` before Task 1.

## Global Constraints

- Never edit: `experiments/robosuite_scene/*.py`, `experiments/b3_robosuite_data/**`, `src/**`, `data_collection/**`, `validation/**`, `experiments/paired_demonstrations/**`, `experiments/pose_snapshot_v2/**`, any sealed package, the B3 spec or plans. Import them read-only.
- Images: robosuite `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` for anything that builds the scene; admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` for tests that import no simulator. Never prune or retag images; only this project's containers.
- Never `env.step()`; one `mujoco.Renderer` at a time; never `env.sim.render` after the model is built.
- Values from the spec, verbatim: bins inner 0.22 × 0.22 m, walls 0.07 m high, 0.01 m thick; slots centred (0.32, −0.33) and (0.32, +0.33); `blue` rgba `0.15 0.35 0.85 1`, `orange` rgba `0.95 0.55 0.10 1`; objects and masses can 0.15, bread 0.05, lemon 0.08, milk 0.20, bottle 0.20, cereal 0.15 kg; start region x ∈ [0.22, 0.40], y ∈ [−0.12, 0.12], yaw ∈ [−90°, 90°); grasp point `min(0.03, height/2)` below the top; hand offset 0.105 m; lift clearance 0.05 m over the wall top; release with the object's bottom 0.02 m above the wall top; 30 Hz, 20 substeps; success = object centre inside the target bin's inner box and below its wall top, gripper open (ctrl > 127) and no finger contact, for 30 consecutive ticks within 900 ticks.
- State 9 floats (`qpos[:9]`), action 8 floats (7 joint targets + gripper 0/255), cameras `front_cam` / `gripper_cam` 512 × 512 — the schema every dataset here uses, so P1 can record with it.
- Seeds: `sha256("props_p0_v1:20261005:<identity>")[:4]` big-endian, as in B3.
- The success definition and the gate thresholds are never changed to get a pass.
- Commits: user identity, no co-author or AI trailers, short messages; never commit `outputs/`, images or videos. Do not push.
- Outputs: `outputs/props_p0/` in the main checkout, mounted at `/out`.

## Review Focus

1. **The object is in the bin's footprint but outside it** (on a wall, or balanced on the rim): `test_success` puts an object on the wall top and asserts not-success.
2. **Wrong bin counted as success:** `test_success` puts the object fully inside the other bin and asserts not-success.
3. **Yaw handled modulo the wrong period:** `test_expert` asserts the planned grasp yaw for cereal at start yaws −89°, 0°, 45° and 89° puts the finger axis within 5° of the box's narrow axis in all four (a symmetric object has period 180°, a round one any).
4. **A mass not applied:** `test_scene` asserts each object's subtree mass equals the spec value within 1%.
5. **One bin slot quietly worse:** the gate report always prints success per slot and per object, and the gate fails on either.

## File map

| file | responsibility |
|---|---|
| `experiments/props_scene/__init__.py` | empty |
| `constants.py` | every spec value above, pure |
| `arena.py` | `BinsTableArena`: `CopiedCameraTableArena` (imported) plus two static bins at given slots |
| `xml_contract.py` | `apply_props_contract(xml)`: robot rewrite reproduced from `robosuite_scene.xml_contract`, object renames (`<object>_main` → `prop_<object>`), masses, render groups |
| `scene.py` | `build_model(objects, blue_slot)` cached; `reset(model, poses, home_jitter)`; `object_pose(model, data, name)` |
| `geometry.py` | `extents(model, name)` from collision geoms in the body frame; `narrow_axis_yaw(model, data, name)` |
| `expert.py` | `PropsExpert(objects, blue_slot)`; `run(initial_state, episode, sink) -> dict` |
| `metrics.py` | `in_bin(model, data, name, bin_colour) -> dict`; `success_streak` |
| `validate.py` | `replay(result, actions, traces) -> dict`; trace hashing |
| `benchmark.py` | the 300-attempt gate: plan, run (resumable), summarise; CLI |
| `README.md` | what, how, gate, costs |
| `tests/` | `test_constants`, `test_success` (pure, admission image); `test_scene`, `test_geometry`, `test_expert`, `test_replay` (robosuite image) |
| `docs/PROPS_P0_<YYYYMMDD>.md` | go/no-go report (Task 8) |

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
mkdir -p "$MAIN/outputs/props_p0"
run() { local image=$1; shift; docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf \
  -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla -e NUMBA_CACHE_DIR=/tmp/numba \
  -v "$PWD:/vla:ro" -v "$MAIN/outputs/props_p0:/out:rw" -w /vla --entrypoint python "$image" "$@"; }
# tests: run "$NEW" -m unittest experiments.props_scene.tests.<module> -v
```

---

### Task 1: constants and scene

**Files:** `__init__.py`, `constants.py`, `arena.py`, `xml_contract.py`, `scene.py`, `tests/test_constants.py`, `tests/test_scene.py`.

**Interfaces:** `build_model(objects: tuple[str, ...], blue_slot: str) -> (env, model)` with `blue_slot ∈ {"left", "right"}` (left = y > 0 slot); cached per arguments. `reset(model, poses: dict[name, (x, y, yaw_deg)], home_jitter_deg: list[7]) -> {"qpos", "initial_state"}` (objects at rest on the table, arm at `Q_HOME` + jitter, 0.2 s settle, no RNG). Bodies: `prop_<object>`, bins `bin_blue`, `bin_orange`.

- [ ] **Step 1: Tests first.** `test_constants` pins every spec value. `test_scene` (robosuite image): for `objects=("cereal", "can")`, `blue_slot="left"`: model contains exactly those two props and both bins; bin wall geoms have the spec sizes and colours and the blue bin's centre is at (0.32, +0.33); each prop's subtree mass equals the spec within 1% (Review Focus 4); the robot part matches the current MuJoCo scene exactly as `experiments/robosuite_scene/tests/test_model_contract.py` checks it (read that test; reproduce its robot assertions); `reset` twice gives byte-equal state, props rest on the table (bottom within 2 mm of z = 0) at the requested xy within 2 mm and yaw within 1°; and a render of both cameras after reset is non-blank and saved to `/out/scene_check/` — look at it: both bins and the object must be visible in `front_cam`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. Masses: scale each prop's collision-geom densities (or set the body inertial) so the subtree mass matches. **Step 4:** run → PASS; view the PNGs and describe them in the task report. **Step 5:** commit `git commit -m "Add the props two-bin scene"`.

### Task 2: geometry

**Files:** `geometry.py`, `tests/test_geometry.py`.

**Interfaces:** `extents(model, name) -> (lx, ly, lz)` half-extents of the collision geoms in the prop's body frame; `narrow_axis_yaw(model, data, name) -> float` (radians, world yaw of the body's narrower horizontal axis, in [−π/2, π/2)); `top_z(model, data, name) -> float`.

- [ ] **Step 1: Tests** (robosuite image): extents match the spec's bounding boxes within 3 mm (milk 4.0 × 4.0 × 14.4, bread 4.8 × 4.0 × 4.8, cereal 10 × 3 × 15, can 5.2 × 5.2 × 8, lemon 7 × 4 × 4.3, bottle 5.5 × 5.9 × 16 cm, axes in whichever order the body frame gives — assert sorted); narrow-axis yaw for cereal and lemon at reset yaws −89°, 0°, 45°, 89° is within 2° of the expected (period 180°); every object's narrow extent is < 0.07 m. **Steps 2–5:** FAIL → implement → PASS → commit `"Measure prop geometry for grasp planning"`.

### Task 3: success metric

**Files:** `metrics.py`, `tests/test_success.py`.

**Interfaces:** `in_bin(model, data, name, bin_colour) -> {"inside": bool, "xy_margin_m": float, "z_m": float, "gripper_open": bool, "finger_contact": bool, "success_tick": bool}`; `success_streak(stable: int, tick: dict) -> (int, bool)` with 30 ticks.

- [ ] **Step 1: Tests** with a hand-built model or synthetic positions (pure where possible, admission image): inside and released ⇒ `success_tick`; on the wall top (Review Focus 1) ⇒ not; inside the other bin (Review Focus 2) ⇒ not; gripper closed or finger touching ⇒ not; 29 ticks ⇒ not success, 30 ⇒ success. **Steps 2–5:** FAIL → implement → PASS → commit `"Define bin success"`.

### Task 4: the expert

**Files:** `expert.py`, `tests/test_expert.py`.

**Interfaces:** `PropsExpert(objects, blue_slot)`; `run(initial_state, episode, sink) -> {"success", "reason", "frames", "stable_ticks", "final_metrics", "final_qpos", "final_state", "phase_reached"}`. `episode` = `{"object", "bin", "seed", "home_pose_jitter_deg"}`. `sink(frame, trace)` per tick: `frame` = images, `observation.state`, `action`, `task` (`put the <object> in the <bin> bin` — the P1 wording); `trace` = `qpos`, `ctrl`, `sim_time`, `substeps`, `phase_index`, and every prop's pose `(x, y, z, qw, qx, qy, qz)`.

Phases (durations in seconds, tune only within the Task 7 list): approach above grasp 2.0 → descend 1.5 → close 1.0 → lift to clearance 1.5 → move over bin 2.5 → lower to release height 1.5 → open 1.0 → retract 1.5 → home 2.5. Hand orientation = home rotation composed with a rotation about world z equal to `narrow_axis_yaw` (so fingers close across the narrow axis). A grasp-verify check after lift: if the object did not rise, end with `reason="grasp"`.

- [ ] **Step 1: Tests** (robosuite image): Review Focus 3 on planned yaw; one episode per object into the blue bin from a central start, all succeed (about 6 × 1 min). **Steps 2–5:** FAIL → implement → PASS → commit `"Script per-object grasps into either bin"`.

### Task 5: replay and determinism

**Files:** `validate.py`, `tests/test_replay.py`.

- [ ] Replaying an episode's recorded actions from its initial state reproduces `qpos[:9]` within 1e-3 and the prop pose within 1e-3; two separate processes running the same episode produce byte-equal trace files (hash compare). Commit `"Prove props episodes replay and repeat"`.

### Task 6: the gate runner

**Files:** `benchmark.py`, `README.md`.

**Interfaces:** `python -m experiments.props_scene.benchmark --out /out/gate_<date> [--shard K/N] [--resume]`. The plan: 6 objects × 2 target bins × 2 `blue_slot` values × 12 seeds = 288 attempts, plus 12 extra in the +y near-base corner (start x ∈ [0.22, 0.26], y ∈ [0.08, 0.12]) = 300. Start pose and yaw from the seed. One JSON line per attempt (object, bin, slot of the target bin, start pose, yaw, success, reason, phase reached, final metrics, wall-clock); finished attempts are skipped on `--resume`; `--shard` splits deterministically so 6 containers can run in parallel. `--summary` prints overall, per object, per target slot, per reason, and the gate verdict.

- [ ] Tests for plan size/balance and resume (admission image, fake runner). Commit `"Run the P0 gate"`.

### Task 7: run the gate (CPU, about 300 × 1 min ≈ 5 h serial, ≈ 1 h on 6 shards)

- [ ] Launch 6 detached shards (`docker run -d --name props-p0-gate-<k> ...`), poll, `--resume` a shard once if it dies.
- [ ] If the gate fails, diagnose before changing anything (look at traces and a few videos/frames of failures — FAILURES #40: confirm the physical state first). Allowed changes, one at a time, each recorded with before/after numbers: grasp depth (0.02–0.05 m), lift clearance (0.03–0.08 m), release height (0.01–0.04 m), phase durations (±50%), per-object friction (0.8–1.5). Re-run the full gate after a change. Not allowed: the success definition, the gate thresholds, the object set or masses.
- [ ] If still red after the list: run the failing objects with robosuite's OSC controller as a comparison (a separate script, not replacing the expert) and record the result. Stop there.

### Task 8: report and handoff

- [ ] **Report** `docs/PROPS_P0_<YYYYMMDD>.md`: the gate table (overall, per object, per slot, per reason) against the thresholds; every tuning attempt with numbers; a contact sheet description (what both cameras show); what the expert does per object (grasp yaw, height); anything surprising; last line exactly `Decision: go` or `Decision: no-go`.
- [ ] **Documentation pass:** `TODO.md` (P0 result, next P1), `docs/BENCHMARK_STATUS.md`, `docs/OPERATIONS.md` (new package, outputs), `docs/INDEX.md`, `docs/FAILURES.md` for anything recurrable, `CLAUDE.md` and `.github/instructions/experiments.instructions.md` (the new package, which image, not hash-bound yet), the `working-on-the-robosuite-path` skill if a new trap was hit.
- [ ] **Commit** `git commit -m "Report props P0"`. Do not merge; do not push.
- [ ] **Handoff file** `outputs/props_p0/HANDOFF.md` (not committed) and the same text as your final message to the user, in this shape:

```
## What was done
<tasks completed, commits (short SHA + subject), branch, nothing pushed>
## Gate result
<the table: overall / per object / per slot, thresholds, Decision: go|no-go>
## What changed from the plan, and why
<every deviation and ruling, with what it costs if wrong>
## Tuning history
<each change, before/after numbers>
## Problems hit
<crashes, environment fixes, anything recorded in FAILURES>
## Open questions for the reviewer
<numbered, each answerable yes/no or with a choice>
## Where to look
<report path, outputs paths, contact sheets, logs, containers kept>

## Handoff prompt for Claude
<a self-contained prompt the user can paste to Claude: what to verify (with exact paths and
commands), what is decided, what is open, and the next step it is asked to take — designing P1>
```

## Self-review

- Spec P0 coverage: scene (Task 1), object geometry and per-object grasp yaw/height (Tasks 2, 4), success (Task 3), determinism and replay (Task 5), the 300-attempt gate with per-object, per-slot and corner checks (Tasks 6–7), bounded tuning and the OSC fallback (Task 7), report (Task 8). P1–P3 are not in this plan.
- The B3 carry-overs are built in: object poses in every trace, corner stratum, per-slot asymmetry check, look-first diagnosis.
- Not known until it runs: whether the tall objects (milk, bottle, cereal) survive the drop into a 7 cm-deep bin upright or not — success does not require upright, only inside.
