# robosuite scene bring-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision 2 (2026-10-02).** Revision 1 pinned robosuite 1.5.2 on MuJoCo 3.13.0 and blocked at Task 2: that pair cannot reset any robosuite environment. This revision re-pins the robosuite image to MuJoCo 3.4.0, adds the model differences revision 1 did not know about, and replaces "any red risk is no-go, no tuning" with a bounded tuning protocol. See *What changed* below.

**Goal:** Prove a robosuite-composed Franka scene can be driven by the existing scripted expert and LeRobot schema, seal a 5-episode PoC dataset, and write a go/no-go report. Phase 1 only. It does not collect B3.

**Architecture:** A new package `experiments/robosuite_scene/` asks robosuite 1.5.2 to compose a Panda on `RethinkMinimalMount`, a floor-height table, two cubes and the two cameras the collector already names. After one `env.reset()`, the compiled XML is rewritten by one function (`apply_scripted_contract`) so the model satisfies what the scripted expert assumes, then our code drives it with `data.ctrl` and `mujoco.mj_step`. `env.step()` is never called. The robosuite path runs in its own image on MuJoCo 3.4.0; the sealed pipeline keeps its own image on MuJoCo 3.13.0 and is not touched.

**Tech Stack:** Python 3.10, robosuite 1.5.2, MuJoCo 3.4.0 (new image only), numpy 2.2.6, LeRobot 0.4.4, pinocchio (`pin` 3.8.0), OSMesa. CPU `runc` containers only. No GPU.

**Spec:** [../specs/2026-10-01-robosuite-scene-bringup-design.md](../specs/2026-10-01-robosuite-scene-bringup-design.md)

**Branch / worktree:** `feat/robosuite-scene-bringup` at `.worktrees/feat-robosuite-scene-bringup`. Do not implement in the main checkout.

## Current state (start here)

- Task 1 of revision 1 is committed as `729b56c`: `Dockerfile`, `contract.py`, `__init__.py`, `tests/test_r1_image.py`. It pins MuJoCo 3.13.0 and image `sha256:3ecd62c5…` (tag `vla-robosuite:1.5.2`). That image is superseded; Task 1 below replaces the pin. Leave the old tag alone; deleting it is the user's call.
- Uncommitted in the worktree from the blocked Task 2: `arena.py`, `cameras.py`, `scene.py`, `mujoco_compat.py`, `tests/test_r2_render.py`. None has passed a test. Task 2 below revises them.
- The ledger under `.superpowers/sdd/2026-10-01-robosuite-scene-bringup/` describes revision 1, including a ruling "do not change either pin; add a process-local adapter". **Revision 2 supersedes that ruling** (next section). Start a fresh ledger entry.

## Why re-pin MuJoCo instead of the adapter

The previous session added `mujoco_compat.py`, which monkey-patches robosuite and `mujoco.mj_fullM` in-process so 1.5.2 can reset on MuJoCo 3.13.0. Revision 2 does not use it. The user's goal for this path is a working robosuite scene to learn and build datasets on, not physics-level comparability with the sealed results, and for that goal:

- robosuite 1.5.2 runs **unpatched** on MuJoCo 3.4.0, the release current when 1.5.2 shipped. Reset, offscreen render, `env.step` and `RethinkMinimalMount` were all verified there. The adapter was verified for reset only, and every robosuite feature Phase 2 touches could meet another 3.4 → 3.13 break.
- The image has to be rebuilt anyway: revision 1's build downgraded numpy 2.2.6 → 1.26.4 and left `pip check` reporting three broken requirements, one of them pinocchio's `cmeel-boost`.
- The cost — two MuJoCo versions in the repository — is contained by the image boundary. Sealed datasets, checkpoints and evaluations run only in the admission image, exactly as before.

If the same-engine property is ever wanted back, the adapter approach is the alternative; it is recorded in the report, not kept in the package.

## What changed from revision 1 (all measured 2026-10-02)

| finding | evidence | consequence in this plan |
|---|---|---|
| robosuite 1.5.2 cannot reset any env on MuJoCo 3.13.0 | stock `Lift` fails at `binding_utils.py:525`; `controller.py:227` uses `data.qM`, absent in 3.13.0; 1.5.2 is the latest robosuite on PyPI | new image pins `mujoco==3.4.0`. Verified there: `Lift` reset, OSMesa offscreen render, `env.step`, `RethinkMinimalMount`, XML recompile, the repo's `benchmark.py` helpers, `PairedDataset` import |
| plain `pip install robosuite` downgrades numpy and adds a second OpenCV | `mink==0.0.5` requires `numpy<2` | install robosuite with `--no-deps`; add only `numba`, `scipy`, `termcolor`. `mink`, `qpsolvers`, `opencv-python`, `pynput`, `pytest` stay out |
| robosuite's `finger_joint2` runs `[-0.04, 0]`; ours runs `[0, 0.04]` | robosuite `panda_gripper.xml` | Task 3 flips the joint axis and range so `qpos[8]` keeps the sign every existing dataset has |
| robosuite hand sits 0.5 mm further along the flange, rotated 0.028° | `right_hand pos="0 0 0.1065"` vs `hand pos="0 0 0.107"` | camera-pose gate is 1 mm / 0.1°, not 1e-6 |
| robosuite fingers are shorter: finger bodies 6.5 mm higher, lowest finger collision 11.7 mm higher | forward kinematics at `Q_HOME` on both models | grasp height becomes a named constant in a forked expert, tunable under the protocol below |
| robosuite arm joints: armature 5.0…0.71, damping 0.1/0.01, frictionloss 0.1; fingers damping 100, armature 1; integrator Euler | saved env XML | Task 3 overwrites these with the current scene's values so the fixed-duration expert tracks as it was tuned |
| env-built names differ from revision 1's guesses | saved env XML | bodies `gripper0_right_leftfinger` / `gripper0_right_rightfinger`, joints `gripper0_right_finger_joint1/2`, hand `robot0_right_hand`, cube bodies `<name>_main` |

## Global Constraints

- Do not edit `src/**`, `data_collection/**`, `validation/**`, `experiments/fixed_layout/**`, `experiments/paired_demonstrations/**`, `experiments/pose_snapshot_v2/**`, or any `*_data/` package. They are imported read-only.
- The admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` is not retagged, rebuilt or overwritten. Nothing sealed is run in the robosuite image. Only the robosuite-free unit tests of this package run in the admission image.
- Robosuite image: `FROM` the admission digest; `robosuite==1.5.2` with `--no-deps`; `mujoco==3.4.0`; `numpy==2.2.6`; `numba==0.68.0`; `scipy==1.15.3`; `termcolor==3.3.0`. The contract stores the image id digest, never the tag.
- Physics clock: `configure_timing(model, 30)` — 600 Hz, 20 substeps per control tick.
- Never call `env.step()`. `env.reset()` runs once per sim handle.
- Camera names stay `front_cam` and `gripper_cam` (`experiments/paired_demonstrations/media.py` maps the LeRobot keys to them). Attributes copied from the current scene: `front_cam` `pos="1.2 0 0.6"` `xyaxes="0 1 0 -0.42 0 0.9"` `fovy="60"` on the world; `gripper_cam` `pos="0.06 0 0.03"` `xyaxes="0 1 0 1 0 0"` `fovy="75"` on the hand body. 512 × 512. `robosuite.macros.IMAGE_CONVENTION = "opencv"` before `robosuite.make`.
- Fixed layout: red `(0.45, 0.0, 0.02)`, green `(0.35, 0.2, 0.02)`, half-size `0.02`, mass `0.05`. Table top at z = 0, robot `link0` at the world origin. No randomisation, no distractors.
- LeRobot features and encoding come from `experiments.paired_demonstrations.media` (`FEATURES`, `ENCODING`, `CAMERAS`, `PairedDataset`). Do not fork those dicts.
- Five episodes, instruction from `instruction("red", "green")`, seeds `0..4`. Not published, not trained on.
- Commits: user identity only, no co-author trailer, short messages. Never commit `outputs/`, images or videos.
- The B2-checkpoint smoke test is **out of this plan**. It needs a GPU and an inference harness the sealed eval package does not expose for a two-cube robosuite scene. It becomes its own small plan after a `go`.

## Tuning protocol (replaces "any red risk is no-go")

A failing gate is first a debugging task. These knobs may be changed, each change recorded in the report with the value tried and the result:

1. `contract.GRASP_Z_OFFSET_M` — try `0.105`, then `0.0985`, then `0.095`. No other values without asking the user.
2. Contact options in `apply_scripted_contract` — `cone`/`impratio` may be set to the current scene's (`pyramidal`, `1`).
3. The values this plan tells `apply_scripted_contract` to write, if one was transcribed wrongly.

Never changed: the `stack_metrics` success window, the IK gains, any shared module, any test tolerance in this plan. If a gate is still red after the knobs above, stop and write `Decision: no-go` naming the gate.

## Review Focus

1. **A gate test that passes without touching the robosuite model.** Every gate test must call `build_sim()` or `run_episode()`; none may assert on constants alone.
2. **`qpos[8]` sign.** A model whose right finger still reads negative passes shape checks. Task 3 asserts both finger ranges are `[0, 0.04]`; Task 5 asserts both state entries are `+0.04` with the gripper open.
3. **Cube rests on the floor plane instead of the table, or falls through.** Task 5 asserts settled cube z within 1 mm of `0.02` and no cube–floor contact.
4. **Collector run in the wrong image.** Task 8's recorder refuses to start unless `mujoco.__version__ == contract.MUJOCO_VERSION`.
5. **Forked expert drifts from the original.** Task 6's AST test fails if `run` differs from `pose_snapshot_v2` by anything but the one named constant.

## File map

| file | responsibility |
|---|---|
| `experiments/robosuite_scene/Dockerfile` | robosuite image |
| `experiments/robosuite_scene/contract.py` | image digests, versions, layout, grasp height, episode count |
| `experiments/robosuite_scene/cameras.py` | copied camera attributes; no robosuite import |
| `experiments/robosuite_scene/arena.py` | `CopiedCameraTableArena` |
| `experiments/robosuite_scene/xml_contract.py` | `apply_scripted_contract`: the only place the compiled XML is rewritten |
| `experiments/robosuite_scene/scene.py` | `BringupEnv`, `build_sim`, `overwrite_reset` |
| `experiments/robosuite_scene/expert.py` | `RobosuiteExpert`, `run_episode` |
| `experiments/robosuite_scene/validate.py` | `frame_from_sim`, replay checks, `fingerprint` |
| `experiments/robosuite_scene/record.py` | trace modes and the 5-episode collector |
| `experiments/robosuite_scene/compare_cameras.py` | side-by-side PNGs for the report |
| `experiments/robosuite_scene/README.md` | what the package is, how to run it (Task 8) |
| `docs/ROBOSUITE_SCENE_BRINGUP_<YYYYMMDD>.md` | go/no-go report (Task 8) |

Shared names later tasks must not rename:

```python
# contract.py
OLD_IMAGE = "sha256:08de1e09…"; IMAGE = "<new id>"
ROBOSUITE_VERSION = "1.5.2"; MUJOCO_VERSION = "3.4.0"
FIXED_CUBES = {"target_cube_red": (0.45, 0.0, 0.02), "target_cube_green": (0.35, 0.2, 0.02)}
GRASP_Z_OFFSET_M = 0.105
EPISODE_COUNT = 5
# cameras.py
FRONT_CAM, GRIPPER_CAM            # dicts of MJCF attributes, as already written
# xml_contract.py
def apply_scripted_contract(xml: str) -> str: ...
# scene.py
def build_sim() -> tuple: ...                      # (env, mujoco.MjModel, mujoco.MjData)
def overwrite_reset(model, data) -> dict: ...      # keys "qpos", "initial_state"
# expert.py
def run_episode(seed: int, sink) -> dict: ...
# validate.py
def frame_from_sim(model, data, action, task) -> dict: ...
def replay_recorded(result, frames, traces) -> dict: ...
def replay_reference(result, frames, traces) -> dict: ...
def kinematic_mismatch(result, traces) -> float: ...
def fingerprint(root) -> str: ...
```

## Running things

Set once per shell, from the worktree root:

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=<contract.IMAGE after Task 1>
mkdir -p "$MAIN/outputs/robosuite_scene"
run() {  # run <image> <python args...>
  local image=$1; shift
  docker run --rm --runtime runc --network none --read-only \
    --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=8g \
    -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla \
    -e NUMBA_CACHE_DIR=/tmp/numba -e GIT_OPTIONAL_LOCKS=0 \
    -v "$PWD:/vla:ro" -v "$MAIN/.git:$MAIN/.git:ro" \
    -v "$MAIN/outputs/robosuite_scene:/out:rw" \
    -w /vla --entrypoint python "$image" "$@"
}
```

Outputs are written under `/out` (the main checkout's gitignored `outputs/robosuite_scene/`), never under `/vla`. Tests run as `run "$NEW" -m unittest experiments.robosuite_scene.tests.<module> -v`. Poll long commands; do not rely on `docker wait`.

---

### Task 1: R1 — re-pin the robosuite image to MuJoCo 3.4.0

**Files:**
- Modify: `experiments/robosuite_scene/Dockerfile`
- Modify: `experiments/robosuite_scene/contract.py`
- Modify: `experiments/robosuite_scene/tests/test_r1_image.py`
- Modify: `docs/superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md` (append a *Corrections* section)
- Modify: `docs/FAILURES.md` (append one numbered entry)

**Interfaces:**
- Produces: every name in the `contract.py` block above. `IMAGE` is the new image id.

- [ ] **Step 1: Change the test first**

In `test_contract_records_a_new_digest`, replace the `MUJOCO_VERSION` assertion with `"3.4.0"` and add:

```python
        self.assertNotEqual(IMAGE, "sha256:3ecd62c5a4f220dbf674494ae8057d9db439c8ff3911981edc29aeee87ccc2c3")
        from experiments.robosuite_scene.contract import FIXED_CUBES, GRASP_Z_OFFSET_M, EPISODE_COUNT
        self.assertEqual(FIXED_CUBES, {"target_cube_red": (0.45, 0.0, 0.02), "target_cube_green": (0.35, 0.2, 0.02)})
        self.assertEqual((GRASP_Z_OFFSET_M, EPISODE_COUNT), (0.105, 5))
```

Replace `SEALED` with: `"src"`, `"data_collection"`, `"validation"`, `"experiments/fixed_layout"`, `"experiments/paired_demonstrations"`, `"experiments/pose_snapshot_v2"`, `"experiments/b1_language_data"`, `"experiments/b2_distractor_data"`, `"experiments/broad_workspace_data"`.

- [ ] **Step 2: Run it and see it fail**

`run "$OLD" -m unittest experiments.robosuite_scene.tests.test_r1_image -v` → FAIL on `MUJOCO_VERSION`.

- [ ] **Step 3: Write the Dockerfile and build**

```dockerfile
FROM sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
# robosuite 1.5.2 does not run on MuJoCo 3.13.0 (binding_utils enum asserts, data.qM removed).
# --no-deps keeps mink (numpy<2) and opencv-python out; only what robosuite imports is added.
RUN pip install --no-cache-dir --no-deps "robosuite==1.5.2" "mujoco==3.4.0" \
 && pip install --no-cache-dir "numba==0.68.0" "scipy==1.15.3" "termcolor==3.3.0" "numpy==2.2.6"
```

```bash
DOCKER_BUILDKIT=0 docker build \
  --build-arg HTTP_PROXY="$HTTP_PROXY" --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  --build-arg http_proxy="$http_proxy" --build-arg https_proxy="$https_proxy" \
  -f experiments/robosuite_scene/Dockerfile -t vla-robosuite:1.5.2-mj3.4.0 .
docker image inspect vla-robosuite:1.5.2-mj3.4.0 --format '{{.Id}}'
docker image inspect "$OLD" --format '{{.Id}}'   # must still print the admission digest
```

The classic builder is required: BuildKit tries to pull a bare `sha256:` reference from a registry. Write the new id into `contract.IMAGE`, set `MUJOCO_VERSION = "3.4.0"`, and add `FIXED_CUBES`, `GRASP_Z_OFFSET_M`, `EPISODE_COUNT`.

- [ ] **Step 4: Run the unit test and the runtime gate**

Unit test PASS in `$OLD`. Then in `$NEW`:

```bash
run "$NEW" -c "
import importlib.metadata as m, robosuite, mujoco, lerobot, pinocchio, torch
assert (m.version('robosuite'), m.version('mujoco'), m.version('numpy'), m.version('lerobot')) == ('1.5.2', '3.4.0', '2.2.6', '0.4.4')
env = robosuite.make('Lift', robots='Panda', base_types='RethinkMinimalMount', has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, renderer='mujoco')
env.reset()
img = env.sim.render(camera_name='agentview', width=512, height=512)
assert img.shape == (512, 512, 3) and img.std() > 0
from experiments.paired_demonstrations.media import PairedDataset
print('R1_OK')"
```

Expected last line: `R1_OK`. The gate resets an environment because an import-only gate is what let revision 1's pin through. `pip check` will list `mink`, `opencv-python`, `pytest`, `qpsolvers` as missing for robosuite; that is intended and nothing else may appear.

- [ ] **Step 5: Record the finding, then commit**

Spec *Corrections* section: MuJoCo pin is 3.4.0 in the robosuite image only; camera names are `front_cam`/`gripper_cam`; the scene has a floor, not a table; R7 tolerance is 1 mm / 0.1°; the smoke test moved to a follow-up plan; the tuning protocol replaces "any single risk failing means stop". `docs/FAILURES.md`, next free number: a version pin chosen from a declared requirement (`mujoco>=3.3.0`) and gated on import only; the pair could not reset an environment; fix is a gate that performs a real reset.

```bash
git add experiments/robosuite_scene/Dockerfile experiments/robosuite_scene/contract.py \
  experiments/robosuite_scene/tests/test_r1_image.py docs/FAILURES.md \
  docs/superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md
git commit -m "Re-pin the robosuite image to MuJoCo 3.4.0"
```

---

### Task 2: R2 — the scene builds and renders headless

**Files:**
- Modify (uncommitted WIP): `experiments/robosuite_scene/arena.py`, `scene.py`, `tests/test_r2_render.py`
- Keep as is: `experiments/robosuite_scene/cameras.py`
- Delete (uncommitted WIP): `experiments/robosuite_scene/mujoco_compat.py`
- Create: `experiments/robosuite_scene/xml_contract.py`

**Interfaces:**
- Consumes: `contract.*`, `FRONT_CAM`, `GRIPPER_CAM`
- Produces: `build_sim() -> (env, model, data)`; `model`/`data` are plain `mujoco.MjModel`/`MjData` compiled from `apply_scripted_contract(env.sim.model.get_xml())`. `env.step` raises `RuntimeError`. `apply_scripted_contract` lives in `xml_contract.py` and, in this task, only upserts the two cameras.

- [ ] **Step 1: Revise the test**

Keep the WIP test and add, right after `build_sim()`:

```python
        data.qpos[:] = env.sim.data.qpos          # same joint order; compare like with like
        mujoco.mj_forward(model, data)
        link0 = data.xpos[model.body("robot0_link0").id]
        np.testing.assert_allclose(link0, [0, 0, 0], atol=1e-9)
        table = model.geom("table_collision")
        top = data.geom_xpos[table.id][2] + model.geom_size[table.id][2]
        self.assertAlmostEqual(float(top), 0.0, places=6)
        self.assertLess(float(model.geom_pos[model.geom("floor").id][2]), -0.04)
        import sys
        self.assertNotIn("experiments.robosuite_scene.mujoco_compat", sys.modules)
```

- [ ] **Step 2: Run it in `$NEW`** — expected FAIL (adapter imported, base not at origin, or floor at z = 0).

- [ ] **Step 3: Implement**

- Delete `mujoco_compat.py` and remove its import and the `install_mujoco_compat()` call from `scene.py`.
- `arena.py`: after `super().__init__`, set the floor geom's `pos` to `"0 0 -0.05"` so it is not coplanar with the table top.
- `scene.py` `_load_model`: after `super()._load_model()`, call `self.robots[0].robot_model.set_base_xpos([0, 0, 0])`. Keep attaching `gripper_cam` to `robot0_right_hand` so robosuite's own renderer can name it. Remove `_rename_cube_bodies`; cube renaming and contact parameters move to Task 3's processor.
- Move `apply_scripted_contract` to `xml_contract.py`; `scene.py` imports it. `build_sim` is otherwise as written.

- [ ] **Step 4: Run in `$NEW`** — PASS: both cameras non-blank from both renderers, unflipped error below flipped, `env.step` raises, base at origin, table top at z = 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/robosuite_scene/{cameras,arena,scene,xml_contract}.py experiments/robosuite_scene/tests/test_r2_render.py
git commit -m "Render the robosuite bring-up scene headless"
```

---

### Task 3: the scripted-expert model contract

**Files:**
- Modify: `experiments/robosuite_scene/xml_contract.py`
- Create: `experiments/robosuite_scene/tests/test_model_contract.py`

**Interfaces:**
- Consumes: `build_sim`
- Produces: a compiled model on which `GraspController`, `stack_metrics`, `apply_action`, `step_control` and `expert_v1.configuration` work unmodified.

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
import mujoco

class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.robosuite_scene.scene import build_sim
        from src.robots.franka_panda.config import SCENE_PATH
        _, cls.model, cls.data = build_sim()
        cls.current = mujoco.MjModel.from_xml_path(SCENE_PATH)

    def test_names_the_shared_code_looks_up(self):
        for body in ("left_finger", "right_finger", "target_cube_red", "target_cube_green"):
            self.model.body(body)

    def test_qpos_layout_and_finger_sign(self):
        m = self.model
        self.assertEqual([int(m.jnt_qposadr[i]) for i in range(9)], list(range(9)))
        np.testing.assert_allclose(m.jnt_range[7:9], [[0, 0.04], [0, 0.04]])
        np.testing.assert_allclose(m.jnt_range[:7], self.current.jnt_range[:7])
        d = mujoco.MjData(m)
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        gap = d.xpos[m.body("right_finger").id] - d.xpos[m.body("left_finger").id]
        self.assertAlmostEqual(float(np.linalg.norm(gap)), 0.08, places=3)   # open = 8 cm apart

    def test_actuators_match_the_current_scene(self):
        m, c = self.model, self.current
        self.assertEqual(m.nu, 8)
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_allclose(getattr(m, field), getattr(c, field), err_msg=field)
        self.assertEqual(int(m.actuator_trntype[7]), int(mujoco.mjtTrn.mjTRN_TENDON))

    def test_joint_dynamics_and_integrator_match(self):
        m, c = self.model, self.current
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(m, field)[:9], getattr(c, field)[:9], err_msg=field)
        self.assertEqual(int(m.opt.integrator), int(c.opt.integrator))
        self.assertAlmostEqual(float(m.opt.timestep), 1 / 600)
        self.assertEqual(int(m.neq), 1)                                       # finger coupling

    def test_cube_contact_parameters(self):
        m, c = self.model, self.current
        for name in ("target_cube_red", "target_cube_green"):
            self.assertAlmostEqual(float(m.body_mass[m.body(name).id]), 0.05, places=6)
            ours = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(name).id and m.geom_contype[g]]
            theirs = c.geom(f"cube_geom_{name.rsplit('_', 1)[1]}").id
            self.assertEqual(len(ours), 1)
            for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"):
                np.testing.assert_allclose(getattr(m, field)[ours[0]], getattr(c, field)[theirs], err_msg=field)
```

- [ ] **Step 2: Run in `$NEW`** — expected FAIL on the first body lookup.

- [ ] **Step 3: Extend `apply_scripted_contract`**

Parse with `xml.etree.ElementTree`. Every edit sets explicit attributes on the saved XML (robosuite writes attributes out, not default classes). Raise `ValueError` if an element an edit needs is missing.

- Rename bodies: `gripper0_right_leftfinger` → `left_finger`, `gripper0_right_rightfinger` → `right_finger`, `target_cube_red_main` → `target_cube_red`, `target_cube_green_main` → `target_cube_green`.
- `gripper0_right_finger_joint2`: `axis="0 -1 0"`, `range="0 0.04"`.
- Joints `robot0_joint1..7` and both finger joints: `armature="0.1"`, `damping="1"`, `frictionloss="0"`.
- `<option>`: add `integrator="implicitfast"`. Leave robosuite's `cone`, `impratio`, `density`, `viscosity` (tuning-protocol knob 2).
- Replace all children of `<actuator>` with eight `<general>` elements carrying the values of `src/robots/franka_panda/model/panda.xml` lines 266–279, written out explicitly because that file uses a default class: `biastype="affine"`, `ctrllimited="true"`, `forcelimited="true"`; arm `ctrlrange` equal to each joint's range; `forcerange` ±87 for joints 1–4 and ±12 for 5–7; `gainprm` / `biasprm` of 4500 / `0 -4500 -450` (joints 1–2), 3500 / `0 -3500 -350` (3–4), 2000 / `0 -2000 -200` (5–7). The eighth acts on tendon `split`: `ctrlrange="0 255"`, `forcerange="-100 100"`, `gainprm="0.07843137255 0 0"`, `biasprm="0 -500 -50"`.
- Add `<tendon><fixed name="split">` over both finger joints with `coef="0.5"` each, and `<equality><joint joint1=… joint2=… solimp="0.95 0.99 0.001" solref="0.005 1"/>`.
- On each cube's collision geom: remove `density`; set `mass="0.05"`, `size="0.02 0.02 0.02"`, `friction="1.2 0.005 0.0001"`, `solref="0.002 1"`, `solimp="0.9 0.95 0.001"`, `margin="0.001"`.
- `visual/global`: `offwidth="512"` `offheight="512"`.
- Rendering groups: the expert renders with MuJoCo's default options, which draw geom groups 0–2, and robosuite keeps its coloured collision geoms in group 0 (it hides them in its own renderer). Set every geom whose group is 0 or absent to `group="3"` and every `<site>` to `group="4"`, so only robosuite's visual geoms are drawn. Contacts are unaffected by group. Add to the test class:

```python
    def test_only_visual_geoms_are_drawn(self):
        self.assertFalse(np.any(self.model.geom_group == 0))
        self.assertFalse(np.any(self.model.site_group < 3))
```

- [ ] **Step 4: Run in `$NEW`** — `test_model_contract` PASS and `test_r2_render` still PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/robosuite_scene/xml_contract.py experiments/robosuite_scene/tests/test_model_contract.py
git commit -m "Rewrite the robosuite Panda to the scripted-expert contract"
```

---

### Task 4: R7 — camera poses match the current scene

**Files:**
- Create: `experiments/robosuite_scene/tests/test_r7_cameras.py`
- Create: `experiments/robosuite_scene/compare_cameras.py`
- Modify: `experiments/robosuite_scene/xml_contract.py` only if the test demands it

**Interfaces:**
- Consumes: `build_sim`
- Produces: `python -m experiments.robosuite_scene.compare_cameras --out DIR` writes `{front_cam,gripper_cam}_{current,robosuite}.png` and `side_by_side.png`.

- [ ] **Step 1: Write the test**

For each of two arm configurations — `Q_HOME[:7]` and `[0.4, -0.3, 0.2, -1.8, 0.3, 1.9, 0.2]` — set `qpos[:7]` on both the current scene (`SCENE_PATH`) and the robosuite model, `mj_forward`, and for both cameras assert: `fovy` equal; `cam_xpos` within `1e-3` m; rotation angle between the two `cam_xmat` under `0.1` degrees. The second configuration is what catches a wrist camera parented to the world.

- [ ] **Step 2: Run in `$NEW`.** Measured hand offset is 0.5 mm and 0.028°, so this should pass once `gripper_cam` hangs off `robot0_right_hand`. To see it fail first, temporarily parent `gripper_cam` to the world in `apply_scripted_contract`, confirm the second configuration fails, and revert. If it fails for real, the fix belongs in `apply_scripted_contract`, not in the tolerance.

- [ ] **Step 3: Write `compare_cameras.py`** — render both scenes at `Q_HOME`, save the four PNGs and one horizontal stack with Pillow.

- [ ] **Step 4: Run it** — `run "$NEW" -m experiments.robosuite_scene.compare_cameras --out /out/camera_compare`; five PNGs exist. **Open `side_by_side.png` and look at it**: framing must match; table, floor, mount and robot-mesh differences are expected and go in the report.

- [ ] **Step 5: Commit** (code and test only, no PNGs)

```bash
git add experiments/robosuite_scene/compare_cameras.py experiments/robosuite_scene/tests/test_r7_cameras.py
git commit -m "Gate camera poses against the current scene"
```

---

### Task 5: R4 and R5 — schema and deterministic reset

**Files:**
- Create: `experiments/robosuite_scene/validate.py` (`frame_from_sim`)
- Modify: `experiments/robosuite_scene/scene.py` (`overwrite_reset`)
- Create: `experiments/robosuite_scene/record.py` (`--trace-only`)
- Create: `experiments/robosuite_scene/tests/test_r4_schema.py`, `tests/test_r5_reset.py`

**Interfaces:**
- Produces: `frame_from_sim(model, data, action, task) -> dict` with keys `observation.state` (float32 `(9,)`, `qpos[:9]`), `action` (float32 `(8,)`), `task`, and one 512×512×3 uint8 image per key in `media.CAMERAS`. `overwrite_reset(model, data) -> {"qpos", "initial_state"}`. `record.py --trace-only --seed N --out DIR` writes `DIR/trace.npz` holding those two arrays.

- [ ] **Step 1: Write the failing tests**

`test_r4_schema`: build, `overwrite_reset`, call `frame_from_sim` with a zero action; assert the shapes and dtypes above and that they equal `FEATURES[...]["shape"]`; assert `frame["observation.state"][7:9]` are both within `1e-3` of `0.04`; assert a wrong-shaped action raises `ValueError`; assert `sha256(data_collection/benchmark.py) == "1c11e65b620da8d4aa3b89cba46d1fc832e718f5d6a5615104a4be05d03c10e4"`.

`test_r5_reset`: two `overwrite_reset` calls give array-equal `qpos` and `initial_state`; each cube's settled xy is within `1e-3` of `FIXED_CUBES` and z within `1e-3` of `0.02`; no contact in `data.contact[:data.ncon]` pairs a cube geom with the `floor` geom.

- [ ] **Step 2: Run both in `$NEW`** — FAIL, names missing.

- [ ] **Step 3: Implement**

`overwrite_reset`: `mj_resetData`; zero `qpos`/`qvel`; write each cube's free joint to its `FIXED_CUBES` position with quat `(1, 0, 0, 0)`; `qpos[:9] = Q_HOME`; `ctrl[:7] = Q_HOME[:7]`, `ctrl[-1] = 255`; step `round(0.2 / model.opt.timestep)` times with `mujoco.mj_step`; `synchronize_observation`; return a `qpos` copy and `capture_state`. Do not call `initialize_layout` (it draws from an RNG).

`frame_from_sim`: `synchronize_observation`, then state, action validation and one `mujoco.Renderer` pass per `CAMERAS` entry.

- [ ] **Step 4: Run** — both tests PASS. Then the two-process gate:

```bash
run "$NEW" -m experiments.robosuite_scene.record --trace-only --seed 0 --out /out/r5a
run "$NEW" -m experiments.robosuite_scene.record --trace-only --seed 0 --out /out/r5b
sha256sum "$MAIN"/outputs/robosuite_scene/r5{a,b}/trace.npz   # the two hashes must be equal
```

- [ ] **Step 5: Commit**

```bash
git add experiments/robosuite_scene/{validate,scene,record}.py experiments/robosuite_scene/tests/test_r4_schema.py \
  experiments/robosuite_scene/tests/test_r5_reset.py
git commit -m "Extract the frame schema and a deterministic reset"
```

---

### Task 6: R3 — the scripted expert stacks one pair

**Files:**
- Create: `experiments/robosuite_scene/expert.py`
- Modify: `experiments/robosuite_scene/record.py` (`--expert-trace`)
- Create: `experiments/robosuite_scene/tests/test_expert_fork.py`, `tests/test_r3_stack.py`

**Interfaces:**
- Consumes: `build_sim`, `overwrite_reset`, `contract.GRASP_Z_OFFSET_M`
- Produces: `RobosuiteExpert` with attributes `env`, `model`, `robot`, `config`, `bounds`, `ik`; `run_episode(seed, sink) -> dict` with keys `success`, `reason`, `released_stack`, `final_xy_m`, `final_z_m`, `stable_ticks`, `env`, `model`, `initial_state`, `final_state`, `final_qpos`. `sink(frame, trace)` receives exactly what `pose_snapshot_v2.Expert.run` passes.

- [ ] **Step 1: Write the fork test (runs in `$OLD`, no robosuite import)**

`test_expert_fork`: parse `experiments/robosuite_scene/expert.py` and `experiments/pose_snapshot_v2/expert.py` with `ast` from source text; take each file's `run` function; in the fork, replace every `Name(id="GRASP_Z_OFFSET_M")` with `Constant(0.105)`; assert `ast.dump` of the two functions is equal. Also assert the fork's `run` contains exactly one such `Name`.

- [ ] **Step 2: Write the stack test (`$NEW`)**

```python
    def test_one_scripted_stack(self):
        from experiments.robosuite_scene.expert import run_episode
        result = run_episode(0, sink=lambda frame, trace: None)
        self.assertTrue(result["success"], result["reason"])
        self.assertTrue(result["released_stack"])
        self.assertLess(result["final_xy_m"], 0.02)
        self.assertTrue(0.035 < result["final_z_m"] < 0.045)
        with self.assertRaises(RuntimeError):
            result["env"].step(np.zeros(result["env"].action_dim))
```

- [ ] **Step 3: Run both** — FAIL, module missing.

- [ ] **Step 4: Implement**

`expert.py` imports the same names `pose_snapshot_v2/expert.py` imports, plus `GRASP_Z_OFFSET_M`. `RobosuiteExpert(Expert)` where `Expert` is `pose_snapshot_v2`'s:

- `__init__` does **not** call `super().__init__()` (that loads `SCENE_PATH`). It sets `self.env, self.model, _ = build_sim()`, `self.robot = FrankaPanda()`, `self.config = configuration(self.model, self.robot)` with `expert_version` set to `"robosuite_bringup_pose_snapshot_v2"`, `self.bounds` from `config["expert_command_contract"]["joint_bounds"]`, and `self.ik = IKController(self.robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10., kp_rot=5.)`.
- `run` is a verbatim copy of `pose_snapshot_v2.Expert.run` with one change: `generate_task_states(.105, robot_speed=1.)` → `generate_task_states(GRASP_Z_OFFSET_M, robot_speed=1.)`. Copying `run` to change one thing is this repo's existing pattern for versioning the expert.

`run_episode`: build one `RobosuiteExpert`, `overwrite_reset` on a fresh `MjData`, episode `{"pick": "red", "place": "green", "seed": seed, "instruction": instruction("red", "green")}`, call `run`, and return the keys above (`released_stack`, `xy_m`, `z_m` come from `outcome["final_metrics"]`).

`record.py --expert-trace --seed N --out DIR` runs `run_episode(N, sink)` and saves per-tick `qpos`, `ctrl`, `sim_time`, `substeps`, `phase_index` plus `initial_state` and `final_qpos` to `DIR/trace.npz`.

- [ ] **Step 5: Run and, if needed, tune**

Fork test PASS in `$OLD`. Stack test in `$NEW`. If it fails, read `result["reason"]` and the phase it stopped in before changing anything:

- `reason == "grasp"` (Verify Lift lost the cube) → tuning-protocol knob 1, then knob 2.
- released but outside the xy/z window → knob 2; then stop and ask the user.
- arm not tracking (large `|ctrl[:7] − qpos[:7]|` at phase ends in the trace) → a Task 3 value is wrong; re-check `test_model_contract`.

Every value tried goes in the task report. Then the expert half of R5:

```bash
run "$NEW" -m experiments.robosuite_scene.record --expert-trace --seed 0 --out /out/r5ea
run "$NEW" -m experiments.robosuite_scene.record --expert-trace --seed 0 --out /out/r5eb
sha256sum "$MAIN"/outputs/robosuite_scene/r5e{a,b}/trace.npz   # equal
```

- [ ] **Step 6: Commit**

```bash
git add experiments/robosuite_scene/{expert,record,contract}.py experiments/robosuite_scene/tests/test_expert_fork.py \
  experiments/robosuite_scene/tests/test_r3_stack.py
git commit -m "Stack one scripted pair on the robosuite scene"
```

If `GRASP_Z_OFFSET_M` changed, update the `0.105` literal in `test_r1_image.py` to the chosen value in the same commit; `test_expert_fork` keeps `0.105`, because that is the original's constant.

---

### Task 7: R6 — replay and kinematic agreement

**Files:**
- Modify: `experiments/robosuite_scene/validate.py`
- Create: `experiments/robosuite_scene/tests/test_r6_drift.py`

**Interfaces:**
- Consumes: `run_episode` result and the sink's frames/traces
- Produces: `replay_recorded(result, frames, traces) -> {"max_policy_state_error", "max_full_qpos_error", "stable_ticks"}`; `replay_reference(result, frames, traces) -> {"max_full_qpos_error", "stable_ticks"}`; `kinematic_mismatch(result, traces) -> float` (metres).

Revision 1's R6 only replayed the episode on the same model, which is deterministic and passes regardless of the robot. `kinematic_mismatch` is the check that speaks to the spec's actual risk: the planner's pinocchio URDF and the robosuite Panda must describe the same arm.

- [ ] **Step 1: Write the failing test**

Run `run_episode(0, sink)` collecting `frame["action"]` and the trace keys `qpos`, `ctrl`, `sim_time`, `substeps`. Assert: `replay_recorded(...)["max_policy_state_error"] < 1e-3`; `replay_reference(...)["max_full_qpos_error"] <= 1e-5`; `replay_recorded(...)["stable_ticks"] >= 30`; `kinematic_mismatch(...) < 1e-3`.

- [ ] **Step 2: Run in `$NEW`** — FAIL, names missing.

- [ ] **Step 3: Implement**

- `replay_recorded`: restore `result["initial_state"]` on a fresh `MjData` of `result["model"]`; per tick, `synchronize_observation`, track max `|qpos[:9] − trace qpos[:9]|` and max full-`qpos` error, check `data.time` against `sim_time` (`1e-9`), `apply_action(model, data, action, ACTIVE_JOINTS)`, require `step_control(model, data, 1/30, ARM_DOF)` to equal the recorded `substeps`, count the `released_stack` streak with `stack_metrics`; finish by comparing against `result["final_qpos"]`. This mirrors the loop in `experiments/paired_demonstrations/validate.py`; copy the logic, do not import that module.
- `replay_reference`: same loop, but write `data.ctrl[:] = trace["ctrl"]` instead of `apply_action`.
- `kinematic_mismatch`: for every 10th trace tick, set `qpos` on a scratch `MjData`, `mj_forward`, and return the max distance between `data.xpos[model.body("robot0_right_hand").id]` and `FrankaPanda().forward_kinematics(qpos[:9]).translation`. Measured at `Q_HOME`: 0.5 mm.

- [ ] **Step 4: Run** — PASS. A replay failure means non-determinism in our own stepping: debug it; it is not a tuning matter.

- [ ] **Step 5: Commit**

```bash
git add experiments/robosuite_scene/validate.py experiments/robosuite_scene/tests/test_r6_drift.py
git commit -m "Gate replay and URDF agreement on the robosuite Panda"
```

---

### Task 8: five-episode seal and the go/no-go report

**Files:**
- Modify: `experiments/robosuite_scene/record.py`, `validate.py` (`fingerprint`)
- Create: `experiments/robosuite_scene/tests/test_seal.py`, `experiments/robosuite_scene/README.md`
- Create: `docs/ROBOSUITE_SCENE_BRINGUP_<YYYYMMDD>.md` (after the run)
- Modify: `TODO.md`, `docs/OPERATIONS.md`, `docs/INDEX.md`, `docs/BENCHMARK_STATUS.md`

**Interfaces:**
- Produces: `fingerprint(root) -> str`; `python -m experiments.robosuite_scene.record --out DIR` writes `DIR/{dataset/, trace_seed_0.npz, episodes.json, seal.json}`.

- [ ] **Step 1: Write the failing seal test (`$OLD`, no robosuite import)**

`fingerprint` over a temp dir: unchanged when only `seal.json` changes; changes when any other file changes; raises `ValueError` on a symlink inside the root.

- [ ] **Step 2: Run** — FAIL, `fingerprint` missing.

- [ ] **Step 3: Implement**

`fingerprint`: sha256 over the sorted `{relative path: sha256(file)}` JSON of every file under `root` except `root/seal.json`; reject symlinks. Copy the logic; do not import `experiments.fixed_layout`. `validate.py` must stay importable without robosuite (import it lazily inside the functions that need it).

`record.py` with no mode flag:

- refuse to start unless `mujoco.__version__ == contract.MUJOCO_VERSION` and `--out` does not exist;
- `PairedDataset.create(repo_id="robosuite_bringup", root=out / "dataset", fps=30, features=copy.deepcopy(FEATURES), vcodec=ENCODING["vcodec"], encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)`;
- for seeds `0..EPISODE_COUNT-1`: `run_episode(seed, sink)` where the sink buffers frames; only if `success`, `add_frame` each buffered frame and `save_episode()`; a failed seed aborts the run without writing `seal.json`;
- write `episodes.json` (per seed: frames, `final_xy_m`, `final_z_m`, `stable_ticks`), `trace_seed_0.npz`, then `seal.json` as `{"fingerprint": fingerprint(out), "image": contract.IMAGE, "grasp_z_offset_m": GRASP_Z_OFFSET_M}`.

`README.md`: what the package is, the image and why it is separate, the `run` helper, the gate order, and a line stating the package is not hash-bound to any sealed run.

- [ ] **Step 4: Run everything**

```bash
run "$OLD" -m unittest experiments.robosuite_scene.tests.test_r1_image experiments.robosuite_scene.tests.test_seal \
  experiments.robosuite_scene.tests.test_expert_fork -v
for t in test_r2_render test_model_contract test_r7_cameras test_r4_schema test_r5_reset test_r3_stack test_r6_drift; do
  run "$NEW" -m unittest experiments.robosuite_scene.tests.$t -v || break; done
run "$NEW" -m experiments.robosuite_scene.record --out /out/robosuite_bringup_$(date +%Y%m%d)
```

Expected: all green; five episodes; `seal.json` present. Then load the dataset once in `$NEW` (`LeRobotDataset(repo_id="robosuite_bringup", root=…)`, read frame 0, check the four feature shapes) and decode one video per camera with `media.decode_video`.

- [ ] **Step 5: Write the report**

`docs/ROBOSUITE_SCENE_BRINGUP_<YYYYMMDD>.md`, one row per gate with the measured value:

| id | evidence |
|---|---|
| R1 | both image ids, `R1_OK`, the versions line |
| R2 | `test_r2_render` |
| R3 | `success`, `final_xy_m`, `final_z_m` for seed 0; 5/5 from `episodes.json` |
| R4 | shapes, finger signs, `benchmark.py` sha256 |
| R5 | the two reset hashes and the two expert-trace hashes |
| R6 | the two replay errors and `kinematic_mismatch` |
| R7 | pose test result and the PNG paths, plus one sentence on what visibly differs |

Then: every tuning-protocol value tried; what `apply_scripted_contract` rewrites, so Phase 2 knows how much of the robot is robosuite's and how much is ours; the MuJoCo 3.4.0 vs 3.13.0 split and what it means (no physics-level comparison between robosuite-scene results and sealed results); the unused adapter alternative. Last line exactly `Decision: go` or `Decision: no-go`.

- [ ] **Step 6: Documentation pass and commit**

`TODO.md` B3 Phase 1 checkbox and result; `docs/OPERATIONS.md` gets a short "second image" section (digest, what runs in it, the `--no-deps` reason, `DOCKER_BUILDKIT=0`); `docs/INDEX.md` lists the report; `docs/BENCHMARK_STATUS.md` gets a dated entry.

Agent-facing docs get the same findings, for Copilot and Claude alike: `.github/copilot-instructions.md` (the second image exists and which MuJoCo each image carries; a version pin is proven by running the thing, not by importing it), `.github/instructions/experiments.instructions.md` (`experiments/robosuite_scene/` is not hash-bound and runs only in the robosuite image), `AGENTS.md` if its table needs a row, and `CLAUDE.md` (second image digest beside the first; which image runs which package). Keep each addition to a few lines and do not restate what `docs/OPERATIONS.md` holds.

```bash
git add experiments/robosuite_scene docs/ROBOSUITE_SCENE_BRINGUP_*.md TODO.md docs/OPERATIONS.md docs/INDEX.md docs/BENCHMARK_STATUS.md \
  .github/copilot-instructions.md .github/instructions/experiments.instructions.md AGENTS.md CLAUDE.md
git status --short            # nothing under outputs/, no PNG, no mp4
git commit -m "Report the robosuite scene bring-up go or no-go"
```

---

## Self-review

- Spec coverage: R1 Task 1; R2 Task 2; R7 Task 4; R4 and the reset half of R5 Task 5; R3 and the expert half of R5 Task 6; R6 Task 7; PoC dataset, seal and report Task 8. Task 3 has no spec risk id: it is the work revision 1 folded into a later task and under-specified.
- Deviations from the spec are listed in *What changed* and written into the spec's *Corrections* section by Task 1: MuJoCo pin, camera names, R7 tolerance, tuning protocol, smoke test deferred.
- Review Focus items map to tests: (1) every gate test builds the sim; (2) Task 3 `test_qpos_layout_and_finger_sign` and Task 5's finger-sign assertion; (3) Task 5's settled-z and no-floor-contact assertions; (4) Task 8's recorder version check; (5) Task 6 `test_expert_fork`.
- Names used across tasks match the *Shared names* block.
- No sealed path is in any commit list; Task 1's test fails if one changes.
- Not verified before writing this plan: the full expert on MuJoCo 3.4.0, and whether the unmodified grasp height holds a cube with the shorter robosuite fingers. Tasks 3 and 6 are where those are found out.
