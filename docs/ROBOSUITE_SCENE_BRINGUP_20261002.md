# Robosuite scene bring-up: go/no-go — 2026-10-02

Phase 1 of TODO B3. Question: can a scene composed by robosuite 1.5.2 be driven by this repo's
scripted expert and recorded in the existing LeRobot schema, with deterministic resets? Nothing new
was varied: one layout, two cubes (red on green), the two existing cameras, five episodes.

Package: `experiments/robosuite_scene/`. Spec:
[superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md](superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md)
(see its *Corrections* section). Branch `feat/robosuite-scene-bringup`.

Every number below was measured on 2026-10-02 by the runs named beside it. The whole test suite was
re-run for this report: 8 tests OK in the admission image, 16 tests OK in the robosuite image.

## Gates

| id | result | evidence |
|---|---|---|
| R1 | pass | Admission image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (unchanged, MuJoCo 3.13.0). Robosuite image `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (tag `vla-robosuite:1.5.2-mj3.4.0`). Runtime gate in the robosuite image (`robosuite.make('Lift')`, `env.reset()`, 512×512 offscreen render, `PairedDataset` import) printed `R1_OK`; versions robosuite 1.5.2 / mujoco 3.4.0 / numpy 2.2.6 / lerobot 0.4.4. `test_r1_image` 2/2 (no sealed path differs from `10f2524`). |
| R2 | pass | `test_r2_render` 1/1: both cameras 512×512×3 and non-blank under `MUJOCO_GL=osmesa`; mean absolute pixel difference of our render against robosuite's OpenCV-orientation render 0.163 (`front_cam`) and 2.567 (`gripper_cam`), against the unflipped OpenGL form 104.9 and 7.00; `env.step` raises. |
| R3 | pass | `test_r3_stack` 1/1 (seed 0): `success` True, `final_xy_m` 0.00912, `final_z_m` 0.04099, 180 stable ticks. Collector: 5/5 seeds succeeded, 528 frames each, `final_xy_m` 0.00861–0.01265, `final_z_m` 0.04098–0.04099, 180 stable ticks each (`episodes.json`, table below). |
| R4 | pass | `test_r4_schema` 2/2: `observation.state` float32 (9,), `action` float32 (8,), images uint8 (512, 512, 3), both fingers within 1e-3 of +0.04 when open (same sign for both, as in the current scene), two renders of one state byte-identical. `data_collection/benchmark.py` sha256 `1c11e65b620da8d4aa3b89cba46d1fc832e718f5d6a5615104a4be05d03c10e4`, unchanged. |
| R5 | pass | `test_r5_reset` 1/1. Two separate processes, reset trace (`--trace-only`): both `2b1af227edebac973b2cc2e3f6aa19bcb4dfcdbd27f5720274ac9745f60c9c06`. Two separate processes, full expert trace seed 0 (`--expert-trace`): both `5bf06859017355b47ccedc08e3f373b61f2ea977cb6cd816c27ec52b559853c9`. The collector's `trace_seed_0.npz` has that same hash. |
| R6 | pass | `test_r6_drift` 3/3 (seed 0): replay of the recorded policy actions, max policy-state error 7.79e-08 rad (full qpos 1.53e-07), 180 stable ticks; replay of the recorded `ctrl`, full-qpos error 0.0, 180 stable ticks; `kinematic_mismatch` (robosuite hand body against the URDF `panda_hand`, every 10th tick) 5.0e-04 m against a 1e-3 m limit. |
| R7 | pass | `test_r7_cameras` 1/1, two arm configurations: `front_cam` 0.000000 m / 0.0001°, `gripper_cam` 0.000501 m / 0.0282° (limits 1 mm / 0.1°), `fovy` equal. PNGs: `outputs/robosuite_scene/camera_compare_v2/{front_cam,gripper_cam}_{current,robosuite}.png` and `side_by_side.png`. Framing and cube positions match; what differs is the surroundings: robosuite shows a white table on a wood floor with a grey wall and a mount ring at the base, the current scene a blue checker floor with shadows and two extra (blue, yellow) cubes. |

Commands: `run "$OLD" -m unittest …test_r1_image …test_seal …test_expert_fork` (8 tests, OK) and one
`run "$NEW" -m unittest …<module>` per robosuite module (1 + 7 + 1 + 2 + 1 + 1 + 3 = 16 tests, OK).
`run`, `$OLD` and `$NEW` are in the package README.

## The five-episode dataset

`outputs/robosuite_scene/robosuite_bringup_20261002/` (23 MB), written by
`python -m experiments.robosuite_scene.record --collect` in the robosuite image, 6 min 01 s
wall-clock (70–73 s per episode including AV1 encoding).

| seed | frames | final_xy_m | final_z_m | stable_ticks |
|---|---|---|---|---|
| 0 | 528 | 0.00912 | 0.04099 | 180 |
| 1 | 528 | 0.00861 | 0.04099 | 180 |
| 2 | 528 | 0.01265 | 0.04098 | 180 |
| 3 | 528 | 0.01002 | 0.04099 | 180 |
| 4 | 528 | 0.00909 | 0.04099 | 180 |

Seal fingerprint `543e300c257d3d3dae668482f389d3a9f4d407fe284f09f521faf3ab2b68bb63`
(`seal.json`; sha256 over every file under the root except `seal.json`), recomputed and equal
before and after loading. `LeRobotDataset(repo_id="robosuite_bringup")` loads 5 episodes / 2640
frames at 30 fps; frame 0 has `action` (8,), `observation.state` (9,) and both images 3×512×512
float32 (declared (512, 512, 3)). `media.decode_video` decoded 2640 AV1/yuv420p frames from each
of the two camera files.

Frames extracted from the decoded videos (episodes 0 and 4, tick 264 and tick 527): mid-episode,
`front_cam` shows the arm carrying the red cube between the fingers above the table with the green
cube still on the table, and `gripper_cam` shows the held red cube at the bottom edge with the
green cube ahead; at the end the arm is back at home and the red cube sits on the green one. In
the final `gripper_cam` frame the red cube is visibly not centred on the green one, which is the
xy offset in the table above. No collision meshes, no garbage frames; episode 4 looks like
episode 0.

## Two MuJoCo versions

The robosuite path runs on MuJoCo **3.4.0** in its own image; the sealed pipeline stays on
**3.13.0** in the admission image, which was not rebuilt or retagged.

- robosuite 1.5.2 imports on 3.13.0 but cannot reset an environment there: its `binding_utils`
  asserts on enums that changed, and `data.qM` was removed (FAILURES #39).
- The image installs `robosuite==1.5.2` and `mujoco==3.4.0` with `--no-deps`, because robosuite's
  dependency `mink` requires numpy<2 and would downgrade numpy 2.2.6. `pip check` therefore lists
  `mink`, `opencv-python`, `pytest` and `qpsolvers` as missing; nothing in this package uses them.
- Consequence: **there is no physics-level comparison between robosuite-scene results and sealed
  results.** They differ in engine version as well as in robot model, table and contact settings.
  A policy number measured on this scene is a number about this scene.
- Unused alternative: a process-local adapter (`mujoco_compat.py`: integer casts for the enum
  asserts, a `qM` alias, a translated `mj_fullM` call) let robosuite 1.5.2 reset on 3.13.0. It was
  verified for reset only and is not in the package. It is the route back if a same-engine
  comparison is ever wanted.

## What `apply_scripted_contract` rewrites

robosuite composes the model; `xml_contract.apply_scripted_contract` then edits the XML before it
is compiled. `env.step()` is never called; the expert writes `data.ctrl` and steps MuJoCo itself.

Rewritten to this repo's values:

1. Body names: `gripper0_right_leftfinger` → `left_finger`, `gripper0_right_rightfinger` →
   `right_finger`, `target_cube_{red,green}_main` → `target_cube_{red,green}`.
2. Finger joint 2: axis `0 -1 0`, range `[0, 0.04]` (robosuite: `[-0.04, 0]`), so both fingers
   read positive when open, as the 9-value state expects.
3. Joint dynamics on the 7 arm and 2 finger joints: armature 0.1, damping 1, frictionloss 0.
4. Integrator `implicitfast` (`build_sim` then sets the timestep through `configure_timing`).
5. Actuators: robosuite's are removed and replaced by the eight `<general>` position actuators of
   `panda.xml` (`actuator1..8`, same gains, force ranges and control ranges), the eighth on a
   fixed tendon.
6. A fixed tendon `split` over the two finger joints and a joint equality coupling them.
7. Cube collision geoms: mass 0.05, size 0.02, friction `1.2 0.005 0.0001`, solref `0.002 1`,
   solimp `0.9 0.95 0.001`, margin 0.001.
8. **Fingertip pads re-parented.** robosuite keeps each pad collision box in a jointless child body
   (`gripper0_right_finger_joint{1,2}_tip`). Each pad geom is moved into `left_finger` /
   `right_finger` with its pose kept; the tip bodies and their inertials stay. Needed because
   `GraspController.is_grasped` and `stack_metrics["finger_contact"]` look for contacts on the body
   ids `left_finger` / `right_finger`, and with the elliptic cone the pads are the only gripper
   geoms that touch the cube.
9. Render groups: geoms in group 0 → 3, all sites → 4, so default-option renders show visual
   geoms only.
10. Offscreen framebuffer 512×512, and `front_cam` / `gripper_cam` if absent (the arena and
    `scene.py` already add them, the second on `robot0_right_hand`).

Still robosuite's: link kinematics, meshes and inertias (hand 0.5 mm / 0.028° off the current
scene's, fingers about 6.5 mm shorter), the finger pads' geometry and friction, the contact cone
(`elliptic`, `impratio` 20), the table, floor, wall and mount.

## Tuning history

R3 was red on the first pass. All attempts are seed 0.

| # | GRASP_Z_OFFSET_M | cone / impratio | pad geoms | success | reason | final xy (m) | final z (m) |
|---|---|---|---|---|---|---|---|
| 1 | 0.105 | elliptic / 20 | in tip bodies | False | grasp | 0.21438 | 0.21330 |
| 2 | 0.0985 | elliptic / 20 | in tip bodies | False | grasp | 0.21425 | 0.21301 |
| 3 | 0.095 | elliptic / 20 | in tip bodies | False | grasp | 0.21442 | 0.21311 |
| 4 | 0.105 | pyramidal / 1 | in tip bodies | False | not_released_and_stable_one_second | 0.04353 | 0.00000 |
| 5 | 0.0985 | pyramidal / 1 | in tip bodies | False | not_released_and_stable_one_second | 0.04674 | 0.00000 |
| 6 | 0.095 | pyramidal / 1 | in tip bodies | False | not_released_and_stable_one_second | 0.04507 | 0.00000 |
| 7 | 0.105 | elliptic / 20 | in finger bodies | True | success | 0.00912 | 0.04099 |

In 1–3 the cube was physically held (lifted 0.213 m, both pads in contact for 59 of 59 Verify Lift
ticks) while the grasp check saw 0 of 59, because the contacts were on the tip bodies. In 4–6 the
finger meshes also touched, so the check passed, but the cube ended on the table 43–47 mm from the
green cube; that was not diagnosed. Attempt 7 is the contract fix (item 8 above); its first four
phases are numerically identical to attempt 1. Final knobs are unchanged from the plan:
`GRASP_Z_OFFSET_M = 0.105`, elliptic / 20. No success window, IK gain, shared module or test
tolerance was changed. See FAILURES #40.

## Open issues

- **Systematic xy placement offset, undiagnosed.** All five seeds land 8.6–12.7 mm from centre
  against a 20 mm success window; seed 2 has 7 mm of margin. In seed 0 it is almost all in x
  (about −9 mm) and is already present while the cube is carried. Phase 2 plans a tighter release
  gate (≤ 12 mm), which seed 2 would fail today.
- **Settled cube height 0.02097 m** against a nominal 0.02 with a 1 mm test tolerance: 0.03 mm of
  margin. It is the 1 mm contact margin copied from the current scene; deterministic.
- **Two OSMesa contexts cannot interleave in one process.** robosuite's renderer and
  `mujoco.Renderer` give garbage if alternated, and closing ours breaks robosuite's. The package
  renders only through `mujoco.Renderer` after `build_sim`. The collector builds five environments
  in one process and checks each seed's first frame is pixel-identical to seed 0's; it was.
- **Finger pad friction and geometry are robosuite's**, not matched to the current scene. The pads
  are 16 mm tall and hold the cube by its upper half.
- **The B2-checkpoint smoke test was not run.** It is deferred to a follow-up plan.
- **R6's numbers are replay numbers** on one seed: they show the recorded episode reproduces
  (7.79e-08 rad, 0.0) and that the URDF used for IK agrees with robosuite's hand to 0.5 mm. They
  say nothing about the placement offset.
- The package is not hash-bound: `seal.json` records a file fingerprint, the image and the grasp
  height, not source hashes.

Decision: go
