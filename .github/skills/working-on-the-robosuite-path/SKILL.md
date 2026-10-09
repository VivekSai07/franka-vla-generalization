---
name: working-on-the-robosuite-path
description: "Use when running, testing, collecting, rolling out or debugging anything under experiments/robosuite_* or experiments/props_scene, when deciding which of the two Docker images a command belongs in, when a robosuite import, reset or render misbehaves, or when moving a scene, checkpoint or layout between the MuJoCo pipeline and the robosuite scene."
---

# Working on the robosuite path

The repo has two worlds. The **sealed pipeline** (every `*_data`, training and eval package) runs
in the admission image on MuJoCo 3.13.0 and is hash-bound. The **robosuite path**
(`experiments/robosuite_scene/`, `experiments/robosuite_sim2sim/`) runs in a second image on
MuJoCo 3.4.0. `experiments/robosuite_sim2sim/` and the `robosuite_scene` README are not
hash-bound; **`experiments/robosuite_scene/*.py` and `experiments/b3_robosuite_data/` are bound by
the B3 seal** (2026-10-03), so do not edit them — fork. Image names, digests and why there are two:
[docs/OPERATIONS.md](../../../docs/OPERATIONS.md). How to run:
`experiments/robosuite_scene/README.md`. What was measured:
`docs/ROBOSUITE_SCENE_BRINGUP_20261002.md`.

**Core idea: robosuite composes the scene; our code drives it.** After one `env.reset()` the
compiled XML is rewritten by `xml_contract.apply_scripted_contract` and stepped with plain
`mujoco`. `env.step()` is banned, and robosuite's controllers are never used.

## Which image

| doing | image |
|---|---|
| anything that derives, verifies or re-hashes sealed data: bank scenes, seals, admission, `provenance()` | admission (3.13.0) only — it does not reproduce bit-exactly on another MuJoCo |
| importing robosuite, building the scene, collecting, rolling out a policy on it | robosuite image (3.4.0) |
| unit tests that never import robosuite (`test_r1_image`, `test_seal`, `test_expert_fork`) | admission |

Crossing the boundary: export **plain arrays** (qpos, qvel, ctrl) from the admission image and
load them in the robosuite image. `robosuite_sim2sim/starts.py` → `rollout.py` is the pattern:
both containers mount the same host folder under `outputs/` read-write at `/out`, the first writes
an `.npz` there and the second reads it.

A new `experiments/robosuite_*` package is not hash-bound until someone gives it a contract
(`b3_robosuite_data` is the pattern: `contract.provenance()` plus a seal). Say
so next to any number it produces, and give it a contract before a result is reported as final.

## Traps, each one hit for real

| trap | what to do |
|---|---|
| A package version "works" because it imports | Prove a pin by running the thing: a real `env.reset()` and render (FAILURES #39). |
| `pip install robosuite` downgrades numpy and adds a second OpenCV | `--no-deps` plus only what it imports; see the package `Dockerfile`. Build with `DOCKER_BUILDKIT=0`. |
| A check says "grasp failed" / "no contact" | Look at the physical state first: cube height in the trace, a video frame. The check was blind while the cube was held 0.21 m up (FAILURES #40). Do not tune until you have looked. |
| Shared code looks things up by body name or body id | robosuite's names differ and its pad geoms live in child `_tip` bodies. Fix it in `apply_scripted_contract` with a test against the compiled current scene, never in shared code. |
| Two renderers | OSMesa contexts conflict. One `mujoco.Renderer` at a time; never call `env.sim.render` after `build_sim()`; assert `image.std() > 0`. |
| Collision meshes show in images | robosuite keeps coloured collision geoms in group 0; the contract moves them to group 3. |
| Mount fails or creates root-owned folders in a worktree | `outputs`, `datasets`, `.cache` are absent from worktrees. Create the empty mount-point directory first, or mount at a path outside `/vla` (outputs go to `/out`). |
| Comparing a robosuite number with a sealed number | Not comparable (different MuJoCo, different scene). Run the control in the same image with the same loop, starts and seeds. |
| "The dataset fingerprint changed" | Two collections differ in `meta/stats.json` key order only. Robosuite datasets are not byte-reproducible yet. |
| Side-by-side images look different between the scenes | Check both sides were put in the same state first (arm, **fingers**, cubes) before blaming the simulator. |
| A long collection dies with `No space left on device` while the disk is nearly empty | `HF_HOME` is on the container's `/tmp` tmpfs and every dataset reopen writes a whole-dataset arrow cache (31 MB at 541 episodes). Size `/tmp` for the sum, or do not reopen per episode (FAILURES #41). `--resume` needs the journal and dataset episode counts to agree. |
| A scene with objects other than cubes (props) | `experiments/props_scene/` is the pattern: own arena subclass, own XML contract (the robot half copied, tested against the current scene), props renamed `prop_<name>`, masses set on the collision geom. Put the success definition and a placement diagnostic side by side and run the physically ideal outcome of the largest object through the definition before sizing a gate (FAILURES #46). |
| A grasp planner that works for the cube fails near the robot base | Check reach along the path, not only at the grasp pose: a straight line from home to a hover point can cross a region the arm cannot hold a vertical hand in (joint 4 saturates), while the grasp pose itself is reachable (FAILURES #49). The Panda limits in the robosuite model are the real ones (joint 4 −3.0718 to −0.0698), narrower than the URDF's. |
| Replaying recorded float32 actions drifts a prop but not the arm | Apply the float32 value you record; compare object poses in replay, not only the robot's state (FAILURES #48). |
| A parallel job over many models, or a probe next to a running gate | Each worker holds a few GB (44 workers exhausted 125 GB); use at most about 30 and contiguous chunks per model. OSMesa rendering is multi-threaded and slows a running gate. A killed command may leave its launching shell loop running (FAILURES #50). |
| `Ran 0 tests ... OK` | A run started right after a file was written saw it empty. Read the test count. |
| A `ManipulationEnv` subclass used with `env.step` (OSC) | It needs a `reward()`; `experiments/props_scene/osc_comparison.py` subclasses and returns 0 without touching the scene module. Robosuite's two gripper joints have opposite signs: opening is the finger gap, not the sum. |

## Names in the robosuite-built model (before the contract renames them)

`robot0_link0..7`, hand `robot0_right_hand`, fingers `gripper0_right_leftfinger` /
`gripper0_right_rightfinger` (→ `left_finger` / `right_finger`), finger joints
`gripper0_right_finger_joint1/2` (joint 2 runs negative until the contract flips it), a `BoxObject`
named `X` is body `X_main` with joint `X_joint0`, table geom `table_collision`, floor geom `floor`.

## Before reporting a robosuite result

State which image and MuJoCo it ran on, what the control scored, and that the package has no
contract. See [evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md) for trial counts and
[diagnosing-policy-failures](../diagnosing-policy-failures/SKILL.md) before naming a cause.
