---
name: working-on-the-robosuite-path
description: "Use when running, testing, collecting, rolling out or debugging anything under experiments/robosuite_*, when deciding which of the two Docker images a command belongs in, when a robosuite import, reset or render misbehaves, or when moving a scene, checkpoint or layout between the MuJoCo pipeline and the robosuite scene."
---

# Working on the robosuite path

The repo has two worlds. The **sealed pipeline** (every `*_data`, training and eval package) runs
in the admission image on MuJoCo 3.13.0 and is hash-bound. The **robosuite path**
(`experiments/robosuite_scene/`, `experiments/robosuite_sim2sim/`) runs in a second image on
MuJoCo 3.4.0 and is not hash-bound. Image names, digests and why there are two:
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

A new `experiments/robosuite_*` package is not hash-bound until someone gives it a contract. Say
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

## Names in the robosuite-built model (before the contract renames them)

`robot0_link0..7`, hand `robot0_right_hand`, fingers `gripper0_right_leftfinger` /
`gripper0_right_rightfinger` (→ `left_finger` / `right_finger`), finger joints
`gripper0_right_finger_joint1/2` (joint 2 runs negative until the contract flips it), a `BoxObject`
named `X` is body `X_main` with joint `X_joint0`, table geom `table_collision`, floor geom `floor`.

## Before reporting a robosuite result

State which image and MuJoCo it ran on, what the control scored, and that the package has no
contract. See [evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md) for trial counts and
[diagnosing-policy-failures](../diagnosing-policy-failures/SKILL.md) before naming a cause.
