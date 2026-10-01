# robosuite scene bring-up — design

**Date:** 2026-10-01
**Status:** design approved, implementation pending
**Supersedes:** none
**Next:** this spec, then writing-plans → `docs/superpowers/plans/2026-10-01-robosuite-scene-bringup.md`
**Branch:** `feat/robosuite-scene-bringup` (from `main` at `624b156`)

## Why this phase exists

The next data axis the user wants (B3) combines five changes relative to B2:

1. Tighter release precision (expert admission gate ≤ 1.2 cm),
2. Harder distractors (inside the carry path, reduced clearance),
3. Full colour randomisation (the original v2 axis),
4. Extra demonstrations near the acquisition-weak pick regions `054` and `048`,
5. A new MuJoCo scene (floor, lighting, and better props).

The user has accepted the one-variable-per-run attribution trade-off (see
`docs/B2_DISTRACTOR_EVALUATION_20260930.md` for the mechanism analysis that motivated items 1 and
4) and the long-term goal is simulator realism closer to robosuite's house style, so item 5
expands to "switch the scene-composition layer to robosuite". That is a different simulator stack,
not just a texture swap, so **item 5 is its own project with its own phases**. This spec covers
only its Phase 1.

Phase 1 proves the simulator swap works end-to-end on a trivial scene. Phase 2 — the actual B3
dataset collection using the proven robosuite path, with items 1-4 layered on — is a separate
spec, written only after Phase 1's go/no-go.

## Non-negotiable: sealed hashes stay valid

Nothing already-sealed is touched. Concretely:

- No edits to `src/robots/franka_panda/model/*` (`scene.xml`, `smolvla_scene.xml`, meshes, URDF),
  `experiments/fixed_layout/source.py`, `experiments/fixed_layout/validate.py`, or any `*_data/`
  package.
- The pinned Docker image `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`
  stays the admission reference for `b1_language_20260925`, `broad_workspace_20260920`,
  `b2_distractor_20260928`.
- Phase 1 introduces a **new, additionally-pinned image tag** that layers `robosuite` on top of
  the current one. Existing admissions are unaffected because their `required_image` field still
  names the old digest.
- `experiments/robosuite_scene/` is a new package. Shared modules (`data_collection/benchmark.py`,
  `validation/`, `src/controllers/`, `experiments/pose_snapshot_v2/`) are consumed read-only, as
  B2 consumed them.

See [FAILURES.md](../../FAILURES.md) #17, #28 for the class of mistake this rule prevents.

## Approach: thin wrapper

Chosen over "fuller integration" and "deepest/OSC" after a brainstorming pass with the user. The
rule: **robosuite composes the scene; our code drives the physics.**

Scene composition (robosuite's job):

- `robosuite.robots.Panda` with `RethinkMinimalMount` (table-flush, visually like the current
  scene; the default `RethinkMount`'s 1 m pedestal is rejected to keep the view angles sensible).
- A customised `TableArena` subclass with:
  - Table dimensions matching the current scene (verified against `smolvla_scene.xml`),
  - **Two camera elements with the pose and FOV copied bit-for-bit from the current scene's
    `camera_wrist` and `camera_shoulder`** (user-selected during brainstorming; see
    "Camera plan" below).
- Two `BoxObject` cubes (the only props in Phase 1; distractor cubes come in Phase 2).
- `PlacementInitializer` to put the two cubes at a single hand-picked, deterministic layout. No
  randomisation in Phase 1 — one layout is enough to prove the pipeline.

Physics driving (our code's job, unchanged):

- Grab `env.sim.model` and `env.sim.data` immediately after `robosuite.make(...)`.
- Write joint-position targets to `data.ctrl[:7]` with our existing `bounded_joint_target`
  controller. **Do not call** `env.step()`, `env.reward()`, or any robosuite controller
  (`OSC_POSE`, `JOINT_POSITION`, `WHOLE_BODY_IK`). The composite controller is a no-op in Phase 1.
- Reuse `pose_snapshot_v2.planner.SnapshotTaskSpaceTrajectory`, our pinocchio IK, our
  `GraspController`, `experiments/fixed_layout/source.py`'s `Expert`, and the LeRobot-wrapper
  used by B1/B2 collection, verbatim.
- Reset: do **not** rely on `env.reset()`'s randomness. Reset by writing the frozen
  `arrays['qpos']` + `arrays['initial_state']` into `data` directly (same mechanism B2 uses).
  The `env.reset()` call happens exactly once per sim handle — to construct the model and data
  objects, after which we overwrite them.

### Camera plan (concrete)

Extract from `src/robots/franka_panda/model/smolvla_scene.xml`:

- Pose (`pos` and `quat`) of `camera_wrist` and `camera_shoulder`,
- FOV (`fovy`), resolution (`height`/`width` or inherited default),
- Any site offsets if the cameras are attached to a non-world body.

Add two `<camera>` tags with identical values into the robosuite arena's MJCF. Resolution is
pinned at 512 × 512 to match the LeRobot schema B1/B2 use. Name them `camera_wrist` and
`camera_shoulder` so the per-episode render loop can address them by name without conditional
logic.

Set `robosuite.macros.IMAGE_CONVENTION = "opencv"` so images arrive unflipped, matching what B1/B2
collected.

## In scope (deliverables of Phase 1)

1. **New package** `experiments/robosuite_scene/`:
   - `arena.py` — custom `TableArena` subclass carrying the two replicated cameras,
   - `scene.py` — compose arena + Panda + minimal mount + two cubes + deterministic placement,
   - `expert.py` — thin shim that wires `pose_snapshot_v2`'s planner to the robosuite-built sim,
   - `record.py` — a trimmed-down collector that produces 5 episodes to the LeRobot schema,
   - `validate.py` — determinism / schema / image-convention asserts,
   - `tests/` — CPU-only unit tests (90-second budget), analogous to B2's test suite.
2. **New Docker image tag** `vla-robosuite:<new sha256>`, built from the current pinned image via
   `pip install robosuite==X.Y.Z` (version pinned in the Dockerfile). The current pinned digest is
   not touched.
3. **A 5-episode PoC dataset** under `outputs/robosuite_scene/robosuite_bringup_<date>/`, sealed
   with its own fingerprint. Not published, not trained on. Byte-reproducible: re-running the
   collector against the same scripted plan produces identical bytes.
4. **A short go/no-go report** `docs/ROBOSUITE_SCENE_BRINGUP_<date>.md` summarising the risk
   results below and declaring whether Phase 2 can start.

## Risks (what Phase 1 must resolve)

| id | risk | pass criterion |
|---|---|---|
| R1 | robosuite install breaks the Docker image | `python -c "import robosuite, mujoco, lerobot, pinocchio, torch"` succeeds inside the new image tag |
| R2 | OSMesa headless rendering fails under robosuite | `robosuite.make(...)` renders two 512 × 512 PNGs without an X display; images are not blank and are in OpenCV orientation |
| R3 | Writing `data.ctrl[:7]` directly conflicts with robosuite's controller | one scripted pick-and-place runs with the composite controller untouched; the sim ends with the pick cube on top of the base cube |
| R4 | 9-state / 8-action schema cannot be extracted cleanly | `observation.state` and `action` have exactly those shapes and dtypes; `benchmark.synchronize_observation` is unchanged |
| R5 | Reset is not bit-deterministic | two runs of the same scripted expert produce byte-equal `trace.npz` files (sha256 equal) |
| R6 | `pose_snapshot_v2` planner fails on robosuite's Panda URDF | the scripted expert hits the pick and place targets within the current admission tolerance (max replay drift < 1 e-3 rad) |
| R7 | Camera replication does not match the current scene's view | side-by-side screenshots of `camera_wrist` and `camera_shoulder` are visually indistinguishable from the current scene except for the floor/table that robosuite's arena provides |

Each risk is checked by a specific test or artefact. **Any single risk failing means stop and
redesign Phase 2** rather than patch around it.

### Smoke test (bonus, time permitting)

Load the B2 winning checkpoint (`xvla_franka_stacking_b2_distractors_v1_20260928/checkpoints/step_050000/`)
and run 4 trials against the Phase 1 scene (red → green stacking, 4 seeds). Expected: close to
the 92.5% control-arm rate; **the point is not the score, it is that the pipeline runs end-to-end
with real weights**. If the score is dramatically different that is itself a finding and surfaces
whether the robosuite scene's visuals differ from B2's enough to matter — directly informing
whether the B3 scene change needs to be bounded in Phase 2.

## Out of scope (reserved for Phase 2 / later)

- Any new axis: no release-precision gate change, no new distractors, no colour randomisation, no
  pick coverage near 054/048, no floor-texture change.
- Any training beyond `--preflight`.
- robosuite's teleoperation stack, OSC controller, built-in reward, Isaac Sim rendering.
- Mobile-manipulation or dual-arm variants.
- B1b vertical variation (deferred — see TODO.md).

## Open decisions (left to writing-plans)

1. Exact robosuite version pin.
2. Whether `arena.py` subclasses `TableArena` or writes a bespoke MJCF. Likely subclass — simpler
   diff, inherits the physics gate.
3. Whether the PoC dataset uses the LeRobot v3.0 wrapper directly or a thinner `np.savez_compressed`
   format for the PoC only. Favour LeRobot to prove schema parity.
4. Whether the B2 smoke test runs inside the same image tag or a derivative (needs `pin` and
   `pinocchio` already present — they are, in the current image).

## Budget

| stage | cost |
|---|---|
| package scaffold + arena + scene + tests | ~2 days implementation |
| Dockerfile + image build | ~0.5 day |
| 5-episode PoC collection | ~15 min wall-clock |
| B2 smoke test (if R1–R7 all green) | ~15 min wall-clock |
| go/no-go report | ~0.5 day |

Target: Phase 1 completed and reported within one working week. If any risk is still open at that
point, replan.

## Success criterion

A single decision in the final report: **go** (Phase 2 B3 spec starts) or **no-go** (identify the
specific risk and the specific redesign needed). No partial credit.
