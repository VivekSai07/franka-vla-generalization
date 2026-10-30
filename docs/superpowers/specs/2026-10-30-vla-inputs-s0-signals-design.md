# S0 — extra-signal sidecar for the sealed P2 episodes — design

**Date:** 2026-10-30. **Status: draft for external review.** Part of the
[VLA inputs programme](2026-10-30-vla-inputs-programme-design.md). CPU only, robosuite image. No package exists; one
read-only probe was run (below).

## Question

Can every P2 frame be given exact wrist force/torque, finger contact forces and target bounding boxes, recorded at the instant
of the frame's observation, **without changing a single sealed byte and with proof that the re-simulated episode is the sealed
one**?

## What is established (read or measured 2026-10-30)

| fact | where |
|---|---|
| per-episode trace in the sealed root: `episodes/episode_NNN.npz` with `qpos` (452 × nq, float64), `ctrl`, `props_pose`, `initial_state`, `final_qpos`, object, bin, slot, start, distractors | `outputs/props_p2_data/props_p2_20261013/episodes/`, written by `props_p2_data.record.Committer` |
| all 540 slots admitted at attempt 0 (r = 0, k = 0) | `journal.jsonl`, 540 lines |
| **replaying the sealed `action` rows from the sealed `initial_state` reproduces the sealed `qpos` byte for byte** (episodes 0 cereal, 271 can, 539 bread: 452 of 452 frames equal, max absolute error 0.0), and the sealed `observation.state` equals `qpos[:, :9]` cast to float32 | probe 2026-10-30, robosuite image, read-only mounts: model from `props_p2_data.episode.expert_for(names_of(e), e["blue_slot"])`, `restore_state`, then per tick `synchronize_observation` → compare → `apply_action` → `step_control(1/30)` |
| the same loop is P2's admission replay (`props_scene.validate.replay`), which passed for all 540 episodes, but only against a tolerance | `props_p2_data/validate.py`, `props_scene/validate.py` line 46 |
| `synchronize_observation` = capture state, `mj_forward`, restore state: contacts and constraint forces are fresh, the integrator state unchanged | `data_collection/benchmark.py` line 36 |
| the policy rollout (P2 evaluator) has the same structure: `synchronize_observation` → read `qpos[:9]` → act → `apply_action` → `step_control` | `experiments/xvla_props_p2_eval/rollout.py` lines 72–88 |
| the scene has **no** force/torque sensors; finger contact today is a boolean scan of `data.contact` | `props_scene/metrics.py`, `robosuite_scene/xml_contract.py` |
| hand body `robot0_right_hand`; finger bodies `left_finger`, `right_finger` | `robosuite_scene/xml_contract.py` lines 9–19 |
| cameras `front_cam` (fovy 60) and `gripper_cam` (fovy 75, on the hand), 512 × 512 | `robosuite_scene/cameras.py` |
| each prop body `prop_<name>` has exactly one collision geom plus visual geoms; masses set explicitly (can 0.15, bread 0.05, lemon 0.08, milk 0.20, bottle 0.20, cereal 0.15 kg) | `props_scene/xml_contract.py`, `props_scene/constants.py` |
| every file reachable from P2's `provenance()` (76 files) is frozen | `outputs/props_p2_data/props_p2_20261013/provenance.json` |

## Design

### Action replay, not re-running the expert

For each of the 540 episodes: build the frozen model, restore the sealed `initial_state`, and per tick run
`synchronize_observation` → **check `qpos` byte-equal to the sealed `qpos[t]`** → compute the signals → `apply_action(sealed
action[t])` → `step_control`. Frozen modules are only imported to build the model and run the shared physics helpers; nothing
is patched. The signals are computed at the same point of the loop where the policy rollout reads its state, so training and
evaluation see the same instant **by construction**, and the evaluator calls the same function (M0).

Signals are computed on a **scratch copy** (`mujoco.mj_copyData`) so the live simulation is never written.

### Signals (per frame; all float64 in the sidecar)

| name | dims | definition |
|---|---|---|
| `wrist_wrench` | 6 | force (3, N) and torque (3, N·m) between the hand body and its parent, at the hand body's origin, in the hand frame, with MuJoCo's force/torque-sensor convention (`mj_rnePostConstraint` on the scratch copy, then the sensor's transform). Gravity of hand, fingers and anything held is included, as on a real wrist sensor without compensation |
| `finger_force` | 2 | per finger body (`left_finger`, `right_finger`): sum of the normal components of `mj_contactForce` over contacts between that finger and any geom outside the robot, in N |
| `target_box` | 2 × 5 | per camera: the target body's rendered geoms (mesh geometry or a documented conservative primitive bound, clipped at the near plane) projected with the camera pose (`cam_xpos`, `cam_xmat`) and `cam_fovy` into the 512 × 512 image; `(u_min, v_min, u_max, v_max)` / 512 clipped to [0, 1], plus `in_frame` (1 if near-plane-clipped geometry projects to a positive-area box intersecting the image, else 0 with the four values 0). Occlusion is ignored by design |
| `props_box` | n_props × 2 × 5 | the same for every prop present. **Recorded, not used by M3** (for a later drawn-on-image or all-object variant, so S0 is not re-run) |

### Review requirements for signal semantics

[Audit R6/R7](../../VLA_INPUTS_AUDIT_20261030.md): exact positions prove the recorded kinematic trajectory, not an
independent historical force measurement. Bind full initial integration state, action bytes, timing, model/runtime and
signal-source closure. Compare reference sensors from the same pre-forward state/solver warm start as the scratch path;
`synchronize_observation` restores integration state after forwarding. Diagnose any reference mismatch without relaxing G3.
Non-perturbation includes full integration state and subsequent replay, not only qpos.

The `in_frame` bit means geometric projection validity, not occlusion-aware visibility. Clip primitives/mesh geometry
against the camera near plane before perspective division; include edge intersections and compiled geom/mesh transforms.
Use the renderer's geom-group convention; if conservative collision bounds are included, document their looseness.
Keep G4 containment, but add analytic tight-box and fully occluded/out-of-frame/behind-camera fixtures plus a full-image-box
mutation. A constant full-image box must fail a correctness test.

Force correctness uses controlled static known-load, zero-contact and known-contact fixtures (both fingers and external
body filtering). Positive force on every elevated trajectory frame and mass-times-gravity differences between unrelated
moving poses are not invariants; record trajectory distributions descriptively instead. Keep the native-sensor parity gate.
Check whether phase_index exists in the sealed files before inferring phases from height/actions. Signal timestamps are
pre-command and must never use future action/state rows. Seal explicit episode/frame identities and report/statistics files.

### Gates (each must pass before the next step; a failed gate stops S0, nothing is relaxed)

| gate | check |
|---|---|
| G1 replay exact | all 540 episodes: every frame's `qpos` byte-equal to the sealed `qpos[t]`, `props_pose` equal to the sealed `props_pose[t]`, the sealed `observation.state` equal to `qpos[t, :9]` as float32, and the final state equal to `final_qpos`; any difference stops the run (no tolerance; the admission gate's tolerance is not used) |
| G2 non-perturbation | one episode replayed with and without the signal computation: identical full integration state and subsequent `qpos` at every frame (the scratch copy really is a copy) |
| G3 wrench correctness | on 3 episodes' frames: `wrist_wrench` equals MuJoCo's own `force` and `torque` sensors on a site at the hand origin, in a copy of the model built with `MjSpec` plus that site and sensors, given the same full state (`mj_setState` with integration state including warm start, then `mj_forward`), to 1e-9 absolute |
| G4 box correctness | 60 frames from 10 episodes, both cameras: every pixel of the target in a **segmentation render** lies inside the projected box expanded by 1 pixel (containment is exact for a correct projection, whatever the occlusion; a flipped axis or wrong fovy fails it). The IoU of segmentation box and projected box is reported, not gated |
| G5 seal | fingerprint over the sidecar files, bound to the P2 seal fingerprint `b1d642b3…f4d898` and to S0's own source hashes (`provenance()`); re-verified in a second process |

If G1 fails on any episode, S0 stops and reports which episodes and where. The fallback (re-running the expert itself,
determinism-proven, with a hook at the observation instant) needs a spec amendment, because it would wrap frozen code at run
time.

### Descriptive report (from demonstrations only; no policy involved)

Per object and segment (*before close*, *closed and lifted*, *after release*, defined from the sealed actions and object
heights; inspect sealed NPZ keys first: the collector declares phase_index): distributions of `wrist_wrench` and `finger_force`, their frame-to-frame variation during the
steady hold, the 0.1 and 99.9 percentiles over training rows (M2 clips to these; fixed here, before M2 trains), the largest
|z| after mean/std normalisation, and `gripper_cam` box in-frame rate by segment (not occlusion-aware visibility).

## Output

`outputs/vla_inputs_signals/p2_signals_<date>/`: `episodes/episode_NNN.npz` (the arrays above, 452 frames each), `index.json`
(per-episode sha256), `provenance.json`, `seal.json`, `report.json`. Never under `datasets/`. Disk well under 1 GB.

## Package

`experiments/vla_inputs_signals/` (robosuite image): `signals.py` (pure functions: wrench, finger forces, projection; used
here and by M0's evaluator), `replay.py` (the loop above), `sidecar.py` (gates, write, seal), `contract.py` (`SOURCE_FILES`,
`provenance()`, seal record). After the seal the package is hash-bound: **fork, do not edit**. M0's evaluator checks S0's
provenance before running A2 or A3, so the rollout signal code is the sealed one.

## Costs

Unrendered physics is about 0.6 s per P2 episode (P2 measurement); with the signal computation and checks, 540 episodes take
minutes on a few pinned workers.

## What S0 cannot show

- That a real sensor would read the same values: these are exact simulator quantities, contact chatter included.
- Anything about policies.

## Out of scope

Force/torque history windows, filtering, gravity compensation, occlusion-aware boxes, drawing boxes on images, any edit to
the sealed dataset.

## Plan

[../plans/2026-10-30-vla-inputs-s0-signals.md](../plans/2026-10-30-vla-inputs-s0-signals.md).
