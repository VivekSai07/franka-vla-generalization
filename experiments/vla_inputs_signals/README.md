# S0 — extra-signal sidecar for the sealed P2 episodes

Per-frame wrist wrench, finger contact forces and target/prop bounding boxes for all 540 sealed P2 episodes
(`outputs/props_p2_data/props_p2_20261013`, seal `b1d642b3…f4d898`), recorded by exact action replay, plus the
`signals.py` functions the M0 evaluator reuses on live rollouts. Design:
`docs/superpowers/specs/2026-10-30-vla-inputs-s0-signals-design.md`. Plan:
`docs/superpowers/plans/2026-10-30-vla-inputs-s0-signals.md`.

Robosuite image only (`contract.IMAGE`, MuJoCo 3.4.0). CPU only. Hash-bound after the seal: **fork, do not edit**.

## Modules

| file | role |
|---|---|
| `replay.py` | replays an episode from its sealed initial state and sealed actions; every tick's `qpos` and prop poses must be byte-equal to the sealed ones, and so must `final_qpos` (`ReplayMismatch` otherwise); rebuilds the phase index |
| `signals.py` | `wrist_wrench`, `finger_force`, `box`, `target_box`, `props_box`, computed at the observation instant, never writing the live `MjData` |
| `sidecar.py` | per-shard builder, merge, report, seal, verify (CLI below) |
| `contract.py` | `provenance()`: this package, every file of P2's sealed provenance, `check_p2_frozen()` |

## Sidecar arrays (per episode, 452 frames, float64 unless noted)

`wrist_wrench` (6: force N, torque N·m, hand frame `robot0_right_hand`, at the hand origin, MuJoCo force/torque-sensor
convention, gravity of everything below the wrist included), `finger_force` (2: left, right; sum of normal contact forces
against anything outside the robot), `target_box` (2 × 5: front_cam, gripper_cam; `u_min, v_min, u_max, v_max` / 512, then
`in_frame`), `props_box` (n × 2 × 5, same for every prop present; recorded, not used by M3), `prop_names`, `segment`
(int8, see below), `phase_index` (rebuilt from the simulation clock, checked against the sealed gripper command),
`frame_index`, `episode_index`, `object`, `split`.

`in_frame` is projected geometric validity, not visibility: occlusion is ignored. Boxes use the geoms the renderer draws
(groups 0–2), clipped at the near plane before the perspective division; primitives use a documented loose local-cube
bound, the prop meshes are exact.

Segments, inferred from the sealed actions and the target's height (the sealed NPZ has no phase index):
`0 before_close` (gripper command open before the first close), `1 closing` (closed, target not yet 2 cm above its start),
`2 closed_and_lifted` (from the first tick the target is 2 cm above its start to the last closed tick), `3 after_release`.

## Gates

G1 replay exact (no tolerance, inside every replay), G2 non-perturbation, G3 wrench equals MuJoCo's own force/torque
sensors to 1e-9, G4 box containment against a segmentation render, G5 seal re-verified in a second process.
A failed gate stops S0; nothing is relaxed.

## Run (robosuite image, from the worktree root, P2 root mounted read-only)

```bash
docker run --rm --runtime runc --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,size=8g -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa -e PYTHONPATH=/vla \
  -v "$PWD:/vla:ro" -v "<main checkout>/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" \
  -v "<out dir>:/out:rw" -w /vla --entrypoint python sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e \
  -m experiments.vla_inputs_signals.sidecar --build K --shards 6 --work /out/shards
# then: --merge --work /out/shards --out /out/p2_signals_<date> ; --report OUT ; --seal OUT ; --verify OUT
```
