# props_scene

Scene, scripted expert and gate for the props project (P0): robosuite's real-life objects sorted into
two bins by our joint-target IK expert. No policy and no LeRobot dataset yet; the per-tick frames and
actions use the repo's dataset schema (state 9 floats, action 8 floats, `front_cam` / `gripper_cam`
512 x 512) so P1 can record with this expert unchanged.

**Image:** robosuite 1.5.2 / MuJoCo 3.4.0 (`sha256:54d861aa...`) for everything that builds the scene.
`test_constants.py`, `test_success.py` and `test_benchmark.py` import no robosuite and also run in the
admission image. Numbers are not comparable with the sealed MuJoCo 3.13.0 pipeline.

**Not hash-bound.** No contract, no seal, no `provenance()`. It imports sealed code read-only
(`experiments/robosuite_scene`, `experiments/paired_demonstrations`, `experiments/pose_snapshot_v2`,
`data_collection`, `src`) and edits none of it. Give it a contract before a result is reported as final.

## Scene

`scene.build_model(objects, blue_slot) -> (env, model)`, cached per prop set and arrangement. robosuite
composes the table, Panda on the minimal mount, the props and `BinsTableArena`; `env.reset()` runs once,
`xml_contract.apply_props_contract` rewrites the compiled XML to the repo's robot contract, and plain
MuJoCo drives it (`env.step` raises). Two open bins, inner 0.22 x 0.22 m, walls 0.07 m high and
0.01 m thick, floor 0.01 m, centred at (0.32, +/-0.33). `blue` is rgba `0.15 0.35 0.85`, `orange`
`0.95 0.55 0.10`; either can occupy either slot (`blue_slot`). **Left is y > 0, from the robot's point of
view; in `front_cam` +y is image-right**, so the `left` slot shows on the right of the picture.

Props (`prop_<name>` bodies): can, bread, lemon, milk, bottle, cereal, with masses set explicitly
(0.15, 0.05, 0.08, 0.20, 0.20, 0.15 kg) and the cube's contact parameters (`friction 1.2 0.005 0.0001`,
`solref 0.002 1`, `solimp 0.9 0.95 0.001`, `margin 0.001`). `scene.reset(model, poses, home_jitter_deg)`
puts them upright at (x, y, yaw), the arm at `Q_HOME` plus jitter, settles 0.2 s, draws no random numbers.

## Expert

`expert.PropsExpert(objects, blue_slot).run(initial_state, episode, sink, render=True)`. Phases, in
seconds: approach 2.0, descend 1.5, close 1.0, lift 1.5, move 2.5, lower 1.5, open 1.0, retract 1.5,
home 2.5. The hand yaw closes the fingers across the object's narrower horizontal side (round and
square footprints keep the home yaw); the grasp point is `min(0.03, height/2)` below the top; the hand
sits 0.105 m above it. The carry clears the wall top by 0.05 m, the release is with the object's bottom
0.02 m over the wall top, and both targets are measured from the object in the hand. Jitter: home pose
+/-1 deg, targets 5 mm (seeded). The recorded float32 action is the command the simulator receives, so
replaying actions is exact.

## Success

`metrics.in_bin`: the prop's body origin (its centre of mass) is inside the target bin's inner box in x
and y and **below the wall top**, the gripper is open (`ctrl > 127`) and no finger touches it, for 30
consecutive ticks, and it still holds at the last tick of the episode (the end of the home phase). 900
ticks is the cap; an episode is about 450.

`metrics.placement` is a diagnostic only (bottom height, tilt, resting on the bin floor). It never
enters the verdict. See the report for why it exists: a 15 cm box standing on the bin floor has its
centre above the 7 cm wall.

## Gate

```bash
python -m experiments.props_scene.benchmark --out /out/gate_<date> --shard K/6 [--resume]   # 6 containers
python -m experiments.props_scene.benchmark --out /out/gate_<date> --recheck 12            # determinism
python -m experiments.props_scene.benchmark --out /out/gate_<date> --summary
```

300 attempts: 6 objects x 2 target bins x 2 arrangements x 12 seeds = 288, plus 12 in the +y near-base
corner (x 0.22-0.26, y 0.08-0.12). Start pose and yaw, home jitter and the expert seed derive from
`sha256("props_p0_v1:20261005:<identity>")`. Go requires: overall >= 95%, every object >= 90%, both
target slots >= 93%, every replay within 1e-3 of the recorded states and prop poses, and 12 re-executed
attempts byte-equal in a second process. About 45 s an attempt with rendering, so about 40 minutes per
shard on 6 shards. Results are one JSON line per attempt in `attempts_shard<K>of<N>.jsonl`.

## Tests

```bash
run $NEW -m unittest experiments.props_scene.tests.test_scene      # also _geometry, _expert, _replay
run $OLD -m unittest experiments.props_scene.tests.test_constants  # also _success, _benchmark
```

By module name, not `discover`. Check the printed test count: a run started right after a file was
written has printed `Ran 0 tests ... OK`.
