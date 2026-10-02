# robosuite_scene

Bring-up of a scene composed by robosuite 1.5.2 and driven by this repo's scripted expert
(TODO B3, Phase 1). One layout, red cube stacked on green, the two existing cameras, the existing
9-state / 8-action LeRobot schema. Result and open issues:
[docs/ROBOSUITE_SCENE_BRINGUP_20261002.md](../../docs/ROBOSUITE_SCENE_BRINGUP_20261002.md).

**This package is not hash-bound to any sealed run.** No contract lists it in `SOURCE_FILES` and it
has no `provenance()`. It imports the sealed modules read-only and must not edit them.

## Image

The package runs in its own image, `contract.IMAGE` (built from `Dockerfile` on top of the
admission image), not in the admission image:

- robosuite 1.5.2 cannot reset an environment on MuJoCo 3.13.0, so this image carries MuJoCo 3.4.0.
  The admission image and everything sealed stay on 3.13.0.
- robosuite and mujoco are installed with `--no-deps` so that `mink` (numpy<2) does not downgrade
  numpy 2.2.6.

Results from this scene are therefore not physics-comparable with sealed results. Three test
modules use no robosuite and run in the admission image: `test_r1_image`, `test_seal`,
`test_expert_fork`.

## Running

From the repository root (or the worktree root, with `MAIN` pointing at the main checkout):

```bash
MAIN=$PWD
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=$(sed -n 's/^IMAGE = "\(.*\)"/\1/p' experiments/robosuite_scene/contract.py)
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

Tests are run by module name, one robosuite module per process:

```bash
run "$OLD" -m unittest experiments.robosuite_scene.tests.test_r1_image \
  experiments.robosuite_scene.tests.test_seal experiments.robosuite_scene.tests.test_expert_fork -v
for t in test_r2_render test_model_contract test_r7_cameras test_r4_schema test_r5_reset test_r3_stack test_r6_drift; do
  run "$NEW" -m unittest experiments.robosuite_scene.tests.$t -v || break; done
```

## Gate order

| step | gate | test |
|---|---|---|
| 1 | R1 image, sealed paths untouched | `test_r1_image` (admission image) |
| 2 | R2 headless render, OpenCV orientation | `test_r2_render` |
| 3 | model contract (names, actuators, finger sign, pads) | `test_model_contract` |
| 4 | R7 camera poses | `test_r7_cameras` |
| 5 | R4 schema | `test_r4_schema` |
| 6 | R5 deterministic reset | `test_r5_reset`, then two `--trace-only` runs with equal sha256 |
| 7 | R3 one scripted stack | `test_r3_stack`, then two `--expert-trace` runs with equal sha256 |
| 8 | R6 replay and URDF agreement | `test_r6_drift` |
| 9 | collect and seal | `record --collect` |

## Recording

```bash
run "$NEW" -m experiments.robosuite_scene.record --trace-only   --seed 0 --out /out/<new dir>
run "$NEW" -m experiments.robosuite_scene.record --expert-trace --seed 0 --out /out/<new dir>
run "$NEW" -m experiments.robosuite_scene.record --collect             --out /out/<new dir>
```

`--out` must not exist. `--collect` refuses to start unless the running MuJoCo is
`contract.MUJOCO_VERSION`, runs seeds `0..EPISODE_COUNT-1`, and writes `dataset/`,
`trace_seed_0.npz`, `episodes.json` and `seal.json`. A failed seed aborts without a seal; there is
no retry or resume. `validate.fingerprint(root)` recomputes the seal's fingerprint.

## Files

| file | content |
|---|---|
| `contract.py` | image ids, versions, cube layout, grasp height, episode count |
| `scene.py`, `arena.py`, `cameras.py` | robosuite composition, `build_sim`, `overwrite_reset` |
| `xml_contract.py` | `apply_scripted_contract`: what is rewritten to this repo's robot contract |
| `expert.py` | `pose_snapshot_v2` expert, `run` copied with the grasp height made a name |
| `validate.py` | frame schema, replay checks, URDF agreement, seal fingerprint |
| `compare_cameras.py` | side-by-side PNGs of the current and robosuite scenes |

Never call `env.step()`: the expert writes `data.ctrl` and steps MuJoCo directly. Render only with
`mujoco.Renderer`; a second OSMesa context in the process corrupts frames.
