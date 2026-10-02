# robosuite_sim2sim

Rolls a policy trained on the MuJoCo scene out on the robosuite scene, with the MuJoCo scene as a
paired control. First use and result: [docs/ROBOSUITE_SIM2SIM_20261002.md](../../docs/ROBOSUITE_SIM2SIM_20261002.md)
(B2 step 50,000: 19/20 on MuJoCo, 3/20 on robosuite).

Not hash-bound: no contract, no `provenance()`. Say so next to any number from it.

## Two stages, two images

| stage | module | image | why there |
|---|---|---|---|
| A | `starts.py` | admission (MuJoCo 3.13.0), CPU | re-derives B2's sealed report scenes and proves them against `bank.json`; that only reproduces on the MuJoCo the bank was sealed with |
| B | `rollout.py` | robosuite (MuJoCo 3.4.0), GPU | runs both arms with one loop; needs robosuite |

Stage A writes `starts.npz` (qpos, qvel, ctrl per trial) and `trials.json`; stage B reads them.

## Running (from a worktree root)

Worktrees lack the gitignored directories. Create the empty mount points once, then mount the main
checkout's real ones onto them:

```bash
MAIN=/home/utxowqr/smolvla_franka
B2D=outputs/b2_distractor_data/b2_distractor_20260928
B2T=outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928
mkdir -p $B2D $B2T datasets .cache $MAIN/outputs/robosuite_sim2sim
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=$(sed -n 's/^IMAGE = "\(.*\)"/\1/p' experiments/robosuite_scene/contract.py)
COMMON="--network none --read-only --user $(id -u):$(id -g) --tmpfs /tmp:rw,size=8g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/vla -e NUMBA_CACHE_DIR=/tmp/numba \
  -v $PWD:/vla:ro -v $MAIN/$B2T:/vla/$B2T:ro -v $MAIN/outputs/robosuite_sim2sim:/out:rw -w /vla --entrypoint python"

# A: 20 starts, about a minute
docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void $COMMON \
  -v $MAIN/$B2D:/vla/$B2D:ro -v $MAIN/datasets:/vla/datasets:ro \
  $OLD -m experiments.robosuite_sim2sim.starts --out /out/<run>

# B: about 20 min for the MuJoCo arm, up to 70 for a failing robosuite arm. Smoke first: --limit 1
docker run -d --name sim2sim-<run> --runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 $COMMON \
  -v $MAIN/.cache:/vla/.cache:ro \
  $NEW -m experiments.robosuite_sim2sim.rollout --run /out/<run>
docker logs sim2sim-<run> 2>&1 | grep '^SIM2SIM'      # poll; do not rely on docker wait
```

`rollout.py` refuses to run on any MuJoCo other than `robosuite_scene.contract.MUJOCO_VERSION`,
and refuses an existing `rollouts/` directory. `--scenes` picks an arm, `--limit N` the first N
trials.

## Reading the result

`result.json` has the per-arm counts, the paired table and how many MuJoCo-arm trials agree with
B2's stored report (the harness check). `analysis/classify.py <run>` labels each trial by first
cause from its trace; look at `cameras.mp4` before trusting a label.

A success count from one arm alone means nothing: the robosuite arm is only interpretable next to
the control.
