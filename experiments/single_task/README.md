# Single-task capability experiment

Implementation only, **not launched**. Read [PLAN.md](PLAN.md) for the exact approved recipe. No old source, data, checkpoint, credential or bound-source list is modified. No commits.

## Files and behavior

- [contract.py](contract.py): manifest membership, frame boundaries, allowed observations, 500-warmup/5000-cosine schedule, exclusive outputs, full source binding, separate seen/held-out counts and robust failure record.
- [runner.py](runner.py): explicit fresh trainer reusing pilot helpers; micro8/accum4, FP32 trainables/AdamW and BF16 autocast, stock loss, chunk50/horizon10. No old top-level trainer, dynamic rewrite, monkeypatch, resume, autotuning or best-checkpoint selection.
- [evaluate.py](evaluate.py): final005000 report/marker/config/processor/stats/cache binding; verified generic reset adapter; unchanged baseline `run_trial` and `restore_scene`.
- [tests/test_contract.py](tests/test_contract.py), [tests/test_runner.py](tests/test_runner.py), [tests/test_data.py](tests/test_data.py): synthetic contracts plus actual read-only data/reset integration.

Train IDs0..8 only: **4752 frames,4311 complete starts**. Validation IDs300..303 only. Both are red-on-green with two cubes. Own statistics use every training frame, never validation. Probes use current observations at0,30,60,107,347 and future actions only as comparison targets; their results never change training or choose a checkpoint.

Full run saves updates1,1000,2000,3000,4000,5000. Every save uses the unchanged exact-FP32 reload/action-agreement helper. Scratch saves1,2 and stops after two optimizer updates (same physical batch and accumulation). It performs offline probes but **no closed-loop trials**. A full run requires its matching completed scratch report but starts afresh from the cached pinned base, not scratch weights.

The new runner additionally enforces **absolute reload action error <=1e-5** immediately after every save, before accepting its marker, and rechecks both scratch markers on full-run admission. The old helper's relative tolerance is unchanged; its returned marker alone is not new-run acceptance. Scratch `asset_files` must exactly equal the current entire cache inventory (policy, VLM and manifest bytes, including added/removed files), not just pinned revision labels.

Evaluation only uses005000, frozen saved processors, and exactly13 original saved initializations. Adapter evidence records original/derived SHA256 and verifies full integration state, first qpos, scene digest and geometry. Derived NPZ contains only initial_state and qpos, under the new evaluation output. **Actual baseline ordinal is episode_id+1**, so baseline20260916+ordinal equals20260917+episode_id; held-out ordinals are301..304, not9..12. Baseline limits remain1200ticks/40simulated seconds and30 consecutive released-stack ticks.

## Exact CLI — for main review, not executed here

Run from the repository root. The container mounts source/data/model cache read-only, permits writes only in the training-output parent, has no network, reads no credential environment file, and does not upload. Both destination directories must be absent; existing/failed runs are preserved and rejected, never overwritten or resumed. Do not redirect a log into a destination before its exclusive creation.

First, scratch only:

```bash
docker run --rm --gpus all --network none --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$PWD:/vla:ro" -v "$PWD/outputs/train:/vla/outputs/train:rw" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -m experiments.single_task.runner --preflight
```

After successful scratch and main review, one fresh full run with automatic final-only evaluation:

```bash
docker run --rm --gpus all --network none --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$PWD:/vla:ro" -v "$PWD/outputs/train:/vla/outputs/train:rw" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -m experiments.single_task.runner
```

Fixed container outputs:
- Scratch: `/vla/outputs/train/smolvla_single_task_20260917_preflight`
- Full: `/vla/outputs/train/smolvla_single_task_20260917`
- Training result remains immutable after completion; evaluation has a separate result under `evaluation/`, plus per-episode traces/videos and initialization provenance. Evaluation failure does not relabel or overwrite successful training.

There is intentionally no independent eval/resume/retry CLI. No extra trial or automatic expansion follows failure/success. GPU OOM, nonfinite values, exhausted disk budget, stale binding, or any other error stops the run. A failed checkpoint remains incomplete. Failure JSON is written independently before attempting a possibly nonfinite partial report; SIGKILL/OOM-kill/power loss cannot be caught by Python.

## CPU verification

Fresh pinned Docker CPU suites: **168 root +20 adaptive-placement +31 single-task =219 tests** (previous212 +7 review-gap tests), no failures/errors/skips. Fail-first regression run:5 test methods,1 passing control,4 failing methods with21 failed subtests,0 errors/skips. Final coverage includes same-revision cache byte/add/remove changes, missing/stale asset reports, both scratch reload markers, the inclusive absolute boundary and all8 scratch/full checkpoint positions. A synthetic gripper difference passes the legacy relative comparison but fails the new absolute gate. Training orchestration uses CPU stubs, not actual pretrained models, optimizers or checkpoint writes. New integration tests pass the original full `require_replay_pass`, verify real row membership and train-only statistics, decode actual training/validation observations, and exactly restore all13 saved resets without stepping. Stubbed evaluation tests exercise13-call counting/seed identities/immutability; printed synthetic successes are **not policy results**. The legacy root suite includes its existing small synthetic/CPU simulation tests; no new actual rollout or numerical route job was run.

Tested Python hashes and measured counts: [CPU_VERIFICATION.json](CPU_VERIFICATION.json).

```bash
docker run --rm --runtime runc --network none --memory 16g --cpus 4 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e MUJOCO_GL=osmesa \
  -e HF_HOME=/tmp/huggingface -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/vla:ro" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -c 'import unittest
for path,n in [("tests",168),("experiments/adaptive_placement_v4/tests",20),("experiments/single_task/tests",31)]:
 r=unittest.TextTestRunner().run(unittest.TestLoader().discover(path))
 assert r.wasSuccessful() and not r.skipped and r.testsRun==n'
```

Protected tracked training/validation/data_collection/src/tests/experiments SHA256-list digest before/after implementation: `0dce7b33f871759f3817244a440858379f86d2ce5c6fbf3b08bd1bcd0c69d703`. Existing machine-specific untracked files untouched.

## Limitations / interpretation

- GPU model construction,2-update preflight,5000-update training, exact model save/reload and13 actual closed-loop trials remain **unexecuted**. CPU mocks do not establish GPU fit, throughput, learning or capability. No learned result exists yet; both run output directories remain absent.
- Final selected policy must achieve at least8/9 seen and3/4 held-out successes for the **provisional capability** flag. Report denominators separately. Four held-out scenes for one instruction do not establish statistical, language, task or broad layout generalization.
- Full-source/cache/dataset checks run before and after phases. They detect persistent changes, not malicious change-and-revert between checks. Read-only Docker mounts are the primary source/data protection. No tolerances are relaxed.
- Reused baseline trial error handling is unchanged: an exceptional failure before baseline trace serialization can leave a partial video without a complete trace. The new evaluation report/failure record retains the active episode; no automatic replay/retry follows.
- Inline implementation review only; no independent reviewer tool was available. Main review and launch remain pending as requested.