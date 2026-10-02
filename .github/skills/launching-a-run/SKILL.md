---
name: launching-a-run
description: "Use when launching a training run, admission check, preflight, or evaluation in this repo — covers the --check/--preflight/--train gate sequence, container flags, required mounts, and the pre-launch checklist."
---

# Launching a run

Long runs here cost hours and cannot resume. Every item below has cost real GPU time at least once.

## 1. Pre-launch checklist

This is the only copy of the checklist; `TODO.md`, `CLAUDE.md` and `copilot-instructions.md` link
here. Numbers in brackets are entries in [docs/FAILURES.md](../../../docs/FAILURES.md).

Do **all** of these before the first container starts. Stop at the first failure.

The package:

1. **Package README updated** — it is in `SOURCE_FILES`; a copied one describing the old recipe is
   hashed into the run's provenance, and editing it later invalidates the run (#17).
2. **No duplicated constants** — grep the runner for every literal that also lives in `recipe()`
   (update horizon, checkpoint schedule, output paths, purpose strings) (#15).
3. **Copied-package residue** — if the package was forked, grep for the source package's name in
   imports, test paths and docstrings. Copied tests keep importing the source module and pass
   while testing the wrong code; derive constants from the contract, do not copy the literal (#20).
4. **Regression tests fail against the unfixed code** — confirm with `git stash`, not by
   assumption (#10, #15).
5. **Tests pass in-container** for the package and its eval package.
6. **Shared helpers are untouched** — `xvla_single_task/contract.py` is bound by four contracts;
   override locally instead of editing it.

The design:

7. **Checkpoint schedule is final, and chosen from the LR curve, not round numbers** — checkpoints
   cannot be deleted afterwards without breaking `verify_checkpoints`, and two runs can only be
   compared mid-training at the same *schedule phase*, not the same step count (#22).
8. **Rollout evaluation is budgeted** — loss and MAE improved monotonically through a run that
   scored 0/72; only rollouts measure capability (#21).

The machine:

9. **Right image** — the admission image for everything sealed; `experiments/robosuite_*` runs in
   the second image (see `docs/OPERATIONS.md`).
10. **Worktree is current** — `git log --oneline -1` inside the directory being mounted. A stale
    worktree has silently run old code (#18).
11. **Mounts exist** — `outputs`, `datasets`, `.cache` are gitignored and absent from fresh
    worktrees.
12. **Disk headroom** — a training run is 27–37 GB, an evaluation ~150 MB. `df -h .`
13. **Nothing else is holding the GPU** — `nvidia-smi`. Other people use this machine.

While it runs:

14. **Nothing hash-bound is edited** — check `SOURCE_FILES` and `provenance()` membership first;
    `provenance()` includes READMEs (#8, #17).
15. **Gates in order** (section 2) and **poll actively** (section 4) (#16, #19).

## 2. Gate sequence

Always in order; each needs its own output root.

```bash
OUT=outputs/train/<run_name>
mkdir -p ${OUT}_admission ${OUT}_preflight ${OUT}
```

| gate | mount `:rw` | proves | cost |
|---|---|---|---|
| `--check` | `${OUT}_admission` | dataset audit, fresh statistics, CPU processor checks | ~1h CPU |
| `--preflight` | `${OUT}_preflight` | 2 GPU updates, FP32 weight + Adam round-trip | ~5 min |
| `--train` | `${OUT}` | the run | hours |

**Do not edit any `SOURCE_FILES` member between `--check` and `--train`.** The admission is
invalidated and must be re-run in full.

## 3. Container invocation

CPU gates:

```bash
docker run -d --name <run>-check --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=8g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla \
  -v "$PWD:/vla:ro" -v "$PWD/outputs/<dataset>:/vla/outputs/<dataset>:ro" \
  -v "$PWD/${OUT}_admission:/vla/${OUT}_admission:rw" \
  -w /vla --entrypoint python \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m experiments.<pkg>.runner --check
```

GPU gates: swap to `--runtime nvidia --gpus all`, add `-e CUBLAS_WORKSPACE_CONFIG=:4096:8`, drop
`NVIDIA_VISIBLE_DEVICES=void`, and mount `.cache` read-only for the base model bundle.

Omitting `HF_HOME`/`XDG_CACHE_HOME` under `--read-only` → `OSError: Read-only file system`.

## 4. Monitor actively

`docker wait` has silently failed to notify here. Poll:

```bash
docker ps -a --filter name=<run> --format '{{.Status}}'
docker logs <run> 2>&1 | grep -E '^TRAIN|^CHECKPOINT|Traceback' | tail -3
```

Check within the first few minutes — most failures surface immediately, and an unattended dead
container has burned 47 minutes. Verify the first checkpoint lands on schedule; if the update
counter stops short of `recipe()['max_optimizer_updates']`, the runner is out of sync with the
contract.

## 5. On completion

Read `result.json` / the run report and confirm the final update count matches the contract before
claiming the run finished.
