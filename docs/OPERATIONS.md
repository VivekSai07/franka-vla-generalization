# Operations

How to actually run things **now**. The old `docker-compose` workflow is superseded — see
[archive](archive/docker-compose-pipeline.md).

Every experiment lives in `experiments/<name>/` as a self-contained package with a `contract.py`
(what is bound and verified) and a `runner.py` (what executes). Nothing runs outside a container.

---

## The pinned image

```
sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
```

Always referenced by digest, never by tag. Python 3.10, LeRobot 0.4.4, PyTorch 2.7.1+cu128,
MuJoCo 3.13.0.

## Standard container flags

CPU work (tests, admission, collection):

```
--runtime runc --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=8g
-e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
-e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1
-e NVIDIA_VISIBLE_DEVICES=void
```

GPU work (preflight, training, evaluation) — same, but:

```
--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8
```
and drop `NVIDIA_VISIBLE_DEVICES=void`.

Omitting `HF_HOME`/`XDG_CACHE_HOME` under `--read-only` causes
`OSError: Read-only file system: '/.cache'`. The `--tmpfs` must be at `/tmp`, not a subpath.

## Required mounts

The repo mounts read-only at `/vla`. These directories are gitignored, so they are absent from a
fresh `git worktree` and must be mounted explicitly when used:

| path | why |
|---|---|
| `outputs` | datasets, training runs, evaluations |
| `datasets` | pilot source episode + manifest |
| `.cache` | X-VLA / SmolVLA base model bundles |

Only the run's own output root is mounted `:rw`. Everything else is `:ro`.

---

## Running tests

Test directories have no `__init__.py` and import siblings directly, so the tests directory must
be the top-level dir with the repo on `PYTHONPATH`:

```
-e PYTHONPATH=/vla ... -m unittest discover \
  -s experiments/<pkg>/tests -t experiments/<pkg>/tests
```

Using `-t .` fails with `Start directory is not importable`.

---

## The three training gates

Always in order. Each writes an exclusively-claimed output root; none can overwrite or resume.

| gate | what it proves | cost |
|---|---|---|
| `--check` | full semantic audit of the dataset, fresh train-only statistics, CPU processor checks | ~1h CPU |
| `--preflight` | 2 GPU updates, exact FP32 weight and Adam round-trip | ~5 min GPU |
| `--train` | the real run from the fresh base | hours |

```bash
mkdir -p ${OUT}_admission ${OUT}_preflight ${OUT}
# --check   → mount ${OUT}_admission :rw
# --preflight → mount ${OUT}_preflight :rw
# --train   → mount ${OUT} :rw
```

**Editing any `SOURCE_FILES` member after `--check` invalidates it** and forces a full re-run.
Finish code changes before starting the sequence.

There is no resume. `--train` always starts from the base bundle; a failed run restarts from zero.

---

## Evaluation

```bash
-m experiments.<pkg>_eval.evaluate --mode=both        # or broad_holdout / fixed_parity
-m experiments.<pkg>_eval.evaluate --checkpoint=3000  # earlier checkpoints, for curves
```

The final checkpoint writes to `evaluation/`; earlier ones to `evaluation_step_NNNNNN/`, so
curve points never collide with a sealed result.

Cost: a 92-trial evaluation is ~150 MB and 1–3h. Timeouts are slower than successes, because a
failure burns the full 1200-tick cap while a success exits early on the stability streak — a slow
evaluation is itself a signal that the policy is failing.

---

## Monitoring

`docker wait` has silently failed to notify in this environment. Poll:

```bash
docker ps -a --filter name=<container> --format '{{.Status}}'
docker logs <container> 2>&1 | grep -E '^TRAIN|^EVAL_|^CHECKPOINT' | tail -2
```

Do not treat "launched" as "will report". A run has failed silently for 47 minutes this way.

---

## Storage

| item | cost |
|---|---|
| training run | 27–37 GB (checkpoints ~3.3 GB; final ~9.9 GB with optimizer state) |
| full evaluation | ~150 MB |
| sealed 200-episode dataset | ~18 GB |

Checkpoints cannot be trimmed after the fact: `verify_checkpoints` requires every checkpoint
listed in a report to exist. Control cost via the checkpoint *schedule* before launching.

`outputs/downloads_*` are hardlinks to files under `outputs/train/` — deleting them frees nothing.

---

## What must never be deleted or edited

- `datasets/` — deleting the pilot manifest permanently broke `local_varied` provenance
- any sealed dataset under `outputs/`
- any `*_preflight/result.json` whose run report records `scratch_result_sha256`
- any file in a contract's `SOURCE_FILES`, or reachable from `provenance()`, **while a run is live**

`provenance()` includes package READMEs. See [FAILURES.md](FAILURES.md) #8, #17.
