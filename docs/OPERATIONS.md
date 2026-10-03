# Operations

How to actually run things **now**. The old `docker-compose` workflow is superseded — see
[archive](archive/docker-compose-pipeline.md).

Every experiment lives in `experiments/<name>/` as a self-contained package with a `contract.py`
(what is bound and verified) and a `runner.py` (what executes). Nothing runs outside a container.

---

## Our Docker images on the RTX machine

The machine is shared; about thirty other images on it belong to other people and projects.
**Only these two are ours. Never remove, retag or prune anything else** (no `docker image prune`,
no `docker system prune`).

| local tag | image id | what it is | runs |
|---|---|---|---|
| `smolvla:rtc-offline` | `sha256:08de1e09…714abc` | the pinned (admission) image, MuJoCo 3.13.0 | everything sealed: collection, training, evaluation |
| `vla-robosuite:1.5.2-mj3.4.0` | `sha256:54d861aa…ccf27e` | the image above + robosuite 1.5.2, MuJoCo 3.4.0 | `experiments/robosuite_scene/`, `experiments/robosuite_sim2sim/` |

Tags are local handles only; commands and contracts use the full digest. Neither image is in a
registry, so losing one means rebuilding it (the second from
`experiments/robosuite_scene/Dockerfile`; the first cannot be rebuilt bit-identically).
`vla-robosuite:1.5.2` (`sha256:3ecd62c5…`, robosuite on MuJoCo 3.13.0, could not reset an
environment) was removed on 2026-10-02. Stopped containers named `xvla-*`, `smolvla-*`, `franka-*`,
`local-varied-*` and `sim2sim-*` are ours and hold run logs.

## The pinned image

```
sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
```

Always referenced by digest, never by tag. Python 3.10, LeRobot 0.4.4, PyTorch 2.7.1+cu128,
MuJoCo 3.13.0.

## The second image (robosuite only)

```
sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
```

Local tag `vla-robosuite:1.5.2-mj3.4.0`; the value of record is `IMAGE` in
`experiments/robosuite_scene/contract.py`. Admission image plus robosuite 1.5.2 and
**MuJoCo 3.4.0** (numpy stays 2.2.6). Only `experiments/robosuite_scene/` and the simulating parts
of `experiments/b3_robosuite_data/` (hash-bound; its tests that import no simulator run in the
admission image) run in it; everything
else, including all sealed packages and three of that package's own unit tests (`test_r1_image`,
`test_seal`, `test_expert_fork`), runs in the pinned image above on MuJoCo 3.13.0.

- Why a second MuJoCo: robosuite 1.5.2 imports on 3.13.0 but cannot reset an environment there
  ([FAILURES.md](FAILURES.md) #39). Results from the two images are not physics-comparable.
- Why `--no-deps`: robosuite depends on `mink`, which requires numpy<2. `pip check` in this image
  lists `mink`, `opencv-python`, `pytest` and `qpsolvers` as missing; that is intended.
- Build with the classic builder, `DOCKER_BUILDKIT=0 docker build -f experiments/robosuite_scene/Dockerfile ...`
  — BuildKit tries to pull the bare `sha256:` base reference from a registry. Never rebuild or
  retag the admission image.
- Scratch size: a collection that reopens a LeRobot dataset builds a Hugging Face arrow cache under
  `HF_HOME` each time (31 MB at 541 episodes). With `HF_HOME=/tmp/hf` on a tmpfs, give `/tmp` at
  least the sum over all reopens (the B3 run needed more than 8 GB; `--tmpfs /tmp:rw,size=24g`
  finished it). FAILURES #41.
- Open files: LeRobot's video decoder cache holds one descriptor per distinct video file for the
  life of the process, and a container's default `ulimit -n` is 1024. A dataset with one mp4 per
  episode per camera needs 2 x episodes descriptors for random sampling (B3: 1,120). Add
  `--ulimit nofile=65536:65536` to any training or random-access run on such a dataset. FAILURES #42.
- Run helper, test order and collector: `experiments/robosuite_scene/README.md`. Its tests are run
  by module name (`-m unittest experiments.robosuite_scene.tests.<module>`), three of them in the
  admission image.

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

**Three `outputs/implementation/` directories must be mounted `:rw`** or several packages report
false failures with `OSError: Read-only file system`:

```
-v "$PWD/outputs/implementation/broad_workspace_data_20260920:/vla/outputs/implementation/broad_workspace_data_20260920:rw"
-v "$PWD/outputs/implementation/local_position_probe_20260919:/vla/outputs/implementation/local_position_probe_20260919:rw"
-v "$PWD/outputs/implementation/local_varied_data_20260919:/vla/outputs/implementation/local_varied_data_20260919:rw"
-v "$PWD/outputs/implementation/b1_language_data_20260925:/vla/outputs/implementation/b1_language_data_20260925:rw"
```

Without them, `broad_workspace_data`, `local_varied_data`, `local_position_probe` and `b1_language_data`
each report errors that are pure harness artefacts. Verified: all four go green once mounted.

### Known pre-existing failures (not merge blockers)

Five packages fail on `main` as well as on every feature branch, all downstream of the deleted
pilot manifest ([FAILURES.md](FAILURES.md) #2) and the later removal of preflight outputs:

| package | typical errors | cause |
|---|---|---|
| `fixed_layout` | 13 | `datasets/franka_pilot_20260915/meta/info.json` missing |
| `single_task` | 5 | same |
| `xvla_single_task` | 4 | same, plus deleted `*_preflight/` checkpoints |
| `smolvla_fixed_layout` | 1 | `Fixed provenance/semantic proof changed` |
| `smolvla_local_varied` | 1 | same provenance family |

Before treating any of these as a regression, run the same package in a worktree at `main` and
compare counts — a branch that *reduces* them has not broken anything.

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

There is no resume for the older packages: `--train` starts from the base bundle and a failed run
restarts from zero. `xvla_b3_robosuite` (v2) is the exception: it snapshots model, Adam and RNG every
2,600 updates and has `--resume-proof` and `--train --resume`. Procedure after a failure:

1. Read `failure.json` (written on any Python exception) and `docker logs <container>`. A hard kill
   (OOM, host reboot) leaves neither; `docker inspect` shows `OOMKilled`, and `result.json` shows
   where it stopped. Keep the dead container; do not `--rm`.
2. Fix the *environment* (flags, memory, disk). Changing a `SOURCE_FILES` member makes it a different
   run and `--resume` will refuse.
3. Relaunch the same `--train` command with `--resume` added, same mounts, same output directory
   (the claim token `run_claim.json` is how it recognises its own). Check at 5 and 15 minutes: the
   log prints `SNAPSHOT` every 2,600 updates and `result.json['resumes']` lists each resume.
4. Everything past the snapshot lands in `quarantine/resume_NN/`; it is evidence, not clutter.

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
| training run | 27–37 GB (checkpoints ~3.3 GB; final ~9.9 GB with optimizer state); B3 v2 adds up to ~32 GB of resume snapshots (three of ~10.5 GB while one is being written) |
| full evaluation | ~150 MB |
| sealed 200-episode dataset | ~18 GB |

Checkpoints cannot be trimmed after the fact: `verify_checkpoints` requires every checkpoint
listed in a report to exist. Control cost via the checkpoint *schedule* before launching.

`outputs/downloads_*` are hardlinks to files under `outputs/train/` — deleting them frees nothing.

**Disk is tight.** As of 2026-09-26: 879 GB total, ~13 GB free (99% full). Check `df -h .` before
any new collection or training launch; the checkpoint schedule alone can need 20-30 GB. See
Storage census below for what is safe to reclaim.

## Current outputs/ census (2026-09-26)

Sealed, results-worthy datasets (all under `outputs/`, each is the *collection root* — the
HF-Hub-ready LeRobot payload is the `dataset/` subdirectory only, ~1.1 GB; `attempts/`, `rejected/`,
`resets/` are raw per-episode audit evidence, ~15-17 GB, not needed for reuse):

| dataset | root | seal (fingerprint) | episodes | headline result |
|---|---|---|---|---|
| `broad_workspace_data` | `broad_workspace_data/broad_workspace_20260920/` | `0351d3da...a601a85` | 200 | X-VLA 86.1% (62/72), matched exposure |
| `local_varied_data` | `local_varied_data/local_varied_20260919/` | `ae8828ea...25dcf52c` | 60 | 2.8% (2/72) on the broad holdout — superseded, generalises poorly |
| `b1_language_data` | `b1_language_data/b1_language_20260925/` | `b152cba2...4837fdf22` | 200 | X-VLA 94.4% (151/160), no positional shortcut — see RESULTS.md |
| `fixed_layout` | `fixed_layout/fixed_layout_20260918/` | `403b8ec2...8d8fba5` | 24 | Tier 2 baseline, 100% at base pose — superseded, single fixed layout |
| `b3_robosuite_data` (2026-10-03) | `b3_robosuite_data/b3_robosuite_20261003/` | `e14af864...92b8e6c0` | 560 | not yet trained on; robosuite scene (MuJoCo 3.4.0), 2.9 GB, `episodes/` holds per-episode traces, `bank.npz` the 100 reserved scenes — see [B3_ROBOSUITE_COLLECTION_20261003.md](B3_ROBOSUITE_COLLECTION_20261003.md) |

Trained models worth keeping (`outputs/train/`; checkpoint dirs `step_NNNNNN/N`, only the final one
per run carries `training_state.pt`, the ~7 GB optimizer state, not needed for inference):

| run | winning checkpoint | result |
|---|---|---|
| `xvla_franka_stacking_broadworkspace_long_v1_20260923` | `step_016700` | 86.1% broad holdout |
| `xvla_franka_stacking_b1_language_v1_20260925` | `step_033400` | 94.4% report, checkpoints [1,8000,16700,25000,33400] all kept for the ablation curve |

Everything else under `outputs/train/` (smolvla_pilot, single_task, fixed_layout, the plain
`broadworkspace_v1`/`localvaried_v1` runs) is superseded Tier 2-4 work — see
[BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for the chronology. `outputs/downloads_20260920/` and
`outputs/downloads_20260926/` are pre-staged, hardlinked download bundles for the user's Windows
machine — the 2026-09-26 one covers both the minimal tier and the full checkpoint-ablation tier
for B1 and broad-workspace (~50 GB, zero extra disk here); read the `README.md` inside each
before touching originals. `outputs/downloads_20261001/b2_distractor/` was copied to Windows on
2026-10-01; all six `model.safetensors` matched `model_sha256` and `seal.json` is
`b6cfff3f...0c12340`. The Windows check did not re-derive the dataset fingerprint. Do not delete
the RTX originals without a separate go-ahead. Full file lists and rationale:
[B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md](B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md).

---

## What must never be deleted or edited

- `datasets/` — deleting the pilot manifest permanently broke `local_varied` provenance
- any sealed dataset under `outputs/`
- any `*_preflight/result.json` whose run report records `scratch_result_sha256`
- any file in a contract's `SOURCE_FILES`, or reachable from `provenance()`, **while a run is live**

`provenance()` includes package READMEs. See [FAILURES.md](FAILURES.md) #8, #17.
