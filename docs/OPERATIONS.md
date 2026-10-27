# Operations

How to actually run things **now**. The old `docker-compose` workflow is superseded — see
[archive](archive/docker-compose-pipeline.md).

Every experiment lives in `experiments/<name>/` as a self-contained package with a `contract.py`
(what is bound and verified) and a `runner.py` (what executes). Nothing runs outside a container.

---

## Download published artifacts

Use the [HF artifact index](HUGGING_FACE_ARTIFACTS.md) for exact dataset/model links,
checkpoint directories and commit-pinned downloads. P1 includes runtime/source and
bundle reconstruction helpers; B3 and earlier archives retain the documented gaps.
Fetching a snapshot does not create the original outputs/ directory layout or satisfy
a source/dataset binding by itself. Follow the linked model card and verify the bundle.

## Our Docker images on the RTX machine

The machine is shared; about thirty other images on it belong to other people and projects.
**Only these two are ours. Never remove, retag or prune anything else** (no `docker image prune`,
no `docker system prune`).

| local tag | image id | what it is | runs |
|---|---|---|---|
| `smolvla:rtc-offline` | `sha256:08de1e09…714abc` | the pinned (admission) image, MuJoCo 3.13.0 | everything sealed: collection, training, evaluation — including B3 **training** (`xvla_b3_robosuite`) and props P1 **training** (`xvla_props_p1`), which need no simulator |
| `vla-robosuite:1.5.2-mj3.4.0` | `sha256:54d861aa…ccf27e` | the image above + robosuite 1.5.2, MuJoCo 3.4.0 | everything that builds the robosuite scene: `robosuite_scene`, `robosuite_sim2sim`, `b3_robosuite_data` collection, `xvla_b3_robosuite_eval` rollouts, `props_scene` (props P0: scene, expert, gate), `props_p1_data` (props P1: sealed instruction-sorting collection), `xvla_props_p1_eval` (props P1 feasibility + rollouts) |

Tags are local handles only; commands and contracts use the full digest. Neither image is in a
registry, so losing one means rebuilding it (the second from
`experiments/robosuite_scene/Dockerfile`; the first cannot be rebuilt bit-identically).
`vla-robosuite:1.5.2` (`sha256:3ecd62c5…`, robosuite on MuJoCo 3.13.0, could not reset an
environment) was removed on 2026-10-02. Stopped containers named `xvla-*`, `smolvla-*`, `franka-*`,
`local-varied-*`, `sim2sim-*`, `b3-*` and `b3v2-*` are ours and hold run logs.

Two environment settings every long robosuite-path run here has needed: a large `/tmp` (16–32 GB;
the HuggingFace arrow cache lands there, FAILURES #41) and, for training on datasets with one video
file per episode, `--ulimit nofile=65536:65536` (FAILURES #42).

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
**MuJoCo 3.4.0** (numpy stays 2.2.6). It runs everything that builds the robosuite scene:
`experiments/robosuite_scene/`, `experiments/robosuite_sim2sim/` (rollout stage), the simulating
parts of `experiments/b3_robosuite_data/` (hash-bound) and the rollouts, scene restore and
feasibility pass of `experiments/xvla_b3_robosuite_eval/`. Everything else — all sealed packages,
B3 training (`experiments/xvla_b3_robosuite/`), and every test that imports no simulator — runs in
the pinned image above on MuJoCo 3.13.0.

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
- Props P0 (`experiments/props_scene/`, no contract, not hash-bound): same flags with `--tmpfs /tmp:rw,size=16g` and
  an output folder mounted at `/out` (`outputs/props_p0/`: `gate_20261005*/` per-attempt JSON lines, `probes/`,
  `contact_sheets/`, `osc_gate/`, `HANDOFF.md`). The gate is six detached containers
  (`python -m experiments.props_scene.benchmark --out /out/<dir> --shard K/6`), about 25 minutes, about 34 s an
  attempt with rendering; poll with `docker ps` and `wc -l`. Never run a parallel job over its models with more
  than about 30 workers: each worker is a few GB resident and 44 exhausted 125 GB (FAILURES #50). OSMesa rendering is
  multi-threaded, so a CPU-heavy probe slows a running gate. Its tests are run by module name and the printed test
  count must be read (`Ran 0 tests … OK` is possible).
- Props P1 (`experiments/props_p1_data/`, sealed 2026-10-07, hash-bound — fork, do not edit): same flags plus
  `--ulimit nofile=65536:65536`, output mounted at `/out` (`outputs/props_p1_data/`:
  `props_p1_<date>/` is the sealed collection — `dataset/`, `episodes/*.npz` per-episode traces,
  `journal.jsonl`, `plan.json`, `provenance.json`, `seal.json`, `validation.json`). Collection is one
  container (`python -m experiments.props_p1_data.record --collect --out /out/<dir>`), true wall-clock
  about 1.1 min/admitted episode (not the 40 s/attempt the gate alone takes — per-episode video encode
  and commit adds about 24 s and is not in the per-attempt timing). `--pilot --slots ... `, `--resume`
  and `--status` are the other modes; never reconstruct the LeRobot dataset object between episodes
  mid-run (FAILURES #41) — only `--resume` reopens it, once.
- Props P2 (`experiments/props_p2_data/`, sealed 2026-10-13, hash-bound — fork, do not edit): **12 parallel
  shard containers**, one per `--collect /out/<run>_shards --shard K` (K = 0..11, 45 contiguous slots each, own
  journal and dataset, resumable alone), `-e LP_NUM_THREADS=4` each (proven pixel-neutral), same flags as P1 and a
  16 GB `/tmp`. ~4 GB RAM per worker (scene models capped at 4). Then `--merge /out/<run>_shards --out /out/<run>`
  aggregates, verifies every episode against its shard, validates and seals (about 1 min). 540 episodes took
  54 min (10.8 h of serial work). `--status /out/<run>_shards` while it runs. Expert re-test: `--gate` per shard
  (unrendered), `--determinism --tag` twice, `--gate-summary`. Report:
  [PROPS_P2_COLLECTION_20261013.md](PROPS_P2_COLLECTION_20261013.md).
- Props P3 (`experiments/props_p3_data/`, sealed 2026-10-20, hash-bound — fork, do not edit): the P2 pattern with **16 shard
  containers** (`--collect /out/<run>_shards --shard K`, K = 0..15, 34 contiguous slots each), each pinned to 3 cores
  (`--cpuset-cpus 3K-3K+2`) with `-e LP_NUM_THREADS=3` (proven pixel-neutral against default threads), 16 GB `/tmp`, same flags.
  **Create `/out/<run>_shards` first** (`--collect` reads its free space before making it; FAILURES #59). ~4 GB per worker.
  544 episodes took 46 min 43 s (12.4 h of serial work: 54.8 s per attempt and 27.0 s per commit under load); merge 60 s.
  Expert re-test: `--gate` per shard, `--determinism --tag` twice (one with the thread cap, one without),
  `--gate-summary`. Report: [PROPS_P3_COLLECTION_20261020.md](PROPS_P3_COLLECTION_20261020.md).
- Props P3 training (`experiments/xvla_props_p3/`, admission image, no simulator; fork of `xvla_props_p2`): the same
  `--check`/`--preflight`/`--resume-proof`/`--equivalence`/`--train [--resume] [--stop-at T]` gates and mounts as P2, with the
  sealed parent `outputs/props_p3_data/props_p3_20261020` and outputs `outputs/train/xvla_props_p3_v1{,_admission,_preflight,
  _resumeproof,_equivalence}`; hash-bound by its own `SOURCE_FILES` from `--check`. Plan:
  [superpowers/plans/2026-10-21-props-p3-training-eval.md](superpowers/plans/2026-10-21-props-p3-training-eval.md).
- Props P3 evaluation (`experiments/xvla_props_p3_eval/`, robosuite image + GPU, not hash-bound): `scripts/xvla_props_p3_eval.sh
  feas|prepare|smoke|all`; run from a worktree it mounts the main checkout's `outputs` and `.cache` (`OUT_ROOT` overrides) and creates
  the (gitignored) bind-mount targets inside the read-only code mount first, or docker refuses to start. Workers
  pinned to 5 cores, peak GPU about 7.3 GiB after a checkpoint switch (as P2). OSMesa only.
- Props P2 evaluation (`experiments/xvla_props_p2_eval/`, robosuite image + GPU, not hash-bound): `evaluate.py prepare`
  once (admission image, mounts `outputs/train`, both parents, `.cache`), then `scripts/xvla_props_p2_eval.sh smoke|all N`
  runs N parallel workers, each pinned to 5 cores (`--cpuset-cpus`, `OMP_NUM_THREADS=5`), ~4 GB GPU at first load and
  ~7.3 GB after a checkpoint switch, so keep N at 8 or below on the 47 GB GPU. A stage is resumable; a dead worker
  leaves missing trial folders and the driver resumes once. 540 rollouts took about 3 h. Feasibility of 120 layouts,
  unrendered, takes under a minute in 12 shards. Report: [PROPS_P2_EVALUATION_20261016.md](PROPS_P2_EVALUATION_20261016.md).
  Archive bundle (hardlinks, 14 GB, re-evaluable and retrainable off RTX): `outputs/downloads_20261017/props_p2/`, README inside.
- Props P1 training (`experiments/xvla_props_p1/`, admission image, no simulator): forked from
  `xvla_b3_robosuite` — same `--check`/`--preflight`/`--resume-proof`/`--train [--resume]` gates, each
  output subdirectory (`_admission`, `_preflight`, `_resumeproof`, the main output) **must be its own
  separate bind mount**, not a subdirectory of one shared parent mount — `claim_resume()`'s
  `is_mountpoint()` check looks for the exact path in `/proc/self/mountinfo`, so a nested subdirectory
  of a single shared mount silently fails resume/`--resume-proof` with "Resume needs the original
  bind-mounted output". Disk guard 65 GiB for `--train` (29.7 GB of checkpoints plus a ~31.5 GB
  three-snapshot peak — a plan estimate of 55 GB missed the transient third snapshot). 126,900 updates
  at 0.848-0.87 s/update, about 30 h uncontended.
- Props P1 evaluation (`experiments/xvla_props_p1_eval/`, robosuite image): `feasibility.py` is CPU
  only (no GPU, ~0.6 s/layout, safe to run alongside anything, including a paused training run) and
  must be run once before any `evaluate.py` rollout. `evaluate.py --mode smoke/screen/report/confirm
  --checkpoint N` additionally needs `.cache/xvla_trial` mounted read-only — the saved checkpoint's
  `policy_preprocessor.json` bakes in an absolute tokenizer path from training time
  (`/vla/.cache/xvla_trial/bundle/tokenizer`), and without that mount `make_pre_post_processors`
  fails with an HF repo-id parse error. **Each rollout process is cheap**: ~4.1-4.3 GB GPU memory and
  ~4.8 CPU cores (inference only, no optimizer/gradients) versus training's ~21 GB — screening or
  reporting all 6 checkpoints in parallel (one container per checkpoint, each writing its own
  `<mode>_step_<N>/` output directory, no file collisions) cut screen from an estimated ~3 h
  sequential to under an hour, confirmed by measurement, not assumed.

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

**Reclaimed 2026-10-17 (user-approved, ~158 GB):** props P2 `resume/` snapshots and the heavy parts of `_preflight/` and
`_resumeproof/` (`checkpoints/`, `resume/`; every `result.json` kept; 70.4 GB); six unused VS Code server versions (~3.7 GB);
and, because the user had copied and verified them on Windows, the bundles `downloads_20260920/` (SmolVLA local-varied),
`downloads_20260926/{b1_language,broadworkspace,smolvla_local_varied}` and `downloads_20261001/b2_distractor`, together with
the checkpoints of `xvla_..._b1_language_v1`, `xvla_..._broadworkspace_long_v1`, `smolvla_..._localvaried_v1` and
`xvla_..._b2_distractors_v1` (83.8 GB; B2's `step_050000/training_state.pt` was kept). Those runs now exist only on Windows and
the Hub, so their repo evaluators' `verify_checkpoints` cannot pass here. Sealed datasets, `datasets/` and the preflight
results were not touched. After the user hash-verified the P2 bundle's two weights on Windows (both matched `model_sha256`), the P1 and P2 intermediate
checkpoints and `training_state.pt` were removed too: each of those runs now keeps only its two bundled checkpoints (P1 88,830 and
126,900; P2 101,520 and 126,900), so `verify_checkpoints` and `evaluate.py prepare` cannot pass in the repo for P1 or P2; re-evaluate
from the bundle with `reevaluate/bundle_evaluate.py`. The removed weights existed in no bundle and no other copy. The same day, on the
user's go-ahead, the rest went too: B3's four unbundled intermediate checkpoints (steps 1, 31,200, 62,400, 93,600; 62,400 and
93,600 had been evaluated in the B3 report, so those weights now exist nowhere), the `training_state.pt` of B2 and B3, and the whole
checkpoints directory of the superseded `xvla_..._localvaried_v1` run (21 GB plus 7 GB). B3 keeps steps 124,800 and 156,000 (bundled);
B2 keeps no checkpoint on RTX. `outputs/` went from 292 GB to 53 GB over the day.

**Reclaimed 2026-10-21 (user-approved, 13 GB; free 50 to 63 GiB):** the four kept weight files of B3 (steps 124,800 and
156,000) and P1 (88,830 and 126,900), all three hardlinks of each (`outputs/train/...`, `downloads_20261005/b3_robosuite/`,
`downloads_20261009/props_p1/` incl. `hf_upload/`). Both are on the Hub and were verified on Windows; re-evaluate from the
Hub download. Run records, results and the bundles' runtime/evidence files stay. The `xvla_base_bundle` is not touched.
The user's HF cache `~/.cache/huggingface` (68 GB) is Qwen models of another project; the UnifoLM entries in it are empty stubs.

**Reclaimed 2026-10-22 (user-approved, ~87 GB; free 31 to 117 GiB):** the Qwen models in the user's HF cache (68 GB, another project, re-downloadable;
the UnifoLM entries were empty stubs); `outputs/egl_rendering_check`; the pre-P2 dataset roots `b1_language_data`, `b2_distractor_data`,
`b3_robosuite_data`, `broad_workspace_data`, `local_varied_data`, `fixed_layout`, `paired_demonstrations`, `pilot_overnight_20260915`; the bundles
`downloads_20260920` and `downloads_20261005`; `publication/`; and the P3 shard audit copy (the sealed P3 root was not touched). Run records,
`datasets/` and the P1/P2 data were kept. Those sealed datasets now exist only on Windows and the Hub, so their packages' verify steps cannot
pass here. Cause of the shortage: `/var/log/syslog` at 134 GB (33 GB/day), root-owned; the user is not in sudoers.
(The syslog was later truncated by someone else.)

**Reclaimed 2026-10-27 (user instruction, ~101 GB; free 84 to 184 GiB):** everything on disk for props P3: `outputs/props_p3_data` (the sealed dataset),
`outputs/train/xvla_props_p3_v1` and its `_admission`, `_preflight`, `_resumeproof`, `_equivalence` directories, `outputs/xvla_props_p3_eval`, the detached run worktree
`.worktrees/p3-train` and the stopped P3 containers. None of it was on Windows or the Hub. Code and docs stay on branch `feat/props-p3`.

**Disk.** As of 2026-10-17: 879 GB total, **248 GB free (71% full)** after the reclaim below (it was 59 GB, 94%; 203 GB before the
P1/P2 intermediates went).
As of 2026-10-05: 879 GB total, ~96 GB free (89% full). Check `df -h .` before
any new collection or training launch; the checkpoint schedule alone can need 20-30 GB. See
Storage census below for what is safe to reclaim.

## Current outputs/ census (updated 2026-10-05)

Sealed, results-worthy datasets (all under `outputs/`, each is the *collection root* — the
HF-Hub-ready LeRobot payload is the `dataset/` subdirectory only, ~1.1 GB; `attempts/`, `rejected/`,
`resets/` are raw per-episode audit evidence, ~15-17 GB, not needed for reuse):

| dataset | root | seal (fingerprint) | episodes | headline result |
|---|---|---|---|---|
| `broad_workspace_data` | `broad_workspace_data/broad_workspace_20260920/` | `0351d3da...a601a85` | 200 | X-VLA 86.1% (62/72), matched exposure |
| `local_varied_data` | `local_varied_data/local_varied_20260919/` | `ae8828ea...25dcf52c` | 60 | 2.8% (2/72) on the broad holdout — superseded, generalises poorly |
| `b1_language_data` | `b1_language_data/b1_language_20260925/` | `b152cba2...4837fdf22` | 200 | X-VLA 94.4% (151/160), no positional shortcut — see RESULTS.md |
| `fixed_layout` | `fixed_layout/fixed_layout_20260918/` | `403b8ec2...8d8fba5` | 24 | Tier 2 baseline, 100% at base pose — superseded, single fixed layout |
| `b3_robosuite_data` (2026-10-03) | `b3_robosuite_data/b3_robosuite_20261003/` | `e14af864...92b8e6c0` | 560 | X-VLA ~85% overall at step 124,800 (88.1% best-of-four on the report); robosuite scene (MuJoCo 3.4.0), 2.9 GB, `episodes/` holds per-episode traces, `bank.npz` the 100 reserved scenes — see [B3_ROBOSUITE_COLLECTION_20261003.md](B3_ROBOSUITE_COLLECTION_20261003.md) |

Trained models worth keeping (`outputs/train/`; checkpoint dirs `step_NNNNNN/N`, only the final one
per run carries `training_state.pt`, the ~7 GB optimizer state, not needed for inference):

| run | winning checkpoint | result |
|---|---|---|
| `xvla_franka_stacking_broadworkspace_long_v1_20260923` | `step_016700` | 86.1% broad holdout |
| `xvla_franka_stacking_b1_language_v1_20260925` | `step_033400` | 94.4% report, checkpoints [1,8000,16700,25000,33400] all kept for the ablation curve |
| `xvla_franka_stacking_b2_distractors_v1_20260928` | `step_050000` | 91.9% report (MuJoCo scene) |
| `xvla_franka_stacking_b3_robosuite_v2` | `step_124800` | robosuite scene, 141/160 best of four (~85%); `step_156000` 105/160 kept for contrast; 46 GB incl. two ~10.5 GB resume snapshots under `resume/` |

B3 superseded runs, kept until you decide otherwise: `…_b3_robosuite_v1_failed_20261003_nofile`
(3.3 GB, died at update ~220), `…_v1_superseded_20261003` (3.3 GB, stopped at ~1,900 for the resumable
v2), and the v1/v2 `_preflight` / `_resumeproof` scratch (~60 GB together). B3 evaluation outputs:
`outputs/xvla_b3_robosuite_eval/` (1.2 GB, all trials with videos).

`outputs/props_p0/` (2026-10-05; props P0, scripted, no dataset and no model): `gate_20261005/` (first gate, 165/300,
before the square-footprint fix), `gate_20261005b/` (164/300, written spec) and `gate_20261005c/` (the amended spec,
283/288, the gate of record) each hold six
`attempts_shard<K>of6.jsonl`, `plan.json`, `recheck.json`, `summary.json`; `probes/` (tuning re-runs, one JSON per
change), `osc_gate/` (OSC comparison), `contact_sheets/`, `scene_check/`, `diag*.py`/`probe.py`/`reach*.py` scratch and
`HANDOFF.md`. A few hundred MB; reproducible from `experiments/props_scene/`, nothing sealed.

`outputs/props_p1_data/props_p1_20261006/` (sealed 2026-10-07, fingerprint `86e3aadb…ac53bcf`): the props P1
instruction-sorting dataset, 540 episodes (480 train, 60 validation), robosuite scene, 3.0 GB
(`dataset/` 2.9 GB, `episodes/*.npz` 62 MB). First and only collection attempt, no superseded runs. Pilot
(`pilot_20261006/`, 16 episodes, never sealed) and `STATUS.md` kept alongside. See
[PROPS_P1_COLLECTION_20261007.md](PROPS_P1_COLLECTION_20261007.md).

Everything else under `outputs/train/` (smolvla_pilot, single_task, fixed_layout, the plain
`broadworkspace_v1`/`localvaried_v1` runs) is superseded Tier 2-4 work — see
[BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for the chronology. `outputs/downloads_20260920/` and
`outputs/downloads_20260926/` were pre-staged, hardlinked download bundles for the user's Windows
machine (deleted from RTX 2026-10-17, see the reclaim note above) — the 2026-09-26 one covers both the minimal tier and the full checkpoint-ablation tier
for B1 and broad-workspace (~50 GB, zero extra disk here); read the `README.md` inside each
before touching originals. `outputs/downloads_20261005/b3_robosuite/` (2026-10-05) stages B3 steps
124,800 and 156,000; it was copied to Windows (byte-identical) and published as two **private** Hugging Face
repos on 2026-10-06 ([dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560), [model](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800);
the publication record is on the Windows side). A Windows audit then found the B3 bundle not portable on its own:
the checkpoints name the tokenizer by its RTX path and the evaluation code was outside the bundle.
`outputs/downloads_20261009/props_p1/` (2026-10-09) stages props P1 (checkpoints 126,900 and 88,830)
and closes those gaps: it carries the base model and tokenizer (`runtime/`), a source snapshot (`code/`),
`reevaluate/bundle_evaluate.py` and a stdlib-only `reevaluate/verify_bundle.py`. A smoke re-evaluation run
with only the bundle mounted reproduced the original trials bit for bit. Transferred to Windows on
2026-10-09 (user), then published as [P1 dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) and
[P1 model](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) (published private; both
public since 2026-10-09). The B3, B2, B1, broad-workspace and local-varied (SmolVLA) bundles gained
`runtime/`, `code/` and a hash-manifest `reevaluate/verify_bundle.py` in a 2026-10-09 repair, uploaded
straight from RTX (no clean-container re-evaluation from those five). Its
`evaluation_browse/` and `outputs/xvla_props_p1_eval_browse/` are hardlinked views of the P1 trials,
split into held-out and trained, with object, bin, side and outcome in the folder names. Browse them
there, but never rename inside `outputs/xvla_props_p1_eval/`: `evaluate.py` finds trials by
`trial_NNN_<layout_id>`. `outputs/downloads_20261001/b2_distractor/` (deleted from RTX 2026-10-17) was copied to Windows on
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
