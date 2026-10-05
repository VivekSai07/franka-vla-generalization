# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [AGENTS.md](AGENTS.md), which points at
[.github/copilot-instructions.md](.github/copilot-instructions.md) — the single source of agent
truth for this repository. Read that file before doing anything else here; it covers the
non-negotiable rules (hash-bound files, dataset deletion, verification gates), evidence
standards, experiment-design rules, and the documentation pass required after every finding.

## What this repo is

Comparing SmolVLA and X-VLA fine-tuning on a Franka Panda cube-stacking task, using sealed,
hash-bound datasets collected from a scripted expert — in a plain MuJoCo scene through B2, in a
robosuite-composed scene for B3. Central question: how much and what kind of data does a VLA need to
generalise? The cube programme closed with B3 (2026-10-05); the next project is real-life props in
robosuite (P0, the scene and a scripted expert, ran on 2026-10-05 and failed its gate; see
[docs/PROPS_P0_20261005.md](docs/PROPS_P0_20261005.md)). See [README.md](README.md) for results and
[TODO.md](TODO.md) for what is open.

## Layout

```
experiments/          one self-contained package per experiment
  <dataset>_data/       collection: plan, expert, scenes, validation, seal
  {smolvla,xvla}_<ds>/  training: contract.py (what is bound) + runner.py (what runs)
  *_eval/               evaluation protocols
outputs/              datasets, training runs, evaluations   (gitignored)
datasets/             pilot source episode + manifest        (gitignored, never delete)
src/                  robot model, controllers, IK, planners
data_collection/      shared simulation and benchmark helpers
validation/           replay, gates, rollout diagnostics
docs/                 see docs/INDEX.md — RESULTS.md, FAILURES.md, OPERATIONS.md, BENCHMARK_STATUS.md
```

Each experiment package pins what it consumes (dataset seal, source file hashes, model weight
hashes); a run that cannot reproduce its binding refuses to start.

Everything runs inside one of two digest-pinned Docker images — no conda, no host Python
dependencies. Nothing runs outside the two images below.

## Pinned image

```
sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
```

Python 3.10, LeRobot 0.4.4, PyTorch 2.7.1+cu128, MuJoCo 3.13.0. Runs every sealed package, B3
training (`experiments/xvla_b3_robosuite/`, which needs no simulator), and every test that imports no
simulator.

Second image, for everything that builds the robosuite scene: `experiments/robosuite_scene/`, the rollout stage of
`experiments/robosuite_sim2sim/`, B3 collection (`experiments/b3_robosuite_data/`), B3 evaluation
rollouts (`experiments/xvla_b3_robosuite_eval/`) and the props scene, expert and gate
(`experiments/props_scene/`, not hash-bound):

```
sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
```

The image above plus robosuite 1.5.2 and MuJoCo 3.4.0 (robosuite 1.5.2 cannot reset an environment
on 3.13.0). Results from the two images are not physics-comparable. Local tags, container names,
build notes and the `--ulimit nofile` / large `/tmp` settings long runs need:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Commands

Run tests for one experiment package:

```bash
docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/vla \
  -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/implementation/broad_workspace_data_20260920:/vla/outputs/implementation/broad_workspace_data_20260920:rw" \
  -v "$PWD/outputs/implementation/local_position_probe_20260919:/vla/outputs/implementation/local_position_probe_20260919:rw" \
  -v "$PWD/outputs/implementation/local_varied_data_20260919:/vla/outputs/implementation/local_varied_data_20260919:rw" \
  -v "$PWD/outputs/implementation/b1_language_data_20260925:/vla/outputs/implementation/b1_language_data_20260925:rw" \
  -w /vla \
  --entrypoint python sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m unittest discover -s experiments/<pkg>/tests -t experiments/<pkg>/tests
```

`-t` must be the tests directory, not `.` — tests have no `__init__.py` and import siblings
directly; `-t .` fails with `Start directory is not importable`.

### Known pre-existing failures (not regressions)

Five packages fail on `main` and every branch, caused by the deleted pilot manifest
([docs/FAILURES.md](docs/FAILURES.md) #2):

| package | errors |
|---|---|
| `fixed_layout` | 13 |
| `single_task` | 5 |
| `xvla_single_task` | 4 |
| `smolvla_fixed_layout` | 1 |
| `smolvla_local_varied` | 1 |

Before treating any of these as a regression, compare counts against `main` in a worktree.

Training/evaluation launch flags, mounts, the `--check` / `--preflight` / `--train` gate
sequence, and monitoring caveats (Docker `wait` does not reliably notify in this environment —
poll actively): [docs/OPERATIONS.md](docs/OPERATIONS.md). Before launching any long run, work
through the checklist in
[.github/skills/launching-a-run/SKILL.md](.github/skills/launching-a-run/SKILL.md) (the only copy).

## Hash-bound files — do not edit

These files are inside sealed provenance and cannot be corrected without invalidating sealed
results. Do not edit them even to fix a stale statement:

- `experiments/broad_workspace_data/README.md`
- `experiments/xvla_local_varied/README.md`
- `experiments/smolvla_local_varied/README.md`
- `experiments/b3_robosuite_data/` (every `.py` including tests, and its README), and
  `experiments/robosuite_scene/*.py`, the B3 spec and the B3 collection plan: the B3 seal's
  `provenance()` hashes them (sealed 2026-10-03)

Before editing anything under `experiments/`, check whether it is in `SOURCE_FILES` or reachable
from `provenance()`. Package READMEs are always in both, except in `experiments/robosuite_sim2sim/`
and the README of `experiments/robosuite_scene/`, which have no contract and no `provenance()`
(`experiments/robosuite_scene/*.py` are bound through `b3_robosuite_data`'s `provenance()`).

Working on the robosuite path: load the `working-on-the-robosuite-path` skill first.

## Evidence and documentation

- Never claim something passed without running it and reading the output.
- Offline metrics (loss, action MAE) are not capability proxies — a checkpoint that scored 0/72
  on rollouts improved offline metrics monotonically through that run.
- Every finding triggers a documentation pass: grep all live docs for the specific figures and
  claims your finding changes, and update every hit in the same turn. Details in
  [.github/copilot-instructions.md](.github/copilot-instructions.md).

## Commits

Commits are authored solely by the user — no AI or co-author trailers, short/simple messages, no
secrets/datasets/checkpoints/generated videos. (This overrides the default Claude Code
attribution footer for this repository.)
