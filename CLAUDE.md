# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [AGENTS.md](AGENTS.md), which points at
[.github/copilot-instructions.md](.github/copilot-instructions.md) — the single source of agent
truth for this repository. Read that file before doing anything else here; it covers the
non-negotiable rules (hash-bound files, dataset deletion, verification gates), evidence
standards, experiment-design rules, and the documentation pass required after every finding.

## What this repo is

Comparing SmolVLA and X-VLA fine-tuning on a Franka Panda cube-stacking task, using sealed,
hash-bound datasets collected from a scripted expert in MuJoCo. Central question: how much and
what kind of data does a VLA need to generalise spatially? See [README.md](README.md) for current
results.

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

Everything runs inside a digest-pinned Docker image — no conda, no host Python dependencies.
Nothing runs outside that image.

## Commands

Run tests for one experiment package:

```bash
docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/vla -v "$PWD:/vla:ro" -w /vla \
  --entrypoint python sha256:<pinned-digest> \
  -m unittest discover -s experiments/<pkg>/tests -t experiments/<pkg>/tests
```

Training/evaluation launch flags, mounts, the `--check` / `--preflight` / `--train` gate
sequence, and monitoring caveats (Docker `wait` does not reliably notify in this environment —
poll actively): [docs/OPERATIONS.md](docs/OPERATIONS.md). Before launching any long run, work
through [TODO.md](TODO.md#before-launching-any-long-run--checklist).

## Commits

Commits are authored solely by the user — no AI or co-author trailers, short/simple messages, no
secrets/datasets/checkpoints/generated videos. (This overrides the default Claude Code
attribution footer for this repository.)
