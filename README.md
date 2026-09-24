# VLA fine-tuning on Franka Panda cube stacking (MuJoCo)

Comparing [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) and X-VLA on a
cube-stacking task, using sealed, hash-bound datasets collected from a scripted expert in MuJoCo.
Everything runs inside a digest-pinned Docker image — no conda, no host Python dependencies.

The central question: **how much and what kind of data does a VLA need to generalise spatially?**

---

## Current headline (2026-09-24)

On the widest protocol to date — 36 reserved envelope-F7 layouts × 2 policy seeds:

| trained on | SmolVLA | X-VLA |
|---|---|---|
| `local_varied` (60 eps, ±1 cm) | 2.8% (2/72) | 2.8% (2/72) |
| `broad_workspace` (200 eps, F7), 0.50 epochs | 31.9% (23/72) | 68.1% (49/72) |
| `broad_workspace`, **matched exposure** (1.67 epochs) | not run | **86.1%** (62/72) |

Two things are now established. The 200-episode collection is what creates wide-workspace
capability at all — narrowly-trained policies score 2.8%, and their only successes are on the single
holdout layout nearest their training distribution. And the earlier 68.1% was **undertraining, not a
ceiling**: the same data at 1.67 epochs instead of 0.50 reaches 86.1% (Fisher p = 0.0165).

It still fails the 65/72 gate, by three trials. All ten remaining failures are the same failure —
the cube is carried across the workspace correctly, then set down **4.51 cm ± 0.20 cm** beside the
target instead of on top of it. The expert demonstrations contain no such offset (max 1.98 cm, zero
episodes above 2 cm), so this is policy-side, not a data defect.

Full numbers, protocols and confidence intervals: **[docs/RESULTS.md](docs/RESULTS.md)**.

---

## Where things are

| I want to… | go to |
|---|---|
| see every benchmark number | [docs/RESULTS.md](docs/RESULTS.md) |
| know what broke and why | [docs/FAILURES.md](docs/FAILURES.md) |
| run training / evaluation | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| see the plan and what's next | [TODO.md](TODO.md) |
| read the chronological log | [docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md) |
| find a dated report | [docs/INDEX.md](docs/INDEX.md) |
| understand dataset design | [DATASET_DESIGN.md](DATASET_DESIGN.md) |
| work on this as an agent | [AGENTS.md](AGENTS.md) → [.github/copilot-instructions.md](.github/copilot-instructions.md) |

---

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
docs/                 see docs/INDEX.md
```

Each experiment package pins what it consumes: dataset seal, source file hashes, model weight
hashes. A run that cannot reproduce its binding refuses to start — this is deliberate and has
caught real corruption.

---

## Quick start

```bash
# tests for one package
docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/vla -v "$PWD:/vla:ro" -w /vla \
  --entrypoint python sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m unittest discover -s experiments/<pkg>/tests -t experiments/<pkg>/tests
```

Full flags, mounts, gate sequence and monitoring: [docs/OPERATIONS.md](docs/OPERATIONS.md).

---

## Prerequisites

Ubuntu 22.04 · NVIDIA GPU ≥ 24 GB VRAM (tested: RTX 6000 Ada 48 GB) · Docker ≥ 24 ·
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

The superseded `docker-compose` pipeline and its setup instructions are archived at
[docs/archive/docker-compose-pipeline.md](docs/archive/docker-compose-pipeline.md).
