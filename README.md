# VLA fine-tuning on Franka Panda cube stacking (MuJoCo and robosuite)

Comparing [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) and X-VLA on a
cube-stacking task, using sealed, hash-bound datasets collected from a scripted expert — first in a
plain MuJoCo scene, latterly in a robosuite-composed scene.
Everything runs inside one of two digest-pinned Docker images — no conda, no host Python dependencies.

The central question: **how much and what kind of data does a VLA need to generalise spatially?**

---

## Latest (2026-10-05): B3 in the robosuite scene — the last cube experiment

X-VLA trained 156,000 updates (5 epochs) on 560 scripted episodes collected in the robosuite scene:
four cube colours with two of the twelve instruction pairs never collected, 0–2 distractors, extra
picks at the region edges. Evaluated on reserved layouts, 160 trials per checkpoint:

| checkpoint | overall | trained pairs | held-out pairs | no distractor | distractors |
|---|---|---|---|---|---|
| 62,400 | 68.8% | 68.8% | 68.8% | 70.0% | 67.5% |
| 93,600 | 72.5% | 72.5% | 72.5% | 78.8% | 66.3% |
| **124,800** | **88.1%** | 88.8% | 87.5% | 95.0% | 81.3% |
| 156,000 (final) | 65.6% | 70.0% | 61.3% | 71.3% | 60.0% |

- **The policy follows colour instructions it never trained on**: held-out pairs within 10 points of
  trained ones at every checkpoint, and 3 wrong-cube picks in 640 trials.
- **Quote 124,800 as about 85% overall and about 75% with distractors.** The report both chose and
  measured it; the independent screen gives 13/20 on the distractor condition.
- **The final checkpoint is the worst**, almost entirely on one side of the workspace, while the
  training loss kept falling. The persistent failures sit where B2's did: picks close to the robot
  base on the +y side.
- Robosuite 1.5.2 / MuJoCo 3.4.0 — not comparable with the MuJoCo-scene numbers below.

Report: [docs/B3_XVLA_EVALUATION_20261005.md](docs/B3_XVLA_EVALUATION_20261005.md). Next: real-life
props in robosuite ([TODO.md](TODO.md)); P0 (scene and scripted expert, no policy) passed its gate on 2026-10-05,
283/288 against 95%, after two spec errors found by the first run were amended
([docs/PROPS_P0_RERUN_20261005.md](docs/PROPS_P0_RERUN_20261005.md)); P1 design is next.

## The results arc

| date | experiment | headline |
|---|---|---|
| 2026-09-24 | broad workspace (spatial), MuJoCo | X-VLA 86.1% at matched exposure |
| 2026-09-26 | B1 language grounding, MuJoCo | 94.4%, no positional shortcut |
| 2026-09-30 | B2 distractors, MuJoCo | 91.9% at 50,000 updates; no detectable distractor effect |
| 2026-10-02 | sim2sim: the B2 policy on the robosuite scene | 3/20 against 19/20 on its own scene |
| 2026-10-05 | B3, robosuite (four colours, held-out pairs, harder distractors) | ~85% overall at the best checkpoint |

Every number with its protocol and caveats: [docs/RESULTS.md](docs/RESULTS.md).

---

## Previous headline (2026-09-26, MuJoCo scene)

**B1 language grounding: 94.4% (151/160), and no positional shortcut.** Every prior result on
this repo (including the 86.1% below) measured spatial capability only — the training data made
cube colour perfectly predictable from position, so a policy that never read the instruction
could score 100% ([FAILURES #27](docs/FAILURES.md)). B1 collected a fresh 200-episode 2×2
factorial dataset (instruction × colour/region arrangement) to break that confound, trained
X-VLA to 33,400 updates, and evaluated on 40 held-out layouts never used for checkpoint
selection:

| contrast | rate | Fisher p |
|---|---|---|
| overall | 151/160 = 94.4% | — |
| arrangement: familiar vs. swapped | 93.8% vs. 95.0% | 1.000 |
| pick colour: red vs. green | 96.2% vs. 92.5% | 0.495 |

Swapped is not worse than familiar — the signature a positional shortcut would leave, since
swapping relocates the correct-coloured cube to the region a shortcut would ignore. See
[RESULTS.md](docs/RESULTS.md) for the full breakdown and checkpoint-selection numbers.

## Prior headline (2026-09-24)

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
the cube is carried across the workspace correctly, then set down roughly one cube width
(**+3.9 cm tangentially**, a ~5° bearing error) beside the target instead of on top of it. The
expert demonstrations contain no such offset (max 1.98 cm, zero episodes above 2 cm), and the
failure is **layout-specific**: a 128-trial measurement found control layouts at 64/64 while three
layouts draw the wrong mode 7 times in 8. It is a policy-side sampling property, not a data defect.

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
