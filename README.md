# SmolVLA — Franka Panda Cube Stacking (MuJoCo)

Fine-tuning [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) on a custom cube-stacking dataset collected in MuJoCo simulation with a Franka Panda robot. Everything runs inside Docker — no conda, no host Python dependencies.

---

## Current status (2026-09-22)

`broad_workspace_data_v1` is **collected and sealed**: 200 episodes, 105,600 frames, 160 train / 40 validation,
seal `0351d3da36294e2124554267bc17738badee26498b1cb82fa7a58d4f4a601a85`. Both policies were retrained on it
and evaluated on the 36 reserved holdout layouts that collection never executed.

**Broad-workspace results** — 36 reserved F7 layouts × 2 policy seeds, plus 20 fixed-parity seeds:

| Metric | SmolVLA | X-VLA | Δ |
|---|---|---|---|
| broad_holdout success rate | 31.9% (23/72) | **68.1% (49/72)** | **+36.2 pp** |
| fixed_parity success rate | (pending) | 100% (20/20) | — |
| offline train arm MAE (rad) | 0.01904 | 0.01655 | −13% |
| offline val arm MAE (rad) | 0.02157 | 0.01748 | −19% |
| layouts failing both seeds | 19 / 36 | 8 / 36 | — |

**Both policies fail the 65/72 holdout gate.** All failures are timeouts, none nonfinite. The gate was set for
the easier local-varied protocol and has deliberately not been lowered to match the outcome.

The 8 layouts X-VLA fails on both seeds were screened with the scripted expert: **all 8 are solvable**. The
holdout is sound, so 68.1% is a real policy limit rather than an artefact of unverified layouts.

**Did the 200-episode collection actually buy generalisation? Yes — decisively.** The same two protocols run
against the *local-varied* checkpoints on the identical 72-trial roster:

| Checkpoint trained on | X-VLA | SmolVLA |
|---|---|---|
| local-varied (48 eps, ±1 cm) | 2.8% (2/72) — 95% CI [0.8, 9.6] | 2.8% (2/72) — [0.8, 9.6] |
| broad-workspace (160 eps, F7) | **68.1%** (49/72) — [56.6, 77.7] | **31.9%** (23/72) — [22.3, 43.4] |
| gain | **+65.3 pp** | **+29.2 pp** |

Confidence intervals do not overlap. Both local-varied models succeeded on exactly one layout —
`evaluation_031`, on both seeds — which is the **closest of all 80** holdout layouts to their training base
(distance 0.0297 vs 0.2209 max). So the 2.8% is not noise: it is the single holdout layout that happens to
fall inside the ±1 cm neighbourhood they were trained in. Outside it, they have no competence at all, despite
scoring 100% on `fixed_parity`.

Two caveats on that claim:
- The comparison confounds **breadth with volume** (48 vs 160 episodes). The defensible statement is that the
  broad-workspace *collection* produced the capability, not that envelope breadth specifically did.
- The SmolVLA cross-eval runs under a data-level binding (`metadata_binding_reproducible: False`): that run
  predates the `local_varied_data` re-seal, so its metadata binding is unreproducible. Training data identity
  is instead proven by recomputed statistics, and checkpoint identity by weight hash.

**Open: 68.1% is likely undertrained, not a ceiling.** X-VLA saw only **0.50 epochs** on the broad dataset
(40,000 chunks over 79,840 starts) versus 1.67 epochs on local-varied — updates were held at 5000 while the
data tripled. Train/val arm MAE 0.01655/0.01748 shows no meaningful overfitting. Next step is to measure the
checkpoint curve (3000/4000/5000) before committing to a longer run; see [TODO](TODO.md) Phase A.

**Earlier local-varied results** (narrow ±1 cm protocol, not comparable to the above — different task difficulty):

| Metric | SmolVLA | X-VLA |
|---|---|---|
| varied_holdout success rate | 88.9% (64/72) | **98.6% (71/72)** |
| fixed_parity success rate | 100% (20/20) | 100% (20/20) |
| offline val arm MAE (rad) | 0.01133 | 0.00996 |

Open question: whether the 200-episode collection actually improved generalisation. A cross-evaluation of the
**local-varied** X-VLA on the identical broad holdout is built and queued; without it 68.1% has no reference point.

---

## Datasets & training runs

| Package | Purpose | Episodes | Status |
|---|---|---|---|
| `experiments/fixed_layout/` | Single-geometry pilot (source reset ep 7) | 24 | sealed |
| `experiments/local_varied_data/` | 60 layouts, ±1° pose jitter | 60 (48 train / 12 val) | sealed |
| `experiments/broad_workspace_data/` | F7 envelope, retries permitted | 200 (160 train / 40 val) | **sealed** |
| `experiments/{smolvla,xvla}_local_varied/` | Local-varied finetunes | — | trained + evaluated |
| `experiments/{smolvla,xvla}_broad_workspace/` | Broad-workspace finetunes | — | trained + evaluated |
| `experiments/xvla_localvaried_broad_eval/` | Cross-eval: old checkpoint, new holdout | — | built, queued |

Training reports: `outputs/train/{smolvla,xvla}_franka_stacking_{localvaried_v1_20260919,broadworkspace_v1_20260921}/result.json`.
Evaluation reports: `.../evaluation/{varied_holdout,broad_holdout,fixed_parity}/result.json`, per-trial `episode.json`, `trace.npz`, `cameras.mp4`.

---

## Variations tried

- **Datasets**: fixed pilot → local-varied (60 layouts × ±1° jitter) → broad-workspace (F7 envelope × ±1° × ±3 cm).
- **Policies**: SmolVLA (`chunk_size=50, n_action_steps=10`) vs X-VLA (`chunk_size=30, n_action_steps=10`).
- **Failure handling**: no-retry (fixed_layout) → retry-with-rejection-budget (broad_workspace, 100 rejections allowed per 200 admits).
- **Rollout precision**: SmolVLA uses FP32; X-VLA uses FP32 params + BF16 autocast (matches training loss path exactly).

---

## Failures we hit (and what we learned)

1. **Expert failure counted as stop** — first broad_workspace run halted at 52/200 because a failed expert bubbled up as a hard error. Fix: wrap in `ExpertRejection`, bill to rejection budget (`57975c5`).
2. **Deleted the pilot manifest** during cleanup, breaking every downstream `provenance()` check. Fix: make manifest optional in `fixed_layout/source.py` (`57f558d`), then re-seal `local_varied_data` with post-surgery provenance and update `SEAL` constants (`f3f3244`).
3. **X-VLA reload verify used CPU + no autocast** — checkpoint verification hit `Input cuda / weight cpu` because obs came from the CUDA training path but the reloaded policy stayed on CPU. Fix: `.to('cuda')` + `torch.autocast('cuda', bfloat16)` in `_reload_and_verify` (`391049c`).
4. **LeRobot per-episode PNG staging accumulated across rejections** — rejected ep 52's 528 PNGs stayed under `episode-000052/`, so ep 53's writes gave 1056 files → encoder coverage check tripped. Fix: purge staging PNG dir + reset `episode_buffer` after every `ExpertRejection` (`4561995`).
5. **Attempts dir naming diverged from LeRobot's episode_index after rejections** — encoder read ticks from `attempts/000052/` (rejected) while PNGs came from ep 53. Fix: expose `dataset._active_attempt_path` set in `record_attempt`, use it in the encoder wrapper (`2435dca`).
6. **Validator hard-coded `slot['episode_index']+1` as expected admit count**, which no longer matched LeRobot's `total_episodes` post-rejection. Fix: read `info.json['total_episodes']` as the count and let the remaining `_metadata` invariants (schema, task, fps, frame count) do the checking (`3d733d2`).
7. **Preflight scratch dir deleted** to free disk before eval → `verify_final` couldn't hash `SCRATCH/result.json`. Fix: make scratch check conditional in the eval package (`c8a6904`). The preflight `result.json` had non-deterministic timing fields so it wasn't reproducible.
8. **Editing source files while eval was still running** → final `c.admission()` re-check saw drift and marked the run "failed" even though all 92 rollouts had already completed cleanly. Lesson: freeze the code path for the duration of the eval.
9. **Rejections skipped the slot instead of retrying it** — this was the root cause behind three separate 35-hour collection failures at the same point. Slot index doubled as the admitted-episode index, and `metadata[i]`, `attempts/{i:06d}`, `resets/episode_{i:03d}` and the train/val split boundaries all depend on `slot == admit`. Skipping broke every one. Fix: retry the same slot until admitted (`3926614`).
10. **The tests asserted the buggy behaviour.** 35/35 passed while production died three times identically, because `test_expert_execution_failure_billed_to_rejection_budget_not_stop` demanded `len(admitted) == 198` — i.e. it *required* slots to be skipped. Rewrote the tests to assert correct behaviour and added a regression test that runs `validate_journal` over scheduler output. Green tests are not evidence when they encode the defect.
11. **Two layouts were physically uncompletable / irreproducible.** `validation_020` failed all 16 expert attempts (cube 2.0–2.1 cm off, admitted episodes ≤1.74 cm). `validation_032` stacked fine but its grasp geometry made the float32 replay drift 1.51e-3 (tolerance 1e-3) *identically on every redraw* — variance <1e-8 across independent ±1° jitters, proving it systematic rather than chaotic, so no amount of redrawing could fix it. Fix: deterministic layout resampling inside the same envelope (`76d8e91`, `c429b1d`). Both fall in the validation split, so the 160 training episodes are untouched.
12. **Resume, rollback and unique rejected paths** — a crash at 179/200 would have discarded everything. Added `resume_state` + `amend_frozen_plan`, episode rollback with stats re-aggregation for drift rejections (`91cc4cf`), and unique `rejected/{i}_k{k}_r{n}` paths after a rename collision killed a run (`7807c7c`).
13. **The evaluation inherited pilot-era workspace bounds.** `restore_scene` gates on `pilot_integrity.validate_scene_geometry`, which caps |y| at 0.40 m — but envelope F7 reaches 0.45 m. Scene derivation was provably correct (every per-array SHA256 matched the sealed bank) yet 12 of 36 holdout layouts were rejected, precisely the widest and most discriminative ones. Relaxing the shared validator would have invalidated the sealed provenance of every dataset, so the eval packages restore with the collector's own F7 geometry gate instead (`e11c784`).
14. **A holdout failure is not automatically a policy failure.** The 80 reserved layouts were never expert-verified, and 8 of them failed X-VLA on both seeds — indistinguishable from an unsolvable layout. Screening all 8 with the scripted expert showed every one is solvable, which is what makes 68.1% reportable. Do this screening before reporting any systematic failure cluster.

Broader lessons — every "just increment a counter" or "just leave state alone on retry" assumption in the fixed-layout pipeline broke under broad_workspace retries. Rejections need explicit cleanup for every piece of shared state (LeRobot buffer + PNG dir + attempts dir semantics + admit-count invariant). And a result is only interpretable once you know what the *absence* of the treatment scores on the same protocol.

---

## Repository Structure

```
smolvla_franka/
├── Dockerfile                         # CUDA 12.8.1 + PyTorch nightly cu128 + lerobot[smolvla]==0.4.4
├── docker-compose.yml                 # 5 services: collect, push, train, server, client
├── .env.example                       # Credentials template — copy to .env and fill in
├── TODO.md                            # Project progress tracker
├── DATASET_DESIGN.md                  # v1 failure analysis + v2 design rationale
│
├── data_collection/
│   └── collect_libero_dataset.py      # v2 collection: 12 task variants, 512×512, ETA display
│
├── huggingface/
│   └── push_dataset_to_hub.py         # Upload local dataset to HF Hub
│
├── inference/
│   ├── policy_server_libero.py        # FastAPI inference brain (GPU, port 8000)
│   └── franka_inference_libero.py     # MuJoCo sim body (queries server, saves videos)
│
└── src/
    ├── controllers/
    │   ├── grasp_controller.py        # Contact-aware gripper controller
    │   └── ik_controller_m2.py        # Inverse kinematics (Pinocchio)
    ├── robots/franka_panda/
    │   ├── robot.py                   # FK / Jacobian via Pinocchio
    │   ├── config.py                  # ACTIVE_JOINTS, Q_HOME, SCENE_PATH
    │   └── model/                     # MuJoCo XML scene + mesh assets
    └── planners/
        └── trajectory_planner.py      # Task-space cubic trajectory planner
```

---

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with ≥ 16 GB VRAM (tested: RTX 5090 32 GB, RTX 5080 16 GB)
- [Docker](https://docs.docker.com/engine/install/ubuntu/) ≥ 24.0
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify the NVIDIA runtime is available:
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

> **RTX 5090 (Blackwell sm_120):** PyTorch stable 2.6 only compiles CUDA kernels up to sm_90 (H100). The Dockerfile uses PyTorch nightly cu128 for native Blackwell kernels. Without it, every operation falls back to slow PTX JIT compilation.

---

## Setup

### 1. Clone and configure credentials

```bash
git clone https://github.tik.uni-stuttgart.de/st195221/smolvla_franka.git
cd smolvla_franka
cp .env.example .env
```

Edit `.env`:
```
HF_USERNAME=your_huggingface_username
HF_TOKEN=hf_...           # write-permission token from huggingface.co/settings/tokens
WANDB_API_KEY=...          # from wandb.ai/settings
HF_REPO_ID=your_username/franka_libero_stacking_v2
```

### 2. Build the Docker image

```bash
docker compose build
```

This installs: CUDA 12.8.1, PyTorch nightly cu128, `lerobot[smolvla]==0.4.4`, MuJoCo 3.x, Pinocchio, FastAPI, osmesa.

---

## Dataset Collection

### Run collection (headless, max speed)

```bash
docker compose run --rm collect
```

Collects 300 episodes across **12 colour-pair instruction variants** (all combinations of red/green/blue/yellow pick→place). Each episode saves two 512×512 video streams (front camera + wrist camera) at 30 FPS.

The script auto-versions the dataset (`franka_libero_stacking_v1`, `v2`, …) based on what already exists in `datasets/`.

**Terminal output during collection:**
```
[17/300] Task: 'pick up blue cube and place it on top of red cube' | cubes: ['blue', 'red', 'yellow']
  [████████████░░░░░░░░] Move to Place        sim  9.8s | ETA 03h12m36s
  [17.85s] SUCCESS (xy=0.5cm, z=4.2cm)
  Progress: 17/300 | avg 41s/ep | ETA 03h12m | elapsed 0.22h
```

**Key design choices (see DATASET_DESIGN.md for full rationale):**

| Parameter | Value | Why |
|---|---|---|
| Instructions | 12 variants | All colour-pair combos; v1 used only red→blue (0% inference success) |
| Image size | 512×512 | Matches SmolVLM2 `max_image_size.longest_edge=512`; 480×640 adds black padding bars |
| Cube clearance | 15 cm | Prevents accidental grasp of wrong cube |
| Distractors | 0–2 random | Model can't use object count as a shortcut |
| Approach jitter | ±5 mm | Prevents path memorisation |

### Monitor a running collection

```bash
docker ps --filter ancestor=smolvla:latest        # check it's running
docker logs --tail 5 <container_id>               # check current episode/ETA
```

### Overnight collection (prevent system sleep)

```bash
loginctl enable-linger $USER          # processes survive logout (run once)

tmux new -s collect
docker compose run --rm collect
# Ctrl+B → D to detach

tmux new -s nosleep
systemd-inhibit --what=sleep:idle --mode=block sleep 36000
# Ctrl+B → D to detach

Super+L                               # lock screen, do NOT log out
```

To check in the morning:
```bash
tmux attach -t collect                # from a system terminal (not VS Code)
# or:
docker logs --tail 5 $(docker ps -q --filter ancestor=smolvla:latest)
```

---

## Push Dataset to HuggingFace Hub

Set `HF_REPO_ID` in `.env` to match your new dataset name, then:

```bash
docker compose run --rm push
```

The dataset is pushed to the **`v3.0` branch** — this is required. LeRobot's training script fetches the branch matching `codebase_version` in `info.json`. Pushing to `main` will cause training to fail silently.

---

## Training

```bash
docker compose run --rm train
```

The compose command **auto-detects fresh vs resume**: if a checkpoint exists in `outputs/train/.../checkpoints/`, it resumes from the latest one; otherwise starts fresh. Safe to re-run after interruption.

**Current hyperparameters (set in `docker-compose.yml`):**

| Flag | Value | Why |
|---|---|---|
| `--policy.expert_width_multiplier` | `0.5` | **Critical** — smolvla_libero expects exactly 2 cameras; omitting crashes with dimensional mismatch |
| `--batch_size` | `32` | Safe for 32 GB VRAM (estimated ~25 GB peak with AMP); batch 48+ risks OOM |
| `--steps` | `80000` | ~16 epochs over 300 episodes at batch 32; 20K (3 epochs) was confirmed undertrained |
| `--policy.use_amp` | `true` | Mixed precision on Tensor Cores |
| `--dataset.repo_id` | `${HF_USERNAME}/franka_libero_stacking_v2` | v2: 300 episodes, 12 instruction variants |

**Training time:** ~8 hours on RTX 5090 at ~3 steps/sec.

### Overnight training (survive user switching / logout)

```bash
loginctl enable-linger $USER          # run once per user

tmux new -s training
docker compose run --rm train
# Ctrl+B → D to detach

tmux new -s nosleep
systemd-inhibit --what=sleep:idle --mode=block sleep 36000
# Ctrl+B → D to detach

Super+L                               # lock screen, do NOT log out
```

### Monitor training progress

```bash
docker logs --tail 5 $(docker ps -q --filter ancestor=smolvla:latest)
```

Example output:
```
Training: 22% |████████▎ | 17400/80000 [1:36:31<5:45:57, 3.02step/s]
INFO step:17K smpl:557K epch:25.54 loss:0.006 grdn:0.194 lr:4.0e-05
```

### Check for duplicate training containers (halves speed)

```bash
docker ps --filter ancestor=smolvla:latest
# Should show exactly ONE container. Kill duplicates:
docker kill <duplicate_container_id>
```

---

## Inference

Inference uses a **server/client split**:
- **Server** — loads the fine-tuned model onto GPU, exposes FastAPI on port 8000
- **Client** — runs MuJoCo simulation, queries server for actions at 30 Hz, saves videos

### Terminal 1 — Start the policy server

```bash
docker compose up server
```

Wait for:
```
✅ Policy & Processors loaded! Starting inference server on http://localhost:8000...
```

### Terminal 2 — Run evaluation (once server is ready)

```bash
xhost +local:docker          # allow container to use your X11 display
docker compose run --rm client
```

Runs 30 episodes. Videos saved to:
```
inference/inference_videos/smolvla_franka_libero_stacking_finetuned_b32_s80k/
├── run_1/
│   ├── eval_episode_1.mp4
│   ├── ...
│   └── inference_summary.txt
└── run_2/    ← next time you run client
```

Each `docker compose run --rm client` auto-increments the run number.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `KeyError: getpwuid(): uid not found` | `HOME=/vla` and `TORCHINDUCTOR_CACHE_DIR` must be set — already in `docker-compose.yml` |
| `unable to allocate shared memory` | `shm_size: '8gb'` on train service — already set |
| `No module named 'transformers'` | Use `lerobot[smolvla]` not bare `lerobot` — already in `Dockerfile` |
| `ValueError: Image height > framebuffer height` | Add `offheight="512" offwidth="512"` to `<visual><global>` in scene XML — already fixed |
| `AttributeError: 'NoneType' ... eglQueryString` | Switch `MUJOCO_GL=egl` → `MUJOCO_GL=osmesa` for headless — already set on collect service |
| `ValueError: A config_path is expected when resuming` | Auto-resume handles this — `docker compose run --rm train` finds latest checkpoint automatically |
| `Server not found` during inference | Client uses `network_mode: host`; server URL must be `localhost:8000` not `server:8000` — already fixed |
| Training ETA doubles unexpectedly | Two training containers running simultaneously — check with `docker ps --filter ancestor=smolvla:latest` and kill the duplicate |
| `duplicate session: training` (tmux) | VS Code terminal is itself inside tmux; open a system terminal (Ctrl+Alt+T) to nest tmux sessions |
| `sessions should be nested with care` | Use `docker logs -f <id>` instead of attaching tmux from within VS Code |
