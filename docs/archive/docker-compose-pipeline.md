# ARCHIVE — docker-compose pipeline (superseded)

> Historical record of the original `docker-compose` workflow and the
> `franka_libero_stacking_v1/v2` HuggingFace datasets. **Superseded** by the sealed,
> hash-bound `experiments/` packages. Kept for provenance of the v1 0% result and the
> v2 language-grounding design; do not follow these instructions for current work.
>
> Current operating procedure: [docs/OPERATIONS.md](../OPERATIONS.md).

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
