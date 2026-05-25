# SmolVLA — Franka Panda Stacking (MuJoCo)

Fine-tuning [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) on a custom cube-stacking dataset collected in MuJoCo simulation with a Franka Panda robot. Everything runs inside Docker — no conda, no host Python dependencies.

---

## Results

SmolVLA fine-tuned on 50 episodes of the franka stacking task significantly outperforms OpenVLA on the same task. Evaluation videos are saved to `inference/inference_videos/` after running inference.

---

## Repository Structure

```
smolvla_franka/
├── Dockerfile                        # CUDA 12.8 + PyTorch nightly + lerobot[smolvla]
├── docker-compose.yml                # 4 services: push, train, server, client
├── .env.example                      # Credentials template — copy to .env and fill in
├── TODO.md                           # Project progress tracker
│
├── data_collection/
│   ├── collect_libero_dataset.py     # Collect with LIBERO-style camera layout
│   ├── collect_stacking_dataset.py   # Earlier collection script
│   └── collect_v2_dataset.py
│
├── datasets/
│   └── franka_libero_stacking_v1/    # 50 episodes, LeRobot v3.0, 30 FPS
│       └── meta/                     # info.json, stats.json, tasks.parquet
│       (data/ and videos/ are gitignored — stored on HF Hub)
│
├── huggingface/
│   └── push_dataset_to_hub.py        # Upload local dataset to HF Hub
│
├── inference/
│   ├── policy_server_libero.py       # FastAPI inference brain (GPU, port 8000)
│   ├── franka_inference_libero.py    # MuJoCo simulation body (queries server)
│   └── policy_server.py             # Generic server (single-camera variant)
│
└── src/
    ├── controllers/
    │   ├── grasp_controller.py       # Contact-aware gripper controller
    │   └── ik_controller_m2.py      # Inverse kinematics
    ├── robots/franka_panda/
    │   ├── robot.py                  # Pinocchio-based FK/Jacobian
    │   ├── config.py                 # Joint limits, home config, scene path
    │   └── model/                    # MuJoCo XML scene + mesh assets
    ├── planners/                     # RRT + trajectory planner
    └── tasks/franka_pp/              # Pick-and-place / stacking task scripts
```

---

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU (tested on RTX 5090, RTX 5080)
- [Docker](https://docs.docker.com/engine/install/ubuntu/) ≥ 24.0
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify NVIDIA Docker runtime is available:
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.tik.uni-stuttgart.de/st195221/smolvla_franka.git
cd smolvla_franka
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `HF_USERNAME` — your HuggingFace username
- `HF_TOKEN` — HuggingFace access token with write permission ([generate here](https://huggingface.co/settings/tokens))
- `WANDB_API_KEY` — Weights & Biases API key ([find here](https://wandb.ai/settings))
- `HF_REPO_ID` — target HF dataset repo (`<HF_USERNAME>/franka_libero_stacking_v1`)

### 3. Build the Docker image

```bash
docker compose build
```

This builds `smolvla:latest` with:
- CUDA 12.8.1 + cuDNN (compatible with RTX 5090 Blackwell sm_120)
- PyTorch nightly cu128 (stable builds lack Blackwell kernels)
- `lerobot[smolvla]==0.4.4` with correct `transformers` pinning
- MuJoCo 3.x, Pinocchio (`pin`), FastAPI

> **Note on RTX 5090:** PyTorch stable 2.6 only compiles CUDA kernels up to sm_90 (H100). The nightly build is required to get native Blackwell (sm_120) kernels and full GPU utilization.

---

## Dataset

The dataset `franka_libero_stacking_v1` contains 50 teleoperated episodes of cube stacking with:
- 2 cameras: `observation.images.image` (agentview) and `observation.images.image2` (wrist)
- 8-DoF actions: 7 joint positions + gripper
- 30 FPS, ~21,800 frames total
- LeRobot format v3.0

### Push dataset to HuggingFace Hub

Set `HF_REPO_ID` in `.env`, then:

```bash
docker compose run --rm push
```

The dataset is pushed to the `v3.0` branch — this is required. LeRobot's training script fetches datasets from the branch matching `codebase_version` in `info.json`. Pushing to `main` will cause training to fail.

---

## Training

Fine-tune `smolvla_libero` (pretrained on LIBERO):

```bash
docker compose run --rm train
```

**Key training flags (already set in `docker-compose.yml`):**

| Flag | Value | Why |
|---|---|---|
| `--policy.expert_width_multiplier` | `0.5` | **Critical.** Matches the 2-camera LIBERO architecture. Omitting this causes a dimensional mismatch crash. |
| `--policy.use_amp` | `true` | AMP for mixed-precision training on Tensor Cores |
| `--batch_size` | `24` | Tuned for 32GB VRAM; try 32–48 for headroom |
| `--steps` | `20000` | ~4.5 epochs over 50 episodes |

Training logs are tracked on [Weights & Biases](https://wandb.ai) under project `smolvla_franka_project`. Checkpoints are saved to `outputs/train/smolvla_libero_stacking_finetuned/` every 20,000 steps.

---

## Inference

Inference uses a **server/client split**:
- **Server** (`policy_server_libero.py`): loads the fine-tuned model, exposes a FastAPI endpoint on port 8000
- **Client** (`franka_inference_libero.py`): runs the MuJoCo simulation, queries the server for actions, saves evaluation videos

### Step 1 — Allow X11 display access (for MuJoCo viewer)

```bash
xhost +local:docker
```

### Step 2 — Start the inference server

```bash
docker compose up server
```

Wait for:
```
✅ Policy & Processors loaded! Starting inference server on http://localhost:8000...
```

### Step 3 — Run evaluation

In a second terminal:

```bash
docker compose run --rm client
```

Runs 10 episodes. Videos are saved to `inference/inference_videos/`.

---

## Data Collection

To collect new episodes:

```bash
# Interactive MuJoCo session with keyboard teleoperation
docker compose run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  smolvla:latest \
  python data_collection/collect_libero_dataset.py
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `KeyError: getpwuid(): uid not found` | `HOME=/vla` and `TORCHINDUCTOR_CACHE_DIR` must be set — already in `docker-compose.yml` |
| `unable to allocate shared memory` | `shm_size: '8gb'` on the train service — already set |
| `No module named 'transformers'` | Use `lerobot[smolvla]` not bare `lerobot` — already in `Dockerfile` |
| `FileExistsError: output directory already exists` | Delete via Docker: `docker run --rm -v $(pwd):/vla smolvla:latest rm -rf /vla/outputs` |
| Files created as root in project dir | All services use `user: "1001:1001"` in `docker-compose.yml` |
