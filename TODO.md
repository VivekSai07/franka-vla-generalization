# SmolVLA Project TODO

## Environment Setup
- [x] Identify RTX 5090 + CUDA 13.0 (driver 580.142) on new lab machine
- [x] Confirm Docker 29.4.2 + NVIDIA runtime already configured
- [x] Write `Dockerfile` — CUDA 12.8.1 + PyTorch nightly cu128 (required for Blackwell sm_120)
- [x] Write `docker-compose.yml` — 4 services: `push`, `train`, `server`, `client`
- [x] Write `.env.example` + `.gitignore`
- [x] Build `smolvla:latest` image
- [x] Verify GPU inside container — PyTorch 2.10.0+cu128, no sm_120 warning, RTX 5090 detected

## Security / Housekeeping
- [x] Remove hardcoded HF token from `huggingface/push_dataset_to_hub.py`
- [x] Fix Windows path → container path `/vla/datasets/franka_libero_stacking_v1`
- [x] `SERVER_URL` in `inference/franka_inference_libero.py` reads from env var (works both in compose and on host)

## Dataset
- [x] `datasets/franka_libero_stacking_v1/` present locally — 50 episodes, LeRobot v3.0, 30 FPS
- [x] Push `franka_libero_stacking_v1` to HF Hub (`VivekSai07/franka_libero_stacking_v1`, branch `v3.0`)
  - Branch must be `v3.0` — lerobot-train fetches from the branch matching `codebase_version` in `info.json`

## Training
- [ ] Fine-tune `smolvla_libero` on `franka_libero_stacking_v1`
  - batch_size=24, steps=20000, AMP enabled
  - **Critical flag:** `--policy.expert_width_multiplier=0.5` (2-camera model: `image` + `image2`)
  ```bash
  docker compose run --rm train
  ```
- [ ] Verify W&B training dashboard — `smolvla_franka_project`
- [ ] Confirm checkpoint saved to `outputs/train/smolvla_libero_stacking_finetuned/`
- [ ] Push fine-tuned weights to HF Hub (`VivekSai07/smolvla_franka_libero_stacking_finetuned`)

## Inference
- [ ] Allow X11 forwarding for MuJoCo viewer: `xhost +local:docker`
- [ ] Start inference server
  ```bash
  docker compose up server   # wait for "✅ Policy & Processors loaded!"
  ```
- [ ] Run 10-episode evaluation
  ```bash
  docker compose run --rm client
  ```
- [ ] Review evaluation videos in `inference/inference_videos/`
- [ ] Compare stacking success rate vs. previous RTX 5080 run

## Future / Optional
- [ ] Collect additional episodes with `data_collection/collect_libero_dataset.py` to expand dataset
- [ ] Experiment with higher batch size (5090 has 32GB — try batch_size=32 or 48)
- [ ] Try `smolvla_base` (not fine-tuned on LIBERO) and compare zero-shot performance
