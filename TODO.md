# SmolVLA Project TODO

## Environment Setup
- [x] Identify RTX 5090 + CUDA 13.0 (driver 580.142) on new lab machine
- [x] Confirm Docker 29.4.2 + NVIDIA runtime already configured
- [x] Write `Dockerfile` — CUDA 12.8.1 + PyTorch nightly cu128 (required for Blackwell sm_120)
- [x] Write `docker-compose.yml` — 5 services: `collect`, `push`, `train`, `server`, `client`
- [x] Write `.env.example` + `.gitignore`
- [x] Build `smolvla:latest` image
- [x] Verify GPU inside container — PyTorch 2.10.0+cu128, no sm_120 warning, RTX 5090 detected

## Security / Housekeeping
- [x] Remove hardcoded HF token from `huggingface/push_dataset_to_hub.py`
- [x] Fix Windows path → container path `/vla/datasets/franka_libero_stacking_v1`
- [x] `SERVER_URL` in `inference/franka_inference_libero.py` reads from env var
- [x] Remove `datasets/` from git — push to HF Hub instead
- [x] Ignore `.local/`, `.bash_history`, generated PNGs in `.gitignore`

## Dataset v1 (deprecated)
- [x] 50 episodes collected on Windows GTX 1650 (single red→blue instruction)
- [x] Push `franka_libero_stacking_v1` to HF Hub (`VivekSai07/franka_libero_stacking_v1`, branch `v3.0`)
- [x] Train + infer on v1 → **0% success rate** (root cause: single instruction, no language grounding)

## Dataset v2
> Root cause of v1 failure: single fixed instruction, 50 episodes, deterministic trajectories.
> Full analysis in DATASET_DESIGN.md.

- [x] Document v1 failure analysis and v2 design rationale → `DATASET_DESIGN.md`
- [x] Upgrade `data_collection/collect_libero_dataset.py`:
  - [x] 12 instruction variants (all colour-pair combinations of red/green/blue/yellow)
  - [x] 300 episodes target (~25 per variant)
  - [x] Variable distractor count per scene (0–2 extra cubes)
  - [x] Wider spawn range: x=[0.20,0.60], y=[-0.40,0.40]
  - [x] Min cube clearance: 15 cm (up from 12 cm)
  - [x] ±5 mm Gaussian jitter on approach trajectories
  - [x] Success threshold tightened: xy < 2 cm (was 3 cm)
  - [x] Success verification uses instructed colour pair (not hardcoded red/blue)
  - [x] 512×512 images (matches SmolVLM2 max_image_size.longest_edge=512)
  - [x] Rolling ETA + live `\r` status line (phase bar, sim time, ETA overwrites in place)
- [x] Collect 300 episodes — completed overnight (~3.8h, avg 41s/ep)
  ```bash
  docker compose run --rm collect
  ```
- [x] Push `franka_libero_stacking_v2` to HF Hub (`VivekSai07/franka_libero_stacking_v2`, branch `v3.0`)
  ```bash
  docker compose run --rm push
  ```

## Training v2
- [x] Hyperparameter analysis based on actual model config (SmolVLM2-500M backbone, 1.2GB weights):
  - batch_size: 24 → **32** (safe for 32GB VRAM, est. ~25GB peak)
  - steps: 20K → **80K** (3 epochs → 16 epochs; 3 epochs confirmed undertrained)
  - LR: keep default 1e-4 (no linear scaling for fine-tuning)
- [x] Train on v2 dataset — **completed overnight (~8h, batch=32, steps=80K)**
  ```bash
  docker compose run --rm train   # auto-resumes from latest checkpoint if interrupted
  ```
- [x] Model pushed to HF Hub: `VivekSai07/smolvla_franka_libero_stacking_finetuned_b32_s80k`
- [x] Monitor via W&B — project `smolvla_franka_project`

## Inference v2
- [x] Fix inference script mismatches vs training:
  - [x] Renderer: 480×640 → 512×512
  - [x] Instruction: removed "perfectly" to match training format exactly
  - [x] Colour combos: expanded from 2 (red↔blue only) to all 12
  - [x] Spawn range + cube clearance: matched to training (0.15m clearance)
  - [x] MAX_STEPS: 600 → 1200 (40s window for suboptimal paths)
  - [x] Video folders: `inference_videos/<policy_name>/run_N/` naming convention
- [x] Fix server URL: `server:8000` → `localhost:8000` (client uses `network_mode: host`)
- [ ] Run 30-episode evaluation and record success rate
  ```bash
  # Terminal 1
  docker compose up server   # wait for "✅ Policy & Processors loaded!"

  # Terminal 2
  xhost +local:docker
  docker compose run --rm client
  ```
- [ ] Review videos in `inference/inference_videos/smolvla_franka_libero_stacking_finetuned_b32_s80k/`
- [ ] Compare success rate vs v1 (0%)

## Next Steps (if inference success rate still low)
- [ ] Collect dataset v3 — 600 episodes (50 per instruction variant) for reliable placement
  - Update `docker-compose.yml` collect command: `--episodes 600`
  - Retrain with steps=120K
- [ ] Try `smolvla_base` (not fine-tuned on LIBERO) — compare zero-shot vs fine-tuned
