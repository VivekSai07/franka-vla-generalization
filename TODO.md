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

## Action/Observation Alignment Fix (2026-09-14) — READ THIS FIRST if inference still fails to pick

Root cause found: all three data collectors (`collect_libero_dataset.py`, `collect_v2_dataset.py`,
`collect_stacking_dataset.py`) recorded each frame's `action` as the *previous* control tick's
stale commanded joint target instead of the action actually computed and sent that tick — so every
episode in `franka_libero_stacking_v1` and `v2` has `(state_t, action_{t-1})` instead of the correct
`(state_t, action_t)`. Confirmed against lerobot's own `lerobot_record.py` reference loop
(`observe → act → send → save`, always in that order). This is a strong candidate for why the
fine-tuned model in v1/v2 "reaches around" the cube but never picks it — see
`docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md` for the full
analysis and `docs/superpowers/plans/2026-09-14-fix-action-observation-alignment.md` for exactly
what changed.

- [x] Fix the reorder bug in all 3 collector scripts so `action` matches what was actually sent
- [x] Add `data_collection/verify_action_alignment.py` — sanity-checks that a collected dataset's
      `action_t` is closer to `state_{t+1}` than to `state_t`
- [ ] **On the actual GPU/Linux collection rig** (this fix was authored on a Windows dev machine
      with no Docker/GPU access — none of this has been runtime-verified yet):
  ```bash
  docker compose run --rm collect python data_collection/collect_libero_dataset.py --episodes 2 --headless
  docker compose run --rm collect python data_collection/verify_action_alignment.py \
    --root datasets/<the_new_versioned_dir> --repo-id <same_name>
  ```
  Confirm it prints `PASS`. If `FAIL`, something in the reorder is still wrong — re-check
  `collect_libero_dataset.py`'s per-tick loop against the design doc before going further.
- [ ] Once verified: re-collect a fresh `franka_libero_stacking_v3` (or overwrite v2 collection
      logic is unchanged otherwise — same 12 instructions, same episode count) with the fixed
      collector, push to HF Hub (`v3.0` branch), and retrain from scratch (previous checkpoints
      were trained on mislabeled data — do not resume from them)
- [ ] Re-run the 30-episode inference evaluation and compare success rate against v1/v2 (both 0%
      or near-0%) to confirm whether this was in fact the root cause
- [ ] If success rate is still poor after this fix, the next suspects (not yet investigated) are:
      the one-tick control latency between `d.ctrl` write and `d.qpos` read at the top of the next
      loop iteration, and whether SmolVLA's `delta_timestamps`/action-chunking config expects a
      different temporal convention than a single-step `(state_t, action_t)` pair
