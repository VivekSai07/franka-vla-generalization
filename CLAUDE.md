# CLAUDE.md

## Current work: read first

Read [docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md) before executing
commands. It supersedes historical defaults below. The validated 420-episode
pilot is complete; a fresh 20,000-update local SmolVLA run is now authorized
and launched (see current status). Do not duplicate it or use the historical
auto-resuming train service. Never resume a pre-fix checkpoint. Research findings
and compatibility caveats are indexed there for humans and other agents.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fine-tuning SmolVLA (`HuggingFaceVLA/smolvla_libero`) on a custom cube-stacking dataset collected in MuJoCo with a simulated Franka Panda arm. Everything runs inside Docker — there is no host Python environment; do not try to `pip install` or run scripts directly on the host.

## Commands

All commands run via `docker compose run --rm <service>` from the repo root. There is no test suite, linter, or build step — this is a research/robotics pipeline, and correctness is verified by running collection/training/inference and inspecting output.

```bash
docker compose build                    # build the image (CUDA 12.8.1 + PyTorch nightly cu128 + lerobot[smolvla]==0.4.4 + MuJoCo + Pinocchio)
docker compose run --rm collect         # collect 300 episodes into datasets/franka_libero_stacking_vN (auto-versioned)
docker compose run --rm push            # push local dataset to HF Hub, branch v3.0 (required — lerobot reads codebase_version from this branch)
docker compose run --rm train           # fine-tune; auto-resumes from latest checkpoint in outputs/train/.../checkpoints if one exists
docker compose up server                # inference: FastAPI policy server on :8000 (wait for "✅ Policy & Processors loaded!")
docker compose run --rm client          # inference: MuJoCo client, 30 eval episodes, needs `xhost +local:docker` first, server must already be up
```

Monitoring/debugging a running container (collect and train are long-running background jobs, often left in tmux overnight):
```bash
docker ps --filter ancestor=smolvla:latest          # verify exactly one container is running (duplicates halve throughput)
docker logs --tail 5 <container_id>                 # check current episode/step + ETA
```

Credentials: copy `.env.example` to `.env` and fill in `HF_USERNAME`, `HF_TOKEN`, `WANDB_API_KEY`, `HF_REPO_ID`. `.env` is gitignored.

## Architecture

**Two-stage pipeline, decoupled by the HuggingFace Hub:** dataset collection and training don't talk to each other directly — `collect` writes to local `datasets/`, `push` uploads it to the Hub, `train` pulls it back down by `repo_id`. Same for models: `train` pushes the fine-tuned checkpoint to the Hub under `policy.repo_id`, and inference's `server` loads it back by that same repo_id.

**Inference is a server/client split across two containers**, not a single script:
- `inference/policy_server_libero.py` — GPU container, loads the fine-tuned SmolVLA policy, exposes `/predict` over FastAPI on port 8000.
- `inference/franka_inference_libero.py` — separate container, runs the MuJoCo sim, queries the server for an action at 30 Hz, saves per-episode MP4s + a summary.
- The client uses `network_mode: host` and talks to the server via `SERVER_URL=http://localhost:8000/predict`, not the Docker service name — this is intentional (see Troubleshooting in README) and will silently break if changed to `server:8000`.

**Robot control stack (`src/`) is layered independently of the VLA policy** — it's the same MuJoCo/Pinocchio machinery used both to generate scripted demonstrations during collection and to execute predicted actions during inference:
- `src/robots/franka_panda/robot.py` + `config.py` — FK/Jacobian via Pinocchio, joint config, scene path. `model/` holds the MuJoCo XML scenes and mesh assets (several scene variants exist, e.g. `scene.xml` vs `smolvla_scene.xml` — check which one `config.py`/`SCENE_PATH` actually points at before editing scenes).
- `src/controllers/ik_controller_m2.py` — inverse kinematics; `grasp_controller.py` — contact-aware gripper control.
- `src/planners/trajectory_planner.py` — task-space cubic trajectory planner; `rrt.py` — sampling-based planner.
- `src/tasks/franka_pp/` — task definitions (pick-and-place / stacking) that compose the above into an episode.

**Dataset generation is instruction-diversity-driven, not just trajectory generation** — `data_collection/collect_libero_dataset.py` samples one of 12 colour-pair pick/place instructions per episode, randomizes cube count (2–4) and spawn positions, and adds jitter to approach trajectories, specifically to force the VLM backbone to ground language to vision rather than memorize a fixed visuomotor reflex. This is a deliberate fix for a documented v1 failure (0% inference success with a single fixed instruction) — see `DATASET_DESIGN.md` before changing collection parameters (instruction set, clearance, spawn range, success threshold); each parameter there is tied to a specific failure mode, not an arbitrary default.

**Image size is coupled to the model, not arbitrary**: cameras render at 512×512 to match SmolVLM2's `max_image_size.longest_edge=512`; other resolutions add black padding that changes what the model sees between collection/training and inference — keep collection, training data, and inference client in sync if this changes.

**Training config lives in `docker-compose.yml`**, not a separate config file — the `train` service command has the full `lerobot-train` invocation including `--policy.expert_width_multiplier=0.5` (required — the pretrained smolvla_libero expects exactly 2 cameras) and the fresh-vs-resume auto-detection logic (checks `outputs/train/.../checkpoints/` for an existing checkpoint before deciding whether to pass `--resume=true`).

See `TODO.md` for current project status/progress and `DATASET_DESIGN.md` for the full v1→v2 failure analysis and rationale behind current collection parameters.
