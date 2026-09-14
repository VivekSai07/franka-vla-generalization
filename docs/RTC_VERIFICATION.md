# RTC machine: verify the action/observation alignment fix

**Read this first, before running a full data collection.** The three
collectors in `data_collection/` were recently fixed for a bug where each
saved frame's `action` was the *previous* control tick's stale commanded
joint target instead of the action actually computed and sent that tick —
see `docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md`
for the full analysis and `docs/superpowers/plans/2026-09-14-fix-action-observation-alignment.md`
for exactly what changed.

That fix was written and reviewed on a machine with no Docker/GPU access,
so it has only been verified by static code review — **nobody has actually
run MuJoCo with the reordered loop yet.** Do not jump straight into a full
300-episode collection. Run the small verification below first.

## Hardware note

This machine is an **RTX 6000 Ada** (Ada Lovelace, sm_89) — not Blackwell
(that label applies to RTX 50-series/B-series GPUs, sm_120). This doesn't
require any changes: the Dockerfile's PyTorch nightly cu128 build covers
sm_89 fine (it's a superset of what stable PyTorch already supports). Don't
expect the sm_120-specific warning the README describes for RTX 5090 — that
just doesn't apply here.

## Steps

```bash
# 1. Clone/pull the repo (uni origin is the canonical remote)
git clone https://github.tik.uni-stuttgart.de/st195221/smolvla_franka.git
cd smolvla_franka
git log --oneline -1
# Expect HEAD at or after commit 0709993 "Add CLAUDE.md, action/observation alignment..."

# 2. Set up credentials
cp .env.example .env
# Fill in HF_USERNAME, HF_TOKEN, WANDB_API_KEY, HF_REPO_ID in .env

# 3. Build the image (CUDA 12.8.1, PyTorch nightly cu128, lerobot[smolvla]==0.4.4, MuJoCo, Pinocchio)
docker compose build

# 4. Verify the NVIDIA runtime is visible to Docker
docker run --rm --runtime=nvidia nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi

# 5. Collect a SMALL test batch — 2 episodes only. Do NOT run the full 300-episode
#    collection yet.
docker compose run --rm collect python data_collection/collect_libero_dataset.py --episodes 2 --headless
# Note the printed "Initializing dataset: <path>" line at the start of output —
# copy the exact directory name, e.g. datasets/franka_libero_stacking_v3

# 6. Run the verification script against that test batch
docker compose run --rm collect python data_collection/verify_action_alignment.py \
  --root datasets/<paste_the_exact_dir_from_step_5> \
  --repo-id <same_dir_name>
```

## What to look for

- **Step 5** should end with `Data collection complete: 2 episodes saved.`
  It may print a few `FAILURE (...) — discarding.` retries first — that's
  normal grasp-failure retry behavior, not a bug.
- **Step 6** must print:
  ```
  PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.
  ```
  and exit 0.

  **If it prints `FAIL` instead: stop.** Do not proceed to full collection.
  Report the full output — something in the reorder needs re-checking
  against the spec doc referenced above.

## Only after a PASS

Proceed to the real collection (this is MuJoCo-sim-bound, not GPU-bound,
so it takes hours regardless of GPU speed — expect roughly the same
~3-4h ballpark as before):

```bash
docker compose run --rm collect
```

This uses the `collect` service's default command in `docker-compose.yml`
(300 episodes). Once collection finishes, push the dataset
(`docker compose run --rm push`) and retrain (`docker compose run --rm train`)
as documented in `README.md` — but **do not resume from any existing
checkpoint** in `outputs/train/`; those were trained on data collected
before this fix and are not valid to continue from.
