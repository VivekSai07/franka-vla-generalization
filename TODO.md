# SmolVLA / X-VLA Franka TODO

Live plan is below. The original `docker-compose` + `franka_libero_stacking_v1/v2` work is
preserved as an archive at the bottom; see [README](README.md) and
[docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md) for current status.

---

## Phase A — maximise the current single-task stacking problem

Goal: establish the ceiling on one axis of variation (cube position in envelope F7) before
adding any new axis. Ordering matters — a failure with two new axes is not attributable.

### A0. Finish what is running
- [x] Cross-evals complete (local-varied checkpoints on `broad_holdout`, 72 trials each)
- [x] Record the full 2×2 + confidence intervals in README
- [x] Note the `metadata_binding_reproducible: False` caveat beside the SmolVLA cross-eval figure

### A1. Measure the checkpoint curve BEFORE training longer
Rationale: X-VLA broad saw **0.50 epochs** (40,000 chunks / 79,840 starts) versus 1.67 epochs on
local-varied — we tripled the data and held updates at 5000. Train/val arm MAE 0.01655/0.01748
(5.6% gap) shows underfitting, not overfitting. But confirm the trend before spending GPU.

- [x] Add a `--checkpoint` option to the broad eval packages. Currently 5000 is hard-pinned in
      four places: `offline_probes` `checkpoint_update=`, `verify_final`'s `completed_updates`
      and `selected_checkpoint_update` requirements, and `checkpoint_directory(5000)`.
      `verify_final` must still pin the chosen checkpoint's `model_sha256` to its training marker.
- [x] Evaluate X-VLA broad checkpoints **3000** and **4000** on the identical 72-trial holdout
- [x] Plot success vs updates (1000/2000/3000/4000/5000 as available)

**Decision gate: PASSED (2026-09-23).** 0.0% @3000 -> 20.8% @4000 -> 68.1% @5000, still
accelerating at the final checkpoint, val/train 1.01-1.06 (no overfitting). Proceed to A2.
Caveat recorded: part of the final jump is the cosine anneal, so do not extrapolate linearly.

### A2. Train to matched exposure (only if A1 says climbing)
- [x] New package (`xvla_broad_workspace_long`) — did NOT edit the existing one. `recipe()`
      hardcodes `max_optimizer_updates=5000` and `schedule.total=5000`; changing them alters
      `SOURCE_FILES` hashes and would invalidate the sealed 5000-update run.
- [x] **16,700 updates** to match local-varied epoch exposure (79,840 starts × 1.67 / 8)
- [x] Fresh run from base, not a resume
- [x] Checkpoint schedule extended so the curve stays measurable
- [x] Re-evaluated on the same holdout: **62/72 = 86.1%** vs 49/72 = 68.1% (Fisher p = 0.0165)
- [ ] Repeat for SmolVLA (2.09 epochs currently) — **hold**; see A4, the remaining failure is not
      obviously exposure-limited and a SmolVLA run would cost ~4 h to answer a different question

**Outcome (2026-09-24): exposure was the deficit, and it is now largely paid off.** Two findings
changed the plan and are recorded in [docs/RESULTS.md](docs/RESULTS.md) and
[docs/FAILURES.md](docs/FAILURES.md) #21–#23:
- the checkpoint curve is **non-monotonic** (25.0% → 0.0% → 86.1%) while offline MAE improves
  monotonically — offline metrics are invalid as capability proxies
- all 10 remaining failures are a single placement offset of roughly one cube width, and the
  expert demonstrations are clean (max 1.98 cm, 0 episodes > 2 cm), so it is policy-side

### A3. Close out
- [x] Update docs with the exposure finding and the curve
- [x] Evaluate checkpoint **12,000** — the LR-matched comparison against the short run's 4,000.
      **Result: 51/72 = 70.8% vs 20.8%, p = 2.2e-9.** Exposure drives capability, not the anneal;
      at matched epochs a 4.4× LR difference changes nothing (p = 0.69).
- [ ] Decide whether the 65/72 gate is right for this protocol — argue it on its own terms,
      not by fitting it to whatever number we got. Note 62/72 misses by three trials and every
      miss is the same 4.5 cm offset, so the gate question is now entangled with A4.

### A4. The 4.5 cm placement offset (new — highest-value open question)

**Design spec: [docs/superpowers/specs/2026-09-24-placement-offset-design.md](docs/superpowers/specs/2026-09-24-placement-offset-design.md)**
Approved ordering: **Tier 2 → Tier 1 → B1 collection design.**

All 10 failures at 86.1% land the cube one cube-width beside the target. Decomposed about the robot
base, the systematic component is **tangential** (a ~5° bearing error) rather than a reach error.
Authoritative magnitudes from the 128-trial T2a run: successes +0.150 cm (sd 0.715, n=96), failures
+3.893 cm (sd 0.723, n=32). Failure is layout-specific, not a global bias.

**Cause: action-sampling multimodality.** 6 of 8 failing layouts are split decisions — same scene,
same weights, different sampling seed, opposite outcome. `modeling_xvla.py:252` draws fresh
`torch.randn` per chunk over `num_denoising_steps=10`; `predict_action_chunk` accepts injected
`noise`, which is the hook for every Tier 1 fix.

Ruled out by test, not assumption:
- **imitating a sloppy expert** — refuted; all 200 demos clean (max 1.98 cm, 0 above 2 cm)
- **unreachable far layouts** — refuted and inverted (failing 0.424 m vs solved 0.472 m reach)
- **sparse training coverage** — refuted; identical 4-D NN distance, τ = 0.000
- ⇒ **more data will not close this gap**; this is the primary input to B1 planning

#### Tier 2 — measurement (first)
- [x] **T2a** mode-probability run: 128 trials complete (rc=0, 223 min). **Control 64/64 = 100%**,
      failing group 32/64. Three layouts (008, 017, 029) draw the wrong mode 7 times in 8; the bad
      mode is **absent** from control layouts. No global bias (successes +0.15 cm). No geometric
      feature separates the hard layouts.
- [x] **T2b** seed budget **established**: 2 seeds cannot distinguish 12.5% from 87.5% per layout
      (024 and 029 both read 1/2). **≥8 seeds per layout is required for any per-layout claim.**
      Aggregate rates survive at 2 seeds. Lock this into the B1 protocol before collecting.

#### Tier 1 — inference-time (downgraded by T2a)
- [ ] **T1b** best-of-N — projected 62 → ~65 of 72, and cannot help the three bad-dominant layouts.
      Marginal; **not a blocker for B1**. Do only if the gate matters more than N× inference cost.
- [ ] **T1a** `num_denoising_steps` 10 → 30; 16-trial probe (~40 min) — cheapest remaining probe
- [ ] **T1c** temporal ensembling — **averaging is unsafe**: the mean of the two modes lands inside
      the empty band and fails everywhere. Cluster-then-pick-largest only.
- Each is a separate protocol in its own package; never compare to the sealed 86.1% unless the
  layout roster and seeds are identical

#### Tier 3 — data design for B1 (low expected value here)
- [ ] Audit expert approach-direction consistency against the sealed dataset (CPU-only, cheap) —
      the only data-side lever with a plausible mechanism
- [ ] Terminal-phase supervision density — hypothesis only
- [ ] **Not planned:** more episodes, denser layout coverage

Only after Tier 2 + Tier 1: decide whether the 65/72 gate is right, and finalise B1 collection.

---

## Phase B — generalisation axes (deferred until Phase A closes)

One axis at a time, each with its own dataset and its own holdout.

### B1. Language grounding (first axis — cheapest, currently unmeasured)
All 200 episodes share one instruction, so the language pathway is currently unverified; the
model may be ignoring text entirely.
- [ ] Collect reversed-instruction episodes (green onto red) over the **same** layouts
- [ ] Test: does behaviour change with the instruction in an identical scene?
- [ ] This is the minimal grounding test — prefer it over 12-colour randomisation

### B2. Distractors (second axis)
- [ ] Add 1–2 non-target cubes, target pair unchanged
- [ ] Measures distractor rejection independently of language

### B3. Full colour randomisation (third axis)
- [ ] Only after B1 and B2 each have a measured result
- [ ] This is closest to the original v1/v2 design; note v2's evaluation was never completed, so
      there is no clean verdict on it — do not treat it as a known failure

### Not planned
- **Bounding-box / segmentation-mask injection.** Would likely raise numbers but hands the model
  the spatial answer, defeating the goal of the vision encoder learning spatial grounding.
  Also incompatible: our models are not autoregressive token decoders (SmolVLA uses flow
  matching, X-VLA a soft-prompt/action-hub head), so coordinate-text injection does not transfer
  from RT-2/OpenVLA. A 4th mask channel would change the observation schema and force
  recollection of every sealed dataset.

---

## Known debt

- [ ] `experiments/{xvla,smolvla}_local_varied/README.md` state seal `f8b10a49…`; actual is
      `ae8828ea…`. Hash-bound — cannot be corrected without invalidating those runs' bindings.
- [ ] `experiments/broad_workspace_data/README.md` still reads as pre-collection. Same constraint
      (`scenes.provenance()` appends it).
- [ ] SmolVLA local-varied metadata binding is unreproducible (pre-re-seal). Cross-eval uses a
      data-level binding instead; documented in the result JSON.

---

## Before launching any long run — checklist

Derived from the failures in [docs/FAILURES.md](docs/FAILURES.md). Each item
cost real GPU time at least once.

- [ ] **Package README updated.** It is in `SOURCE_FILES`; a copied one describing the old recipe
      gets hashed into the run's provenance (#17, and the earlier `sed`-mangled X-VLA README).
- [ ] **No constant duplicated between contract and runner.** Grep the runner for the horizon,
      checkpoint schedule and any literal that also lives in `recipe()` (#15).
- [ ] **Regression tests fail against the unfixed code.** Confirm with `git stash`, not by
      assumption (#10, #15).
- [ ] **Nothing hash-bound is edited while a job runs.** Check `SOURCE_FILES` and
      `provenance()` membership first; remember `provenance()` includes READMEs (#8, #17).
- [ ] **Worktrees are at the intended commit.** `git log -1` inside the worktree (#18).
- [ ] **Shared helpers are not edited to accommodate one package.** `xvla_single_task/contract.py`
      is bound by four contracts; override locally instead.
- [ ] **All gitignored dirs the container needs are mounted** — `outputs`, `datasets`, `.cache`.
- [ ] **Run `--check` and `--preflight` first.** Editing any `SOURCE_FILES` member after `--check`
      invalidates it and forces a full ~1h re-run (#16).
- [ ] **Poll actively.** `docker wait` has silently failed to notify here; do not assume
      completion will be reported (#19).
- [ ] **Forked a package? Repoint its tests.** Copied tests keep importing the source module and
      will pass while testing the wrong code; re-derive constants from the contract rather than
      copying the literal (#20).
- [ ] **Checkpoint schedule chosen from the LR curve, not round numbers.** To compare two runs
      mid-training you must match *schedule phase*, not step count — update 4,000 meant 80%
      annealed in one run and 24% in the other, making the comparison worthless (#22).
- [ ] **Plan for rollout evaluation, not proxies.** Loss and MAE improved monotonically through a
      run that collapsed to 0/72; budget the GPU hours for real rollouts (#21).

## After every finding — documentation pass

Results, learnings and mistakes all trigger this, in the same turn they are produced:

- [ ] Grep the live docs for the specific figures and claims the finding changes
- [ ] Update **every** hit — README.md, docs/RESULTS.md, docs/FAILURES.md, docs/BENCHMARK_STATUS.md,
      TODO.md, the relevant spec
- [ ] Where a claim is superseded rather than wrong, keep the history: what was believed, what
      replaced it, why the first reading was mistaken
- [ ] Add recurrable mistakes to docs/FAILURES.md — including analysis errors, not just pipeline ones
- [ ] Re-check relative links

---
---

# ARCHIVE — original docker-compose pipeline (last accurate ~2026-09-14)

Historical record of the `franka_libero_stacking_v1/v2` HF datasets and the v1 0% /
language-grounding findings. Superseded by the `experiments/` packages above.

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
