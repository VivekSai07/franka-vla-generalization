# Single-task implementation plan

**Goal:** One fresh pinned-base red-on-green run, then exactly thirteen saved-reset evaluations. This task implements and CPU-tests only: no GPU preflight, training, rollouts, commits, old-source edits, or credentials.

**Architecture:** Reuse the tested pilot statistics, chunk boundaries, microbatch checks/backward, fixed probes, precision promotion/reload and checkpoint writer. A small explicit trainer replaces only the hard-coded 20k orchestration. A generic initialization adapter explicitly follows paired_instruction_probe.derive_train_state (which rejects validation), then calls evaluate_pilot.restore_scene/run_trial unchanged. No monkeypatching, text rewriting, new framework, paired-data dependency, or altered physics/score.

**Stack:** Existing pinned Docker image sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc; installed LeRobot/PyTorch/MuJoCo; Python/unittest inside Docker only.

## Exact contracts
- Base lerobot/smolvla_base revision c83c3163b8ca9b7e67c509fffd9121e66cb96205, cached/offline/local-only. Seed 20260917. Never resume or autotune.
- Original validated franka_pilot_20260915 dataset; full require_replay_pass before/after training and evaluation. Old hash lists remain unchanged; independently bind every new Python module/test and reused source.
- Train episode IDs 0..8 only, validation 300..303 only, verified against manifest split membership, instruction, pair, cube count and raw episode metadata. 4752 train frames / 4311 complete 50-action starts. No frame-random split. Train-only statistics.
- Policy inputs: state9, two 3x512x512 images and exact task only. Actions8; no privileged fields; future actions are targets only. Fixed train and validation probes at offsets 0,30,60,107,347, initial and every1000 updates; diagnostic metrics never select checkpoints.
- 5000 optimizer updates; micro8 x accum4 =32; FP32 trainables and AdamW storage, BF16 training autocast; frozen VLM, train expert/state projection; unchanged stock loss including padded action dimensions; chunk50/horizon10.
- LR update1=2e-7; update500=1e-4; cosine floor1e-5 at5000. AdamW betas(.9,.95), weight decay1e-10, norm clip10 as pilot. Finite loss/gradient/parameter checks; optimizer precision at first/save/end. Checkpoints1,1000,2000,3000,4000,5000. Progress every20; disk budget35GiB at entry and per-save headroom; no automatic restart.
- Exclusive full output /vla/outputs/train/smolvla_single_task_20260917. Separate exclusive scratch sibling smolvla_single_task_20260917_preflight: exactly2 updates at identical batch/precision plus save/reload, zero rollout calls. Full run requires matching completed preflight, starts anew from base, never its weights.
- Final checkpoint ONLY. Bind complete training report, saved marker, model/processor/config/stats bytes and source snapshot; preserve all training bytes during evaluation. Exact FP32 reload helper; frozen saved processors, no stats recomputation for inference.
- Adapt only allowed original demo NPZs into evaluation/initial_states, with source SHA, initial integration state, first qpos, scene digest and geometry verification. Preserve source identities/provenance; never write original data.
- Exactly IDs0..8,300..303, seed20260917+episode_id. Pass **episode_id+1** as actual baseline ordinal (baseline SEED=20260916); report that ordinal, never a consecutive evaluation index as seed identity. 1200 control ticks /40 simulated seconds, strict30-tick released-stack score unchanged. No scene-bank/final-bank expansion.
- Report seen and held-out separately; provisional capability flag requires >=8/9 and >=3/4, not statistical generalization. Stop on nonfinite/error, preserve partial report and independent failure record; no retries.

## Tasks / verification sequence
1. Add tests before implementation in this package. RED: missing module; subsequently test exact subset/leak rejection, row boundaries/probes, LR endpoints, exclusive paths,13trial seeds, initialization SHA/split/scene failures, final-only checkpoint gate, preflight dispatch and immutable training bytes.
2. Implement contract helpers, runner and evaluation adapter here only. Reuse helper implementations explicitly, never call old trainer/main or baseline hard-coded020000 preflight.
3. Run new synthetic CPU tests; real read-only dataset/gate/reset integration checks (no policy construction, rendering, stepping, CUDA or optimizer run). Verify both train and held-out initialization adapters.
4. Run all root168 tests plus adaptive-placement20 and new tests with CPU-only/network-none pinned Docker, read-only repository, temporary cache/output. Hash protected tracked sources before/after; no edits outside this package.
5. Record exact counts, CLI and limitations in README. Main review precedes all GPU execution; no claim of learned capability from CPU tests.

## Alternatives rejected
Calling the old trainer changes approved counts/schedule/probe selection. Monkeypatching its globals or rewriting its source obscures the experiment boundary. A new generalized training framework is unnecessary. Explicit orchestration plus existing helpers is the minimal isolated implementation.