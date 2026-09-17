# Single-task capability experiment — 2026-09-17

User approved and requested launch of this fixed experiment. Planner development
and the blocked paired collection/training program are paused. No new expert
data, difficult-layout repair or model architecture change is included.

## Approved budget

- Existing validated pilot subset only: red on green, exactly two cubes.
- Train episode IDs **0–8** (9 demonstrations, 4,752 frames, 4,311 full chunks).
- Held-out IDs **300–303** (4 demonstrations); never included in training statistics.
- One fresh pinned SmolVLA base, **5,000 optimizer updates**, effective batch32
  (micro8 x accumulation4), FP32 trainable/Adam state with BF16 autocast, frozen VLM.
- LR:500-update warmup to1e-4, cosine decay to1e-5. Seed20260917. Stock loss,
  complete50-action chunks, two512RGB cameras and9-state/8-action contract unchanged.
- Statistics recomputed on this exact training subset. Sampling repeats the
  subset; 160,000 sampled chunks are approximately37.1 passes, not new demonstrations.
- Final checkpoint only; automatic **13 rollouts**, nine seen and four held-out,
  each once with40-second timeout andten-action horizon. Seeds20260917+episodeID.
- Provisional expansion gate: >=8/9 seen and>=3/4 held-out correct successes.
  Four held-out scenes do not establish general reliability or language grounding.
- No retries, automatic extensions, final-bank tests, uploads or alternate policies.

This reuses existing expert-success-filtered scenes across their original spatial
range. It simplifies instruction and distractors, not the robot geometry or
continuous workspace. It is not the full multi-instruction benchmark, a matched
causal ablation, or a promise of success. Original planner semantics remain in
the existing recorded labels, which replay correctly; no collector is rerun.

## Verification and launch

219 CPU tests passed. Independent review identified two admission gaps, corrected
with seven regressions: cache inventory must match successful scratch preflight;
absolute reload prediction error must meet1e-5 rather than relying on relative tolerance.

GPU scratch container `smolvla-single-task-preflight` exited0/noOOM: two finite
optimizer updates, both checkpoints FP32-exact with prediction error0. No scratch
rollouts; those weights are not resumed. Old bound sources/data unchanged.

Full detached container: `smolvla-single-task-5000`.
ID: `25c3d0bbb3b4682d81371dcba715363efb29d5a1768b02eb33ad03de67d849b3`.
Output: `outputs/train/smolvla_single_task_20260917`.
Pinned offline image, network disabled, projectread-only, trainingoutputs writable.
No credentials loaded. Modeltraining then final-checkpoint evaluation are sequential
in the same runner; either infrastructure failure stops the job.

Launch is not completion. Check actual result/container status before reporting
success. Keep the remote host/Docker running; SSH can disconnect. Approximate
fulljobcompute1.5–2.5hours based on earlier throughput and13rollouts, excluding
unexpected blockers; initial2–4hour estimate included implementation/preflight.

Do not change experiment source during execution. At completion report training,
seen-scene task success, held-out success and integrity separately. Regardless
of outcome, stop for a decision rather than extending the run automatically.