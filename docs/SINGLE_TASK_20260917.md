# Single-task capability experiment — 2026-09-17

## Completed: capability gate failed

Container `smolvla-single-task-5000` exited **0**, no OOM, from18:06:53 to20:07:11
UTC (about2h00m18s). All5,000 updates and13rollouts executed. Successful program
completion is not task success: **seen1/9, held-out0/4**, below approved8/9 and3/4.
`provisional_capability` is false. No expansion, retry or additional training.

Fresh completion check verified5,000 consecutive finite loss/gradient/LR/time
records, all six model checkpoint hashes, selected005000 data/source/cache/stats
bindings, unchanged training-byte inventory and recomputed evaluation summaries.
Every checkpoint's recorded GPU reload error is0 with exact FP32 trainable weights.
Completion inspection did not rerun GPU inference or full physical replay.

Final offline fixed-probe results:

| Metric | Train | Held-out |
|---|---:|---:|
| Full50 arm MAE(rad) |0.011930|0.079761|
| First10 arm MAE(rad) |0.009787|0.041201|
| Gripper binary accuracy |97.96%|92.80%|

These probe sets include frame0 unlike the broad pilot's original validation
probes; do not directly rank experiments by their aggregate MAE. Held-out
full-chunk error is approximately6.7times training error in this run.

Recorded-trace stage decomposition:
- Seen: six trials contacted and lifted the target above5cm; one stacked
  (episode7), five failed after lifting (episodes0,1,2,4,5).
- Seen: two had only brief bilateral contact without5cm lift (episodes6,8);
  one had no target bilateral contact (episode3).
- Held-out: all four had no target bilateral contact; none exceeded5cm target lift.
Contact/lift are proxies, not a guarantee of a stable load-bearing grasp. These
counts are from saved traces, not a new all-object replay audit.

Interpretation: narrowing the task produced substantial seen-scene grasp/lift
behavior, but did not establish reliable stacking or held-out grasping. Both
execution beyond the grasp and layout generalization remain problems. Removing
instruction ambiguity is not sufficient. This was not a matched causal ablation
of the previous run, so improvement cannot be attributed to task restriction alone.

The fixed-budget stopping rule now applies. Preserve outputs and stop for a
decision; do not resume planner development, collect more data, extend epochs,
switch architecture or run additional rollouts automatically. Any next proposal
must distinguish the seen-scene placement/continuation failure from the held-out
approach/grasp failure rather than treating both as one language-grounding issue.

Evidence: [training result](../outputs/train/smolvla_single_task_20260917/result.json),
[evaluation result](../outputs/train/smolvla_single_task_20260917/evaluation/result.json).
The parent training report retains `task_success_evaluated:false` because it was
finalized before evaluation and byte-frozen. The separate complete evaluation
report is authoritative for the13 task outcomes; do not rewrite either artifact.

The launch text below is historical; this job is no longer running.

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