# Small reusable-benchmark validation — 2026-09-15

## Scope

Six successful demonstrations, exact recorded-action replay, original alignment
test, and a bounded SmolVLA learning pipeline. This is not the 300-demo pilot
and does not establish benchmark-quality policy performance. No upload or
pre-fix checkpoint resume is permitted.

## Environment and provenance

- Branch: `feat/reusable-validation`; baseline commit `26b5e31`.
- Docker image: `smolvla:rtc-offline`, ID
  `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
- Same image as the historical alignment smoke: LeRobot0.4.4,
  MuJoCo3.13.0, PyTorch2.7.1+cu128, Pinocchio3.8.0, OSMesa rendering.
- Base policy revision: `lerobot/smolvla_base` at
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- VLM tokenizer/config revision:
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Full policy snapshot supplies the VLM weights; no separate VLM weight
  download is used. Public downloads succeeded; training is network-disabled.
- Strict pretrained loading passed with9-D state,8-D action,two cameras,
  expert width0.75:450,046,176 parameters,99,880,992 trainable.
- Dataset manifest records source/asset hashes, runtime versions,camera
  configuration,state/action units and full MuJoCo integration-state traces.

## Collection outcome

Completed root: `datasets/franka_benchmark_validation_20260915_retry`.
Container: `smolvla-validation-6-retry`; normal exit0.
Six successes from eight attempts,3,120 frames at nominal30Hz,520 per episode.
Two failed placement attempts were discarded and logged. Replay-only ground
truth stays in sidecars and is never fed to a policy.

| Episode | Split | Instruction | Cubes | Attempt seed |
|---:|---|---|---:|---:|
| 0 | train | red on green | 2 | 20260915 |
| 1 | train | green on blue | 3 | 20260917 |
| 2 | train | blue on yellow | 4 | 20260919 |
| 3 | train | yellow on red | 2 | 20260920 |
| 4 | validation | red on blue | 3 | 20260921 |
| 5 | validation | green on red | 4 | 20260922 |

Conditions are deterministic; retry keeps the scheduled pair/count but changes
the seed and layout. This tiny split is deliberately integration-only, not a
representative per-instruction validation set. Failed attempts do not consume
a successful episode index. The manifest is marked complete only after
LeRobot finalization and exact count checks.

The first foreground attempt was interrupted with exit130 during terminal
activity. Its incomplete root `datasets/franka_benchmark_validation_20260915`
and container `smolvla-validation-6` are retained; never use it for training.
The retry ran detached to avoid terminal lifecycle interference. Earlier v1/v2
data remains untouched.

## Replay gate: PASS

For every episode, the verifier restores the saved integration state and runs:

1. Original high-precision actuator commands: compare all recorded qpos and
   terminal qpos to1e-5 tolerance.
2. Actual float32 actions read from the finalized LeRobot parquet: compare
   arm/finger state, including terminal state, to1e-3 tolerance.

Results: six of six pass in each mode. Reference maximum drift is exactly0
on this runtime. Maximum recorded-action arm/finger-state drift is approximately
1.2692e-7; maximum full-qpos drift across these replays is approximately3.0731e-6.
Reference and recorded-action replays finish with177–178consecutive released-
stack ticks, exceeding the30-tick gate. Final XY offsets range approximately
2.18–12.92mm, with Z separation approximately41.98mm.

No tolerance was loosened to obtain PASS. Saved controls, float32 actions,
observation states,substep counts,timestamps and sidecar checksums are checked.
The result is bound to a fingerprint of dataset metadata,data,videos and
sidecars; an old PASS is invalidated before a replay rerun.

### Timing/scoring caveat found in review

MuJoCo's step integrates qpos after computing geometry/contact fields. Reading
those derived fields immediately afterward can observe the prior physics
substep. The original collector/control loop was retained for comparability;
its nominal30Hz loop can also overshoot1/30second by one physics substep.
Actual timestamps/substep counts are preserved in replay metadata.

New replay/rollout scoring forwards a separate simulator-state copy to evaluate
current geometry and contact without modifying the replay solver warm-start
state. This independent fresh-geometry check passed for every saved episode.
The collector's image/state timing convention is not thereby retroactively
changed or proven pixel-perfect. Before a final benchmark, explicitly settle
observation synchronization and fixed-rate cadence; do not silently modify
them between model comparisons.

## Original alignment gate: PASS

The unchanged `verify_action_alignment.py` returned exit0:

```text
mean |action_t - state_t|     (current, should be LARGER): 0.023539
mean |action_t - state_t+1|   (next, should be SMALLER):   0.017174
PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.
```

This remains an aggregate seven-arm-joint test, not proof of all sensor timing.

## Learning pipeline: completed; task success 0/2

Container `smolvla-learning-smoke` completed offline with exit 0, no OOM, using:

- Four training episodes only; computed mean/std/min/max/quantiles exclude
  validation rows. Images use identity normalization before model preprocessing.
- Within-episode50-step chunks, batch2,100optimizer updates,seed20260915,
  AdamW at1e-4 with BF16 autocast. Fixed learning rate is a smoke-test choice.
- Pretrained action expert retained; actual action-projection weight changes
  and finite gradients/loss are required.
- Fresh local output `outputs/validation/smolvla_20260915`; never auto-resume.
- Policy and processors saved/reloaded; seeded physical action predictions
  compared at1e-5relative/absolute tolerance.
- Two validation initial states, up to600ticks each,10actions between replanning,
  same physical controller. Joint/gripper range clipping counted; binary
  gripper thresholding is a documented adapter, not counted as clipping.
- Rollout videos and metrics stay under the ignored output directory. Any
  clipping invalidates the label of unassisted success.

### Measured results

| Check | Result |
|---|---|
| Optimizer updates / batch size | 100 / 2 |
| Training time, excluding load/reload/rollouts | 12.28 seconds |
| First / last sampled training loss | 0.508504 / 0.153093 |
| Gradients and loss | Finite for all updates |
| Maximum tracked action-output projection change | 0.00345515 |
| Reloaded seeded physical-action difference | Exactly 0 |
| Peak PyTorch allocated CUDA memory through training | 2,102,137,856 bytes (about 1.96 GiB) |

The CUDA figure is PyTorch's allocated-memory counter, not total GPU usage or
a prediction for production fine-tuning. First/last losses are on different
random batches and do not establish convergence or validation improvement.

| Held-out rollout | Task | Control ticks | Range-clipped ticks | Success | Final XY offset |
|---:|---|---:|---:|---|---:|
| 4 | red on blue | 600 | 310 | No | 0.28064 m |
| 5 | green on red | 600 | 252 | No | 0.29108 m |

Both cubes remained at approximately equal height at termination; neither
rollout formed a released stable stack. Wall times were 101.39 and 102.91
seconds. Each 600-tick rollout spans approximately 20.4 simulator seconds at
the measured 34 ms cadence. Range clipping counts include either joints or
gripper; no separate attribution is available in this run. Binary gripper
thresholding is part of the adapter, not counted as clipping.

**Interpretation:** recording, replay, tensor/schema adaptation, finite
optimization, actual weight updates, checkpoint/processor serialization and
closed-loop execution all worked. **Task learning has not succeeded.** Four
training episodes and 100 updates are intentionally too small an experiment
to establish performance or dataset sufficiency; failure does not diagnose
whether data amount, representation, optimization or deployment is limiting.

### Follow-up before bulk collection

- Inspect videos and separately measure gripper/joint clipping and phase-level
  failures. No manual visual assessment is claimed in this report.
- Freeze a correct timing convention: the source model timestep is 2 ms and
  the actual recorded control interval is consistently 34 ms (about 29.41 Hz),
  despite nominal 30 FPS metadata. Do not silently compare models under
  different timing or resample commands without validating replay.
- Resolve image/state synchronization and use the fresh-geometry released-stack
  criterion consistently in final collection and evaluation.
- The smoke train/validation split holds out two colour pairs entirely, with
  only four training pairs. It is an integration challenge, not the planned
  balanced in-distribution final validation protocol.
- Keep pi-family adapter validation, benchmark quotas and the larger training
  budget as separate approval-gated milestones.

All containers are stopped after normal completion. No bulk collection, Hub
upload or checkpoint resume was performed. Source/documentation changes remain
uncommitted on the implementation branch for review.

## Tests and known limits

Eight unittest cases passed inside the image: deterministic conditions, full-
state round-trip, action validation, recording lifecycle, fingerprint changes,
nonmutating fresh scoring, training-only statistics and chunk boundaries.
Runtime replay and strict model loading are additional integration evidence.

No model comparison, pi-family adapter validation, broad task coverage, physical
robot validation, bulk collection,Hub upload or training-quality claim follows
from this small run. Inspect videos and fix the final benchmark's evaluation
protocol before scaling. All generated data/models are local and Git-ignored.