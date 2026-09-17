# X-VLA bounded trial: compatibility passed, batch preflight checker corrected

## Current launch — 2026-09-18

User explicitly authorized corrected preflight and full training. The exact
micro4/accum2 scratch now passes: two finite updates, FP32 model/Adam reload
exact, prediction difference0, final data binding passed, no scratch rollouts.
Container `xvla-single-task-preflight-20260918` exited0/noOOM. Outputs and freshly
generated CPU admission use new dated names; failed reports were not overwritten.
47 relevant CPU tests passed before the new admission was generated. Test suites
ran in separate processes to avoid duplicate test-module name collisions.

The user separately approved deleting ONLY three failed-scratch weight artifacts:
model.safetensors at checkpoints000001/000002 and optimizer.pt at000002, within
outputs/train/xvla_single_task_20260917_preflight. They totaled14,074,777,030bytes.
Failed logs/configs/processors/statistics/reload reference/report remain; old
recorded hashes are historical and those three files can no longer be verified.
All SmolVLA checkpoints, public base assets and successful micro1 proof remain.
This freed space for the fresh scratch; approximately42GiB remained at full launch.

Full run: `xvla-single-task-5000-20260918`, container ID
`e1f6a4fb3024e62c35e0236de508896be7c67555ed7294ded51cefdd84877993`.
Output: `outputs/train/xvla_single_task_20260918`.
Fresh pinned pretrained initialization, unchanged approved5k/micro4xaccum2
recipe. Final-only13rollouts follow only if all training/reload/integrity gates
pass. Project is read-only, training outputs writable, network disabled,
CPU OSMesa rendering/GPU learning; no credentials or dependency changes.

Launch is not completion. Preserve sources and the new CPU admission while
running; do not start duplicates or automatically extend budgets. No upload.
The historical blocked/approval-needed statements below are superseded by this
explicit authorization and measured successful scratch gate.

## Scope and approved recipe

User approved a bounded model comparison and then explicitly selected the
resource-informed recipe: fresh X-VLA base, 5,000 optimizer updates, microbatch4
x accumulation2 (effective8), full FP32 trainable weights/Adam with BF16 autocast.
Policy/prompt LR peaks1e-4, VLM1e-5;500-update warmup, cosine to10% of each peak.
Nine train demonstrations0–8, four held-out300–303; no new data/planner work.
Native30-action chunk, ten-action execution horizon, final-only13rollouts,
provisional seen8/9 andheld-out3/4 gate. No automatic retries or extensions.

This is a model-plus-recipe comparison, not an architecture-only ablation.
Effective batch8 implies40,000 sampled chunks, versus160,000 in SmolVLA's
single-task run. Domain0 is an explicit default, not a claim of Franka-specific
pretraining. Two real cameras plus a masked zero third view preserve the model's
three-view capacity; stock ImageNet preprocessing and224resize, no LIBERO rotation.
Token limit64 retains the full instruction and fits244 model sequence tokens.
State9 remains native identity-mapped; action8 uses train-only mean/std and
native auto clean-action MSE on real dimensions only, gripper included.

## Assets and existing runtime

Existing pinned smolvla:rtc-offline image contains XVLA and imports successfully;
no dependency installation or old-runtime/source edits.

- Model `lerobot/xvla-base` revision `cdb7964e4fe842935d671bfab5a5ebe00a96648c`.
- Tokenizer `facebook/bart-large` revision `cb48c1365bd826bd521f650dc2e0940aee54720c`.
- Nine public asset files,3,521,794,465bytes, separately cached under `.cache/xvla_trial/bundle`.
- Model fileSHA256 `f05bc0fab1c9523d7f5d6b41a651313641ca2227a88822249845da8e20e036c9`.
- Strict CPU loading verified879,482,456FP32parameters. Actual CPU backward
  confirmed901trainable tensors with finite gradients. Saved report/asset hashes
  verified; no credentials used for public downloading and no upload.

## Initial GPU preflight succeeded

Container `xvla-two-update-preflight`: exit0/noOOM. Two updates atmicrobatch1,
fullFP32weights+Adam/BF16autocast, all finite. Peakallocated16,445,541,376bytes.
Model and Adam reload exact, prediction maxerror0, finaldatasetbindingPASS.
One chunk inference measured0.15464seconds; this is not sustained rollout latency.
[Initial result](../outputs/validation/xvla_preflight_20260917/result.json).

## Exact approved-batch scratch stopped after two updates

Container `xvla-single-task-batch4-preflight` completed bothmicro4/accum2updates:
loss1.336406/1.116348, gradnorm4.88558/7.44840, update1.2632/1.0069seconds.
Peakallocated21,333,082,112bytes. No OOM or model nonfinite failure.

It then stopped at strict_final_reload with AttributeError: saved observation-only
preprocessing produces `action=None`, but the new orchestration checker tried
to read `.shape`. This is an assistant-added input-check bug, not evidence of an
X-VLA/dataset/optimizer incompatibility. Independent CPU reconstruction restored
the actual saved Adam state successfully; a GPU reload-only read-only diagnostic
located the precise exception without optimizer steps or artifact overwrites.

Fixed only check_model_inputs to accept absent/None action for inference.
Training collation still requires4x30x8; malformed present action tensors still
fail. Three regression tests include the actual saved processor and native model
input builder, observed failing before the fix and passing after it.
**47 CPU tests passed** (44existing+3new), no errors/skips. This is the relevant
X-VLA suite, not a claim that every repository suite was rerun.

[Failed scratch result](../outputs/train/xvla_single_task_20260917_preflight/result.json)
remains unchanged. Existing CPU admission reports are deliberately stale relative
to the checker fix and still reject current sources; no hash was rewritten to
turn a failed run into a pass. Earlier CPU report generation also exposed stale
source bindings and a test-container permission restriction, resolved through
fresh inspection while preserving earlier reports, not bypassing data gates.

## Next step and current stop

Full5,000-update training and13rollouts have **not started**. To proceed under
the fixed no-retry agreement, obtain approval for one fresh two-update scratch
after this checker-only correction. Preserve the failed root; use a new versioned
output and fresh independently generated CPU admission/test evidence. Require
exact model/Adam/prediction reload and current data/cache/source binding before
fresh-base main training. Do not resume scratch weights or silently autotune batch.

New implementation is isolated in experiments/xvla_trial (assets/compatibility)
and experiments/xvla_single_task (fixed recipe/evaluation). Generated CPU reports
stay local. Old SmolVLA files and datasets remain unchanged; no push authorized.