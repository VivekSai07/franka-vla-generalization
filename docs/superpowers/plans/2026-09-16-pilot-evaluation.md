# Pilot evaluation implementation plan

**Approved design:** selected 020000 checkpoint, all 120 independent tuning
scenes in canonical order, horizon 10 / chunk 50, at most 1200 control ticks
(40 simulator seconds), 30 Hz / 20 physics substeps. Final scenes are never
executed. Implementation and CPU tests only; launch requires main-agent review.

**Scope:** new standalone evaluator and its tests only. Preserve all existing
dataset/training/source bindings; no credentials, commits, GPU jobs or rollouts.

1. Test first: exact tuning selection, final exclusion, missing/extra/duplicate
   scene rejection, strict 30-consecutive success/reset, failure-inclusive
   aggregate denominators. Observe failure before adding implementation.
2. Add fail-closed preflight: completed selected training checkpoint/model SHA,
   saved processors, fresh replay/final validation and bank fingerprint, exact
   source/runtime/timing/schema. Verify full integration state, qpos, artifact
   SHA and scene geometry for tuning scenes only. No resampling or retries.
3. Test and implement trial execution: deterministic per-scene policy reset and
   seed, actual transient chunk-call counter, canonical observation/action
   adapters, strict post-step success, separate clipping-free success, two-view
   video, raw/applied actions and pre/post state/time/diagnostics including
   terminal integration state. Nonfinite actions stop only their own trial.
4. Test and implement exclusive output creation, per-trial progress, failed-run
   persistence, and post-run unchanged fingerprint gates before complete.
5. Run targeted and existing unit tests in the existing offline Docker image,
   with network disabled, no GPU devices, and read-only repository mount.
   Review scoped diffs and document remaining launch-time checks here.

**Operational notes:** persisted paths require the repository mounted at /vla.
The whole bank is hashed for the existing final-validation binding; final NPZs
are not restored or evaluated. Binary gripper thresholding is the canonical
adapter, not clipping. Progress rates use finished trials (including failures),
with the fixed planned denominator 120 reported separately. `ticks` counts
action attempts; chunk calls must equal ceil(ticks/10). Nonfinite-action attempts
have a raw action but no applied command/physics step. `completed_control_ticks`
and simulator duration separately count only applied commands. MP4 contains both
views side-by-side, one frame per attempt plus the terminal scene. NPZ terminal
metrics/state are explicit even when the first action is nonfinite.

## Verification and review handoff

- TDD: initial four contract tests failed before implementation, then passed.
   Subsequent checkpoint/restore/trial tests failed for missing behavior; the
   late-nonfinite test exposed attempted-versus-completed tick accounting.
- CPU-only Docker image `smolvla:rtc-offline`, network disabled, runtime `runc`,
   no GPU devices, read-only repository: **69 tests passed**, including **15 new
   evaluator tests**. Existing truncated-video rejection emits its expected
   libav error while passing. CPU tests require `MUJOCO_GL=disable`; EGL cannot
   initialize without driver libraries in this GPU-disabled container.
- Read-only CPU preflight against actual selected 020000 artifacts passed:
   fresh replay/final-validation bindings, 120 tuning full-state/qpos/geometry
   checks, saved config/processors, selected model SHA, and source/runtime/timing.
   No policy was loaded; no simulation was stepped in this preflight.
- Existing tracked/staged diffs remain empty. Evaluator is 320 lines; no
   existing source, dataset, checkpoint or scene-bank files were changed.
- **Pending main-agent review/launch checks:** real FP32 policy and saved
   processor loading on CUDA, deterministic GPU kernel compatibility, EGL
   rendering and actual policy chunk cadence. Tests use a fake policy/renderer
   and real CPU MP4 encoding, not learned-policy rollouts. Mount at `/vla` and
   retain offline/no-credential configuration for launch. No learned-policy
   success result is available; no rollouts, GPU jobs or commits were started.