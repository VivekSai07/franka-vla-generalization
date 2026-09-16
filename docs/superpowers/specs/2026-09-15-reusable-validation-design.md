# Reusable Franka validation design

Approved direction: user authorization on2026-09-15 to prepare a shared
benchmark and proceed with a small end-to-end batch. This document chooses
bounded smoke-test parameters; it does not authorize300episodes.

## Scope and budget

Collect six successful episodes in a new explicit directory; four training,
two validation. Schedule six colour/count conditions deterministically,
including0/1/2 distractors, and retain the scheduled condition on retry.
Use base seed20260915 and a maximum of30 total attempts. This is an integration
sample, not balanced coverage of all12instructions and not a final test set.
Train SmolVLA for100updates at batch2, from the revision-pinned base checkpoint,
with its pretrained expert retained. Use training-only statistics. Save at the
end, reload locally and run two bounded policy rollouts on held-out initial
states. No expectation of high success from100updates/sixdemos.

## Data contract

- State: seven arm joint positions in radians and two finger positions in metres.
- Action: seven absolute joint targets in radians and gripper command0closed/
  255open. Distinguish actual commands from achieved next state.
- Views: front_cam and gripper_cam,512x512 RGB, canonical LeRobot keys unchanged.
- Frequency: nominal30Hz. Record actual simulator timestamps/substep counts;
  the existing loop may overshoot1/30s by a physics step. Do not silently alter
  control cadence while validating the existing fix.
- LeRobot v3 data remains authoritative; replay-only sidecars contain MuJoCo
  integration state, full measured qpos trace, original high-precision controls,
  object geometry diagnostics, attempt seed, hashes and camera/controller config.
- No ground-truth object state is passed to policy preprocessing.
- Metadata has an incomplete/complete lifecycle. Only finalize the manifest
  after dataset.finalize and exact successful sidecar count. Existing roots
  must never be overwritten. Failed attempts get reasons but not demo indices.

## Replay and evaluation

Restore the recorded integration state against identical hashed source/assets
and MuJoCo version. First replay actual high-precision controls as a deterministic
reference, then replay the persisted float32 LeRobot actions. Compare measured
arm/finger states at every frame, enforce valid lengths/shapes/finiteness, and
report position error plus strict stacking outcome. Replay tolerances are1e-5
for reference qpos,1e-3rad/m for persisted policy-state components. Require
released stack: XY<0.02m and0.035<Z<0.045m for30consecutive recorded ticks with
open command and no finger contact with the top cube. Fail closed on drift,
incomplete dataset or provenance mismatch. These are validation thresholds,
not a physical-hardware safety certification.

The unchanged action-alignment verifier must also returnPASS/exit0 on the
new dataset. A policy rollout uses the same action application and timing,
clip/validate bounds explicitly and count interventions; success with invalid
actions/interventions must not be hidden. At most600control ticks per rollout.
Use short execution horizon10 for this integration test, record it explicitly.
This is not a direct comparison to prior reported policy scores.

## Training and model adapters

Use lerobot/smolvla_base revisionc83c3163b8ca9b7e67c509fffd9121e66cb96205,
checkpoint architecture0.75expert width/16VLM layers, replacing source feature
metadata with the new dataset's exact inputs/outputs and new processors/statistics.
Maximum padded dimensions remain32. Inspect loading errors, parameter counts,
finite gradients/loss, optimizer changes and reload prediction agreement.
No pre-fix output/checkpoint resume. Disable W&B and Hub upload. Model/tokenizer
downloads still need authorized network access; stop and document if unavailable.

The pi-family will reuse raw demonstrations with adapters for names, resizing,
normalization and optional relative actions/FAST tokens. Its software/memory
compatibility is a later gate, not implied by this SmolVLA smoke test.

## Verification and handoff

Unit tests first for state serialization, deterministic conditions, action
validation and completion lifecycle. Run inside the existing verified image.
Capture fresh evidence of each runtime gate, including failures. Keep one
status index readable by humans and agents. Update CLAUDE.md/AGENTS.md to direct
agents to it and to prevent accidental bulk jobs. Do not claim the milestone
complete unless collection,replay,alignment,train,reload and rollout all ran.