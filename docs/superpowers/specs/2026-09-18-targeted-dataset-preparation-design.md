# Targeted single-task dataset preparation

Date: 2026-09-18

## Status and scope

The user approved proceeding with the dataset direction recommended in the
[coverage review](../../DATA_COVERAGE_AND_PUBLIC_REFERENCES_20260918.md).
This document specifies the first bounded stage for review: **12 fixed expert
demonstrations, plus 24 frozen future evaluation initializations**. The precise
layout/budget below is proposed, not yet approved for execution. No new collection
or policy job has started. Implementation follows written-spec approval.

This is a data-production feasibility gate, not a claim that 12 demonstrations
will train a successful policy. A larger collection and fixed-model learning
comparison require their own frozen budget after this gate. No training,
downloaded benchmark evaluation, automatic budget extension or planner redesign
is part of this stage.

## Why this stage

Nine original layouts were sparse, but X-VLA also failed to close on a seen
layout and failed placement after successful lifts. Spatial coverage alone is
not an established remedy. The scripted expert also has known placement failures.
We must establish that deliberately specified layouts yield valid complete
demonstrations instead of resampling away inconvenient failures.

Alternatives considered:

1. **Targeted, fixed-layout preparation (chosen):** bounds cost, measures actual
   coverage and checks the whole skill sequence before a larger data investment.
2. Broad random recollection: less implementation work, but repeats the risk of
   sparse joint pick/place coverage and hides hard regions behind successful draws.
3. Resume planner/controller redesign: potentially necessary if this fails, but
   explicitly deferred rather than implicitly restarted by data approval.

## Unchanged policy/data contract

- Only "pick up the red cube and place it on top of the green cube"; two cubes.
  No reverse instructions, distractors, new objects or appearance randomization.
- State: nine measured joint/finger coordinates. Action: seven absolute joint
  command targets plus gripper (0 closed / 255 open). No target relabeling.
- Two existing cameras, 512x512 RGB, AV1/yuv420p, 30 Hz observations/actions;
  600 Hz physics, 20 synchronized substeps and original 0.2-second settling.
- Same robot, cameras, cube assets, reset posture, governor and success geometry.
- Privileged positions/contact/phase labels remain audit sidecars, never policy
  inputs. Adding an audit label is not itself additional policy supervision.
- Preserve every existing dataset, result, checkpoint, source binding and report.
  No credential files, host Python, dependency installs, uploads or pushes.

## Proposed preparation layouts

Full factorial of four red-pick locations and three green-destination locations:

| Pick ID | World X (m) | World Y (m) |
|---|---:|---:|
| P0 | 0.22 | -0.25 |
| P1 | 0.55 | -0.25 |
| P2 | 0.22 | 0.10 |
| P3 | 0.55 | 0.10 |

| Destination ID | World X (m) | World Y (m) |
|---|---:|---:|
| D0 | 0.30 | 0.30 |
| D1 | 0.50 | 0.30 |
| D2 | 0.40 | -0.05 |

Order: P0D0, P0D1, P0D2, P1D0, ..., P3D2. Each pick has every destination;
destination choice is not a fixed offset from the pick. All are inside the
original spawn envelope and exceed 0.15 m initial center separation. This is a
restricted capability region spanning negative Y, a left-X boundary and central
placements, not coverage of the entire original workspace or its convex hull.
The coordinates are proposed probes, not measured reachable/successful layouts.

Spawn at Z=0.05 m with identity cube quaternions; park blue/yellow as before.
Set both XY positions explicitly before settling. Validate actual settled
geometry, capture full integration state and camera/body poses, and save both
requested and achieved initial positions. Geometry rejection stops preparation;
it does not trigger a different layout. No trajectory executes before the plan
and all initial states have been frozen.

Master seed: 20260918. Namespace: `targeted_dataset_preparation_v1`.
Derive each RNG seed as the first four bytes, big endian, of SHA256 of
`namespace:master:identity`; use separate `layout`, `expert`, and `evaluation`
identities. Record identities and resolved seeds. Existing trajectory Gaussian
jitter remains unchanged; no extra preparation-layout jitter is added.

## Independent future evaluation initializations

Before expert execution, reserve two scenes per factorial pair: **24 scenes**.
Use independent RNG streams and sample each cube's XY coordinate uniformly
within +/-15 mm of its cell center. Reject only geometric invalidity/duplicates,
never expert or policy failure; at most 1,000 candidate draws per slot, then stop.
Check 0.15 m separation, original bounds and existing geometry validity.

Save the exact settled states and hash the bank. No expert or policy runs on
these scenes during preparation; do not inspect their task outcomes to choose
training layouts. They test nearby-layout generalization within this restricted
distribution, not whole-workspace capability. Subsequent training layouts must
remain disjoint from these exact states; future evaluation must retain every
reserved scene, including failures. Seed independence does not imply a guaranteed
minimum spatial gap or extrapolation test.

Old episodes 300-303 are development probes from now on for targeted work.
The existing 240-scene final bank remains untouched. These 24 scenes are a
separate future capability evaluation, not a replacement for that final bank.

## Expert and implementation boundaries

Use the existing `experiments.pose_snapshot_v2.expert.Expert` unchanged. It copies
trajectory endpoints and preserves the existing timed phases, IK, bounded joint
commands and gripper behavior. It is not known to succeed on these layouts.
Do not use the original aliased trajectory constructor or resume v3/v4 planning.

New orchestration/validation belongs in one isolated experiment package. Reuse
existing timing, initialization semantics, media and replay helpers where their
contracts fit; do not weaken old validators or modify provenance-bound sources.
The old paired recorder's fixed schedule cannot be relabeled as this experiment.

Minimal components:

1. Immutable plan and explicit-position initializer with disjoint RNG streams.
2. Recorder invoking the unchanged expert once per slot, saving pre-command
   images/state and the actual command, plus full trace and terminal outcome.
3. Validator/report for coverage, transitions, media, command reconstruction,
   replay and old-source preservation. Existing replay/scoring tolerances remain
   authoritative; no threshold relaxation to admit a failure.

Renderer shutdown, media failure, nonfinite data, geometry errors and partial
writes produce an incomplete result. Output creation is exclusive; no automatic
resume, overwrite or implicit append. Docker mounts old inputs read-only, writes
only to the new output/dataset roots, uses the existing pinned image without
network or GPU. One preparation collector may run at a time.

## Acceptance and separate stage measurements

Each accepted episode must pass all of the following:

1. Correct task/roles, finite bounded commands, synchronized observation-action
   alignment, original command-governor reconstruction and complete media.
2. At least 30 consecutive closed-command ticks, including at least five
   consecutive post-step bilateral contacts with the red cube. These are
   proposed operational admission thresholds, not universal grasp criteria.
3. Red cube center lifts more than 50 mm above its settled initial height while
   bilateral contact is present. Persist contact and height traces; do not infer
   holding from a small hand-distance alone.
4. Correct release and stable stack using the existing strict metric: XY <20 mm,
   relative Z 35-45 mm, open gripper, no top-cube finger contact and at least
   30 consecutive stable control ticks at completion.
5. Both reference-command and persisted-float32-action replay pass, with their
   existing tolerances and strict task outcome. Verify timestamps, all encoded
   frame intervals, source/runtime hashes and scene-state bindings.

Audit without inventing additional success labels: hand-goal error and orientation
at first close; all gripper transitions and durations; first/sustained contact;
lift; transport contact loss; release position; saturation and final stable ticks.
Use actual achieved poses and correctly defined hand geometry, not commanded
joints or finger-body origins mislabeled as fingertip positions.

The existing expert closes on a fixed phase schedule. This stage deliberately
does not add timing diversity, recovery demonstrations, loss reweighting or
gripper-label changes. If all 12 pass, we have demonstrated reliable production
on these exact probes, **not solved the learned policy's no-close behavior**.

## Budget, failure handling and completion

- Maximum 12 original expert attempts, one per fixed layout, no retries.
- Stop at the first task, admission or replay failure. Retain its initial state,
  all available observations/actions, trace, terminal state and failure reason;
  unattempted slots remain explicit. Preserve completed episodes but mark the
  dataset incomplete and unusable for the proposed learning comparison.
- Replay execution is validation, not an additional expert attempt. At most
  two full replays per attempted episode, one for each required representation.
- Maximum 3 hours total collection/validation wall time after successful software
  preflight. A timeout preserves the partial result; no automatic budget extension.
- 24 bank initializations require settling only; they are not policy trials.
- A passing report requires **12/12** accepted demonstrations and both replay
  checks for every episode, with no missing slot or ignored validation error.
- On failure, report the exact blocker and stop. Do not retune IK, expand a gate,
  swap the failed layout, silently shrink the region or start a different model.
- On success, report data/coverage/transition checks and propose the larger
  collection and fixed-model learning budget. Do not launch training automatically.

## Verification plan

Before actual expert execution, test deterministic plans/seeds, all 12 role/XY
combinations, geometry clearance, split disjointness, old-bank exclusion,
observe-command-record order, never-close/one-tick-close rejection, contact/lift
gates, incomplete-episode preservation and no-retry behavior using synthetic cases.
Reuse the established Docker tests and independently review the new orchestration.
Snapshot/hash all old files actually consumed, then verify unchanged afterward.

## Review checklist

- [x] Existing code, spatial findings, failed expert history and current status read.
- [x] Alternatives compared; dataset-first direction already approved by user.
- [x] Exact first-stage budget, layouts, labels, isolation and stop rules proposed.
- [x] Self-review: scope is preparation only; no promised learning improvement,
  success filtering, holdout reuse, bound-source edits or hidden planner work.
- [ ] User review of this exact written preparation specification.
- [ ] Implementation plan after that review; tests/code and execution afterward.

No visual companion is needed for this textual contract; the existing spatial
audit figure remains available in the coverage review.
