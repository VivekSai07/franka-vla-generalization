# Local varied data: bounded collection design

Date: 2026-09-19

## Status, purpose and alternatives

The user approved moving toward local varied data after the 68-trial sensitivity
screen. The user approved this exact budget and collection on2026-09-19.
Implementation/tests and independent review precede launch; no automatic training.

Goal: produce a verified dataset covering a small neighborhood of the successful
fixed layout, then separately test whether learning from it improves local robustness.
The prior screen measured SmolVLA19/32 and X-VLA31/32 shifted-scene successes;
it did not establish a sharp distance boundary or a sufficient demonstration count.

Alternatives considered:

1. **Chosen proposal: 60 local demonstrations, explicit train/validation split.**
   Bound cost and independently vary both objects while preserving the working task.
2. More center-only recordings: cheaper, but does not directly address changed layouts.
3. Broad workspace randomization: larger learning and expert-reachability problems;
   deferred until a local varied-data comparison is understood.

## Exact proposed budget

| Set | Count | Geometry | Purpose |
|---|---:|---|---|
| Training, varied |40|40 distinct four-coordinate local layouts|Local behavior learning|
| Training, center |8|Original fixed reset, distinct expert RNG seeds|Retain original capability|
| Validation |12|12 distinct local layouts|Offline fitting diagnostics, never optimization/statistics|
| Future policy evaluation bank |24|24 separately generated local layouts|Frozen final-checkpoint comparison later|

Total expert attempts: **60**, one per demonstration slot, no replacements/retries.
Total initializations to freeze: **77 unique geometries** (76 varied plus center).
Only the 60 training/validation demonstrations execute experts. The 24 policy-bank
scenes receive no expert or policy execution in this data stage and are not screened
for task success. Their physical validity checks are initialization checks only.

Expected data: 60 x528 =31,680 rows; train48 x528 =25,344 rows; validation12 x528
=6,336 rows. Both cameras together63,360 decoded frames. Exactly two saved-control
replays for each successful demonstration, up to120 total; no regenerated-expert
replay consuming hidden extra attempts. Failed/partial episodes remain preserved.

These counts are resource limits and coverage choices, not statistically established
minimums for learning. An eventual comparison changes data volume AND coverage;
it must not be described as a coverage-only ablation without a size-matched control.

## Domain and deterministic geometry sampling

Use the same sealed pilot episode7 reset as both successful fixed-layout baselines.
Define each new layout by four offsets in metres:
`[red_dx, red_dy, green_dx, green_dy]`, each within[-0.010,+0.010].
The limit is per axis: a cube can move up to14.14mm diagonally. Simultaneously
moving both cubes expands beyond the earlier single-cube/single-axis screen;
it is a proposed training neighborhood, not already demonstrated robustness.

Generate a separate stratified Latin-hypercube set for each of40training,
12validation and24evaluation layouts. For a set ofN, divide each coordinate intoN
equal bins, independently permute bins per coordinate and draw once within each
assigned bin. This gives marginal coverage without forcing a fixed pick/place
offset relationship; it does not exhaust all possible four-dimensional combinations.

Master seed20260919; namespace `local_varied_data_v1`. Derive independent RNG
seeds as first four SHA256 bytes big endian of `namespace:master:identity`.
Identities explicitly distinguish split, coordinate, and expert execution. Use
NumPy RandomState consistently and record the full generated values, seeds and
software versions. Expert RNG is separate from layout RNG.

Reject nonfinite/out-of-range layouts, invalid settled geometry, exact duplicate
four-coordinate values or full scene hashes across the new varied splits. Reject
exact overlaps with the original17 position-probe geometries. Center is an explicit
allowed repeat only in training. No minimum train/test spatial separation is claimed:
this is a local interpolation test, not held-out regions. If generation fails its
checks, stop with its proposed plan; do not automatically redraw the entire set.

All requested offsets and full saved states must be frozen before any expert call.
No selection/rejection based on expert reachability or policy success. Do not read
or run the existing sealed240final-scene bank for this experiment.

## Reset intervention and preserved behavior

Restore the source full integration state, alter only the specified red/green
free-joint X/Y coordinates, then refresh derived geometry with the existing
state-preserving synchronization helper. No settling or dynamics before capture.
All other integration fields, including robot state, cube Z/quaternions, velocities,
controls, time, warm-start state and inactive objects, remain bit-identical.
Verify exact full-state diff, requested world-space offsets, unchanged camera
extrinsics, original workspace/clearance, settled heights and no initial robot-cube
contact. Forwarded collision geometry is not proof of stable manipulation.

Use unchanged `experiments.pose_snapshot_v2.expert.Expert`. Retain its existing
Gaussian5mm waypoint jitter, timed gripper phases, IK and bounded joint targets.
Different expert seeds produce genuine independent executions, not copied episodes.
The8center episodes are geometry repeats, not necessarily duplicate trajectories;
report exact action/state sequence hashes and pairwise variation as before.

Single instruction: pick up the red cube and place it on top of the green cube.
Two cubes, no distractors or appearance changes. State9 measured joint/finger
coordinates; action8 absolute seven-joint commands plus gripper0closed/255open.
Two512RGB cameras, AV1/yuv420p,30Hzcontrol/600Hzphysics/20substeps. No oracle
poses/phases/contacts in policy inputs; they remain audit sidecars only.

## Preparation gate inside the budget

Training episode IDs0-47, validation48-59. First12attempts are training examples:
fourcenter executions plus eight varied layouts. Choose these eight deterministically
from the40training candidates before any execution: for each coordinate in the
listed order, select its minimum then maximum candidate, taking the next-most
extreme unused candidate when an index repeats (tie-break by original candidate ID).
Thus all eight are unique; this covers marginal extremes, not every joint corner.

Ordering: fourcenter, eight selected varied, remaining fourcenter, remaining32varied
in original candidate order, then12validation. Preserve a mapping from generated
candidate IDs to actual episode IDs; do not regenerate or renumber based on outcomes.

Every attempt must be recorded, checked and replay-validated before the next.
After all first12pass, proceed to the remaining48 without changing the frozen plan.
A failure stops the entire batch. Successful partial episodes remain available as
evidence, but the dataset is incomplete and cannot admit the proposed training run.
No seed replacement, new root retry, solver adjustment or geometry shrinking.

## Acceptance and recording

Reuse the proven fixed-layout gate semantics in an isolated new adapter, not by
changing fixed-layout constants or old validators. Each accepted episode requires:

- Correct instruction/roles, finite bounded float32 rows,528frames,20substeps,
  observation-command alignment and reconstructed joint-governor commands.
- At least30consecutive closed-command ticks, at least5consecutive bilateral target
  contacts, target lift>50mm while bilateral contact is present.
- Final correct released stack: XY<20mm, relativeZ35–45mm, gripperopen, no top-cube
  finger contact and at least30consecutive stable ticks at completion.
- Reference replay full-qpos drift<=1e-5 and persisted-float32 replay policy-state
  drift<=1e-3, with successful strict final stack in both modes.
- Full video decoding/interval validation and ordered encoder-input/media bindings,
  authentic raw failed-attempt evidence and terminal state when actually available.
- Exact plan/reset/split/row counts, source/runtime/file fingerprints and immutable
  prior inputs. Any exact duplicate full action/state trajectory blocks readiness;
  no deduplication or replacement to conceal it.

Record failures and missing slots, per-stage durations/contact/lift/release/placement,
saturation, per-attempt wall time, CPU resources and actual source/config values.
No complete status until60/60demos and120/120replays pass with a final seal.

## Isolation, storage and telemetry

New experiment package and exclusive output under a new local-varied namespace;
implementation plan will list exact paths. No edits to frozen baseline source,
checkpoints, packaged releases, old data or admissions. Docker-only Python in
the existing pinnedimage, CPU/runc/OSMesa collection, no installs or credentials.
Repository and old data read-only; only this new output root writable.

Require at least40GiBfree at start and30GiBremaining throughout, six-hour wall cap
for collection/encoding/replays/final validation, no automatic extension or deletion.
Current free space approximately59GiB; actual compression/cost is not guaranteed.
Check destination filesystem, not only repository filesystem. Retain logs/container.

Local durable metrics remain authoritative. Since collection is CPU-only, no claim
of collectorGPUutilization; unrelated device activity must not be attributed to it.
Export the completed data-stage summary to the existing private W&B project via
a separately reviewed importer, without credentials in the collector or modifying
the two baseline W&B runs. Missing telemetry remains explicitly unrecorded.

## Subsequent learning comparison (not launched by this specification)

After data gates pass, propose exact training/evaluation budgets separately. Both
frozen baselines and newly trained models must eventually be scored on the same
24reserved scenes and new preregistered policy seeds. Do not use those evaluation
outcomes to select checkpoints or tune recipes before reporting the first result.
The future policy seed list is frozen before collection as specified below;
the later evaluation protocol must use it without outcome-driven replacement.
Retain original fixed-center regression trials and old offset scenes as development
diagnostics. No automatic retraining, continued fine-tuning or extra evaluations.

Reserve three future policy seeds now using the same SHA256 rule with identities
`policy:0`, `policy:1`, `policy:2`. Persist their resolved integers before collection;
assert unique and disjoint from both baseline seed range20260918..20260937 and all
new expert seeds, stopping on collision. This predeclares a possible72trials per
checkpoint, not authorization to execute them or expand a later approved budget.

A10mmneighborhood is smaller than20mmstack tolerance: report release/placement
response to changed destinations as well as binary success. Success alone cannot
establish visual grounding. Fresh evaluation scenes have not been expert-success
filtered; failures must remain in the denominator.

## Review checklist

- [x] Inspect completed study, prior recorder/gates, storage and running jobs.
- [x] Compare local stratification vs center-only vs broad randomization.
- [x] Fix counts, splits, layout method, preparation order, reset and stop semantics.
- [x] Self-review: distinguish diagonal range, same-domain validation, data-volume
  confounding, no automatic training and truthful telemetry. No visual UI needed.
- [x] User approved exact60-demo/24-scene/six-hour data-stage specification and collection.
- [ ] Implementation plan, tests, independent review and actual resource admission.
- [ ] First12preparation, full60collection and120replays.
