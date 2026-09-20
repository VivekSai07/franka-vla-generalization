# Paired demonstrations — stage one

**Implementation for review, not a collected or training-ready dataset.** No real
expert trajectory, collection job, GPU job, training, evaluation or commit was
launched during implementation. Existing bound source/configuration files remain
unchanged. The narrow historical-geometry amendment is approved; main launch
review is still required before use.

## Approved legacy exception — 2026-09-17

Only these exact missing messages may pass the historical inventory gate:

- `datasets/franka_libero_stacking_v1: 2 episodes without full initial geometry`
- `datasets/franka_libero_stacking_v2: 1 episodes without full initial geometry`

The messages are **not removed** or represented as checked geometry. Every
`historical_geometry_exceptions` entry records dataset/count, original message,
`reason: userapproved20260917`, `geometry_checked: false`, exclusion from new
train/eval, and the actual metadata file hashes already bound by the inventory.
The inventory, new manifest and validation results retain this evidence;
validation and the final readiness gate recheck it and the live inventory.
Failed validation also retains the manifest caveat, with a separate
`historical_inventory_rechecked` indicator rather than inventing verification.

All other missing geometry and non-geometric integrity errors remain blocking,
including missing metadata, changed episode counts, metadata/replay mismatch,
hash mismatch, nonfinite/corrupt/unindexed artifacts, missing prior scene/paired
manifests and duplicate geometry. There is no broad ignore flag or legacy-root
exclusion. The new-only plan and reset schedule naturally exclude all legacy
episodes from training and evaluation. Geometric overlap with the three unchecked
legacy episodes **cannot be ruled out**.

Fresh evidence: **24 focused / 129 full CPU Docker tests pass** (12.282 / 26.075 s).
Actual CLI preflight **exits 0**, configuration `passed`, collection `not_started`,
1,906 files / 1,164 geometry hashes, both missing messages and their exception
records present, zero trajectories and `training_ready: false`. Metadata SHA256:

- [v1](../../datasets/franka_libero_stacking_v1/meta/info.json):
  `7561ab94498836e48a26e182f98aa9d59ed94fcb09b2bd5e64b64c5edc18d649`.
- [v2](../../datasets/franka_libero_stacking_v2/meta/info.json):
  `67f2bee9ece73ad6a85cf08b40b5a0b901d3e299a3aaf44831dc36fabe26229d`.

See [approved amendment](../../docs/superpowers/specs/2026-09-17-paired-demonstration-learning-design.md#approved-historical-geometry-amendment--2026-09-17)
and [current preparation](../../docs/PAIRED_LEARNING_PREPARATION_20260917.md).
No real collection/validation/GPU job or commit was run for this amendment.

## Encoding correction — 2026-09-17

The earlier agent implementation introduced **unapproved lossless RGB H264** and
an exact decoded-video-versus-raw-observation SHA256 requirement. Both are now
removed at the user's direction. This restores the approved original collector
format: **AV1/yuv420p**, using pinned LeRobot `libsvtav1`, CRF 30 and GOP 2,
with serial immediate encoding and one encoder thread. No original collector,
bound source, old dataset, manifest or approved design file was changed.

AV1 is lossy: decoded RGB is **not required to equal raw rendered RGB**, and no
replacement pixel-similarity tolerance is introduced. Validation still decodes
every frame and checks the actual codec/pixel format, 512x512 dimensions, 30 FPS,
timestamps from zero at 1/30-second intervals, exact episode interval coverage
and all frame counts. These checks do not prove pixel-perfect correspondence or
detect arbitrary content reordering with otherwise valid timestamps.

The independent seeded-expert replay still compares regenerated **raw** render
hashes exactly against the recorded raw transcript, alongside states, commands
and timing. That comparison does not involve AV1 decoding and is unchanged.
Synthetic tests prove lossy decode differences are accepted by the media gate,
invalid formats/timing/intervals are rejected, and changed raw replay hashes fail.
Actual expert rerender reproducibility remains untested until an authorized run.

## Historical blocker and verification — superseded by the approved exception

Before the amendment, configuration-only preflight in the pinned image passed
configuration checks, but the **CLI exited 2 (collection blocked)**:

- 1,906 historical files bound; 1,164 distinct all-four-cube initial geometry hashes.
- Two episodes in [the legacy v1 dataset](../../datasets/franka_libero_stacking_v1/meta/info.json)
  and one in [the incomplete v2 dataset](../../datasets/franka_libero_stacking_v2/meta/info.json)
  lack full initial cube geometry. Hashing videos or 9-D robot observations cannot
  recover it. No seed-only independence claim, ignored-root flag or bypass exists.
- Supply authentic recoverable historical initial geometry, or explicitly revise
  the approved coverage requirement in a reviewed design. Do not fabricate geometry,
  delete old datasets, or rewrite old manifests to unblock this implementation.
- All **124 CPU tests pass** (20.450 s): the prior 122 plus two additional test
  methods, with existing regressions corrected for the lossy contract. The focused
  recording/media suite passes **19 tests** (5.774 s). TDD first demonstrated the
  old H264/gbrp output, old preflight contract and non-AV1 acceptance failures.
  Synthetic RGB fixtures test LeRobot writing, concatenation, AV1/yuv420p metadata,
  full PyAV decoding/interval coverage and the policy-facing decoder without
  decoded/raw equality. Synthetic-clock/IK tests exercise all three replay modes,
  528 frames, and gripper changes at 107/347. These are **not real expert feasibility
  evidence**. Reset-only unit fixtures settle briefly; no expert trajectories execute.
- Expected corrupt-video tests emit a decoder error; that error is the tested
  rejection, not a failed suite. Native packages emit deprecation and SVT encoder
  warnings (including preset mapping); no codec fallback or test skip is used.
- Actual all-36 schema/media/replay/alignment gates remain **unexecuted**. No paired
  validation PASS or scene-state collection artifact has been created in the workspace.

Plan digest: `ed2f7b814d76b2eaf2575123a4a7b6ce5d624cceb286253e560223db8f8a8dba`.
Tracked-file SHA256-list digest before/after this encoding correction:
`d04d5c3cc4169c068fa5a22207b1de8a35292e3b477081c29b8bbc3e841df4e6`.
Image: `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
Runtime: MuJoCo 3.13.0, NumPy 2.2.6, Pin 3.8.0, LeRobot 0.4.4,
Torch 2.7.1+cu128, torchvision 0.22.1+cu128, TorchCodec 0.4.0, PyAV 15.1.0.

## Source boundaries and interfaces

- [plan.py](plan.py): stdlib-only `make_plan()`, `validate_plan()`, `digest()` and
  `complete_starts()`. Exactly 18 layouts, 36 episodes, fixed split/layout/direction
  streams from master 20260917. Train ordinals 0–11, held-out 12–17. Canonical
  color combination order red/green/blue/yellow; forward then reverse per layout.
  Explicit `subsets` give paired 24, control 12, heldout 12. Control indices:
  0,3,4,7,8,11,12,15,16,19,20,23. No trainer or policy evaluator is implemented.
- [expert_v1.py](expert_v1.py): `Expert.run(initial_state, episode, sink)` and
  `initialize_layout(model, layout)`. Explicit manually extracted loop with
  per-direction `RandomState`; no global RNG/label patching or source rewriting.
  Imports the original task states, IK, trajectory planner, joint governor,
  gripper, synchronization, fixed step and success helpers unchanged.
- [record.py](record.py): `preflight()`, `collect(new_root)`, `run_schedule()`.
  Prerecords the plan, historical inventory, **all 18 full integration states**,
  qpos/control/body/camera arrays and hashes before the first expert invocation.
  Restores the same complete state before either direction. A first failed task
  or integrity exception ends the batch; existing roots cannot be resumed.
- [media.py](media.py): small serial `PairedDataset` subclass. Uses the original
  collector's pinned LeRobot AV1/yuv420p encoding path and native metadata handling;
  no custom RGB H264 encoder, metadata override or global patch remains. Rejects
  non-AV1, non-pinned-thread, parallel, batched and streaming saves. `decode_video()`
  fully decodes and returns frame counts, without hashing lossy decoded pixels.
- [validate.py](validate.py): `validate(new_root)`, `require_paired_pass(new_root)`.
  New paired report only; deliberately **no** legacy replay manifest or pilot
  PASS files. Existing stock training/legacy gates cannot mistake this root for
  the old validated pilot. Future trainers must explicitly consume the new gate.

Base collector SHA256:
`35209e08f874689e643f01ac5ff3db6ce493ac81a0e17270511e4470448869cf`.
Versioned adapter SHA256:
`82d14688b33b0786ad0764fa9ff8d653464d70362216d6d49fc6961059f7a294`.
Both, all shared sources/assets, all five new Python modules, approved spec,
runtime versions and required image reference are bound by `provenance()`.
Docker image identity is enforced by launching the exact digest below, not by
claiming an in-container constant independently attests to its image. The
adapter's real-trajectory equivalence/feasibility is not yet experimentally tested.

## Gates and saved artifacts

The fresh run root will contain these artifacts (none yet collected):

- Plan and manifest JSON, old-file/geometry inventory JSON, one-attempt JSONL ledger.
- Eighteen state NPZs under the `states` child; each includes `mjSTATE_INTEGRATION`,
  qpos, initial controls, forwarded body/camera positions and camera rotations.
- A canonical LeRobot `dataset` child: 24 train +12 held-out demos, 19,008 rows,
  two cameras/38,016 decoded RGB frames if the complete batch passes.
- Thirty-six `transcripts` NPZs with initial/final integration state, every measured
  qpos, float64 sent controls, requested/applied velocity, governor saturation,
  phase index/time, jittered target positions, substeps, success flags and
  synchronized body/camera poses, image time and per-camera raw RGB SHA256.
- Episode expert/encoding/total wall time and batch elapsed time. Failed partial
  transcripts and unencoded PNGs are preserved; no next attempt or resumed root.
- A separate paired-validation report. `training_ready` is false until this report
  passes and its fresh source/runtime/inventory/data fingerprints still match.

Validation checks all exact counts and ordered tasks/task IDs; same complete
reset across directions; family-level split isolation; old/new all-cube geometry
duplicates; canonical float32 schema/finiteness/bounds; every governor command
reconstructed from Q_HOME, including the initialization exception; timestamp and
20-substep alignment; exactly 479 complete 50-action starts per 528-frame episode;
no temporal or dummy action padding persisted as data; AV1/yuv420p format,
video dimensions/FPS/timestamps/interval coverage and actual full decoding/counts.
Exact raw observation RGB hashes are checked only by the seeded expert replay,
not against lossy decoded media.

For **each of all 36 episodes**, validation runs:

1. The true seeded versioned scripted expert, comparing commands, IK requests,
   jitter targets, phases, states and rendered image hashes exactly to recording.
2. Independent float64-control physics replay (original full-qpos tolerance 1e-5).
3. Persisted float32-action physics replay (original 9-D state tolerance 1e-3).

Both control replay modes require the original released-stack predicate for at
least 30 final ticks. There are 108 explicit episode/mode validation results,
not 108 new demonstrations. True-expert validation includes rendering and will
cost additional CPU time. Original [action alignment](../../data_collection/verify_action_alignment.py)
runs unchanged afterward, with exit 0 and explicit PASS required. First mismatch
stops validation; no tolerance relaxation. Validation never calls the legacy
pilot plan/quota acceptance routines or writes their success markers.

## Exact commands — review first, do not launch yet

Run commands from the underscore repository root. They do not use Compose or load
credential files. Source and all old artifacts are read-only; only the dedicated
paired output namespace is writable. No GPU devices or network are exposed.

Configuration-only check (safe, writes no states; actual amended exit **0**,
not a collection or training-readiness pass):

```bash
docker run --rm --network none --runtime runc \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e MUJOCO_GL=osmesa -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$PWD:/vla:ro" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -m experiments.paired_demonstrations.record --preflight
```

**Future collection command** — held for main launch review. Root
must not exist, disk must have at least 30 GiB free. No override/retry/resume CLI:

```bash
mkdir -p outputs/paired_demonstrations
docker run --rm --network none --runtime runc \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e MUJOCO_GL=osmesa -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/paired_demonstrations:/vla/outputs/paired_demonstrations:rw" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -m experiments.paired_demonstrations.record \
  --root /vla/outputs/paired_demonstrations/paired_20260917
```

After a successful collection, run the validator with identical mounts/options:

```bash
docker run --rm --network none --runtime runc \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e MUJOCO_GL=osmesa -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/paired_demonstrations:/vla/outputs/paired_demonstrations:rw" -w /vla \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  python -m experiments.paired_demonstrations.validate \
  --root /vla/outputs/paired_demonstrations/paired_20260917
```

This is a separate **real replay/rendering job**, not the configuration preflight;
it was not launched during implementation. Never point legacy validators/trainers
at this dataset to bypass its gate. Intended output root:
`outputs/paired_demonstrations/paired_20260917` (not created). Recording will write
`prior_inventory.json`, `manifest.json`, `plan.json`, `attempts.jsonl`, 18 `states`,
36 `transcripts` and the new `dataset` child if successful. Validation writes
`paired_validation.json`; only its complete, fresh PASS grants readiness.

CPU suite: use the same pinned read-only/no-GPU Docker configuration with
`python -m unittest discover -s tests -v`.

## Deferred work

The [implementation plan](../../docs/superpowers/plans/2026-09-17-paired-demonstration-stage-one.md)
records future two-model training (5,000 FP32-trainable updates/model, effective
batch 32, condition-specific train statistics) and 72 policy evaluations. No
downstream training/evaluation implementation is included until these interfaces
and the data gates are reviewed. No hyperparameter/split/budget changes follow
automatically from collection success.