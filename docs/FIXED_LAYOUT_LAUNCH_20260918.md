# Fixed-layout collection — 2026-09-18

## Completed — all 24 demonstrations admitted

Container exited0/noOOM at2026-09-18T14:33:22Z, about1h01m43s after launch.
Exactly24started/24admitted attempts, first-three preparation passed, no failure
markers, retries or substitutions. Final validator reports `training_ready:true`.

- 24episodes /12,672frames /30Hz; train0-19, validation20-23.
- Both camera streams fully decoded:25,344frames.
- 48/48saved-control replays passed; maximum reference full-qpos drift0,
  maximum policy-state drift across replays0.00013670161220646904 (<1e-3).
- Minimum across episodes:240consecutive closed ticks,238bilateral-contact ticks,
  0.2013046763m contact-associated lift,180final stable-stack ticks.
- No exact duplicate state/action sequences. All24initial full-qpos vectors match
  the same pinned source reset. This is repeated fixed-layout data, not spatial
  holdout/generalization or learned-policy success.
- Dataset files136,482,012bytes; complete evidence root1,976,474,588bytes.
  Filesystem still approximately121GiB free.

Fresh read-only completion audit recomputed the entire file fingerprint, matched
current source/runtime provenance and saved source reset, checked all24initial
qpos and48unique successful replay records. It did not repeat media decoding,
physics replays, expert execution or model inference; those completed in the run.

Seal: `403b8ec2d16094863faf611a13768e770dbf19a363c477f0f509fb67d8d8fba5`.
Evidence: [validation](../outputs/fixed_layout/fixed_layout_20260918/validation.json),
[journal](../outputs/fixed_layout/fixed_layout_20260918/journal.jsonl),
[seal](../outputs/fixed_layout/fixed_layout_20260918/seal.json).
No training launched. Future training must use20train episodes only for statistics;
the four validation episodes share geometry and cannot measure spatial generalization.

## Authorization and storage cleanup

User authorized disposable cache cleanup and collection after restoring storage.
Removed only unused Docker build cache, initially older than24hours then all
remaining unused build cache. No images, containers, volumes, datasets, checkpoints,
model assets, wheels, Conda environments or experiment evidence were deleted.
All51images,41containers and4volumes remained; unrelated running container
`moveitFrancesco` was untouched. The pinned image still resolves correctly.

Observed filesystem free space increased from approximately18GiB to123GiB,
about105GiB recovered. Build cache became0B. Docker's per-prune accounting and
filesystem changes differ because storage can be shared; these are observed
filesystem readings, not a sum of advertised cache record sizes. Future builds
may need to rebuild/download intermediate layers, but runnable images remain.

## Software verification

- Fresh focused suite:63tests passed, no failures/errors/forbidden calls.
- Separate synthetic integration container exited0; its9-test run passed in
  1819.455seconds, including8 overlapping helper tests and the end-to-end test.
  Combined unique coverage:64tests, not72.
- Synthetic end-to-end fixture:24episodes,12,672rows,25,344camera frames,
  48saved replay-evidence sets,8mutation rejections. Real AV1/Parquet persistence;
  numerical simulator doubles, zero actual expert/native-physics calls.
- Independent code review and scoped re-review found no remaining concrete
  blocker after finite video interval, destination disk, persistence lifecycle
  and creation-time ordered-PNG/media binding corrections.
- Real source/reset preflight passed without stepping. The old NPZ lacks saved
  camera rotations/reset-control arrays: these are explicitly model-derived or
  restored from integration state, not fabricated historical measurements.
- One imported test-helper file gained an unrelated storage test while integration
  ran. Transcript patch at13:07:08Z confirms its two imported helper functions
  were untouched. Production/integration/runner files predate that test launch.
  Current hashes are verified; no exact historical whole-helper hash is invented.

## Collection contract

24expert executions,20train/4same-geometry validation, all from exact pilot
episode7 reset. Only unchanged expert waypoint RNG varies. First3preparation
attempts count within24. Two saved-control replays per successful episode, no
regenerated expert. Stop on first task/admission/media/replay/resource failure;
no retries, seed/layout replacement, planner repair or automatic training.

30GiB remaining-free-space floor and3hour wall cap unchanged. Pinned image
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`,
CPU/runc/OSMesa, no network/GPU/credentials, read-only repository and old data,
only new output namespace writable. Retain container and logs after exit.

Output: `outputs/fixed_layout/fixed_layout_20260918`.
Container name: `smolvla-fixed-layout-24-20260918`.
Software validation is not evidence of actual expert success. Actual launch
and episode outcomes must be recorded from container/output state.

## Actual launch

Launched at2026-09-18T13:31:39Z, container
`728c7a71c5da0bb7c55d603fc9c99f9e9b70e9382b4625b3479c8f478309cb43`.
Docker confirmed running, exact pinned image, read-only root and networknone.
Implementation commit5e6e88d; launch preflight measured122.22GiB free,
valid source/reset, all tested Python hashes unchanged and output absent.
This records startup only, not completion or any admitted demonstration count.
