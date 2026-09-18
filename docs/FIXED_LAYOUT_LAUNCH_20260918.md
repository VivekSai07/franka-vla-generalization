# Fixed-layout collection — 2026-09-18

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
