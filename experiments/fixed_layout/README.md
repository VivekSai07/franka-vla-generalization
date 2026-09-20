# Fixed-layout data-stage pipeline

## Latest authorization — 2026-09-18

User subsequently authorized cache cleanup and collection. Build-cache-only
cleanup restored about123GiB free; the30GiB floor is unchanged. The64distinct
software tests now pass, including full synthetic24-slot admission. See the
[launch report](../../docs/FIXED_LAYOUT_LAUNCH_20260918.md) and current container
state before running anything. The implementation-time blocked statements below
are historical. No automatic training/retry is authorized.

**Software preparation only. No collection, expert, real control replay, renderer,
policy inference or training was run to implement Task 2.** Collection still needs
explicit authorization after restoring storage. About 18 GiB was free during
implementation; the non-overridable reserve is **30 GiB**, continuously checked.
Storage probes the nearest existing **output** parent before creation, then the
output directory itself at every budget boundary, not the repository filesystem.
CPU tests use numeric simulator doubles and synthetic on-disk evidence, not
physical task-success evidence.

The approved [plan](../../docs/superpowers/plans/2026-09-18-fixed-layout-baseline.md)
and [design](../../docs/superpowers/specs/2026-09-18-fixed-layout-baseline-design.md)
remain authoritative. No old provenance-bound source or report is changed.

## Fixed contract

- Exactly 24 new expert attempts: train IDs 0–19, validation IDs 20–23. Validation
  has **the same geometry**, not spatial holdouts. The optimistic source selection
  (pilot episode 7, previously successful expert/policies) is not representative.
- IDs 0, 1, 2 are the preparation gate **within** the 24, not extra attempts.
  Each episode must pass persisted-row, raw-evidence, media and both replay gates
  before the next expert call. Any failure stops; no retries, resume, replacement,
  relaunch under another root or automatic training.
- Freeze all IDs/seeds, exact source NPZ/reset and source/runtime inventory before
  the first execution. Restore exactly that complete integration reset every time;
  no resettling/randomization. Only the independent expert waypoint RNG varies.
- Unchanged snapshot-v2 expert, gripper phases, governor, 5 mm Gaussian waypoint
  jitter (not a bounded range), 30 Hz / 600 Hz, 528 frames, float32 state9/action8,
  absolute joint commands and gripper 0/255. Two 512×512 RGB AV1/yuv420p streams;
  existing encoder defaults are CRF30/GOP2/one encoder thread.
- Two control replays only: float64 reference controls and persisted float32
  commands. The validator **never calls `expert.run()`**. The old paired
  `validate_trace()` helper is reused; its expert-regenerating replay helper is not.
- Reference full-qpos error ≤1e-5; float32 policy-state error ≤1e-3. Reference
  geometry/clock match original trace. Both require successful final stable stack.
  Each original/replay requires ≥30 consecutive closed ticks, ≥5 consecutive
  bilateral target contacts, >50 mm lift during bilateral contact, and ≥30 final
  consecutive strictly released stack ticks. XY<20 mm, relative Z35–45 mm,
  open gripper, no finger contact. Hand-body-origin distance is descriptive only.

## Persistence and admission

[record.py](record.py) creates only the canonical, previously absent output root:
`outputs/fixed_layout/fixed_layout_20260918` (relative to the repository).
There are no path, source, seed, reserve, timeout, retry or resume CLI overrides.

Before media writing or budget checks, each delivered callback is copied into
an exclusive, flushed/fsynced NPZ in its attempt directory. It contains the
original two RGB arrays, float32 measured state/action, instruction, and **every
field emitted by the unchanged expert trace**, including full float64 qpos,
commands, geometry, image hashes, phases and governor diagnostics. Only one RGB
tick is buffered by this sidecar writer. Failed attempts keep these files; they
are never saved as successful imitation episodes. Existing evidence is never
replaced. Dataset metadata maintained internally by LeRobot is finalized normally;
row/video files are independent per episode rather than appended over validated
files. Only its own temporary encoding images are cleaned up by the existing
library; no previous dataset, failure directory or artifact is deleted.

Full integration state is saved at reset and, when returned, at termination.
**The unchanged expert callback does not expose per-tick integration state or
live post-step contacts.** Original post-step geometry comes from the next actual
pre-step qpos, or the returned terminal qpos. Contacts are geometrically
reconstructed without advancing physics and checked alongside independently
measured post-step reference-replay contacts/geometry. These are collision
contacts, not force measurements. No hidden state or missing RGB is fabricated.
If an expert is interrupted before returning, the terminal marker points to the
last actually delivered pre-step observation (or reset if none), explicitly
marking actual interruption state/unemitted frames unavailable. If encoding fails
after an expert return, its known terminal state remains identified as available.

The append-only journal records started/admitted/failed attempts and first-three
admission. A replay's exclusive mode directory is its durable budget token;
failed or partial modes cannot be rerun. Each replay preserves per-tick pre/post
geometry, contacts and actual controls before checking the next boundary.
Finalization failures cannot replace the original expert/sink exception.
Abrupt process kill or hardware I/O loss can leave a partial final file; missing
evidence/seal always blocks admission, and the root still cannot be resumed.

[validate.py](validate.py) reads actual parquet, raw NPZ and fully decoded video,
not writer buffers or a cached PASS. It checks dtype/shape/finiteness, strict
feature inventory, task and row indices, timestamps, original pixel hashes,
state/action equality, phase/gripper schedule, governor reconstruction, source
reset/terminal consistency, original/contact gates and both saved replay outcomes.
Lossy AV1 pixels are **not** claimed bit-identical to original RGB. Original
episode video intervals must cover decoded frames exactly once.
Endpoints must be finite real scalars; each camera file belongs to exactly one
episode, checked against all current episode metadata before admission.

[media.py](media.py) binds media at creation without re-encoding. Immediately
before the unchanged pinned serial encoder, it checks every canonical ordered
PNG against the saved raw camera/frame pixel hash and trace hash, recording PNG
byte hashes as well. It hashes the encoder output before and after persistence
and writes an exclusive per-camera record of raw order, codec settings, final
media bytes, path and interval. The validator checks this mapping against raw
trace evidence and the actual video bytes, in addition to decoding every frame.
Wrong-camera, unrelated and reordered synthetic encoder inputs are regression
tested. This adds one PNG/raw verification pass at creation, **no second encode**
and no arbitrary PSNR/MAE tolerance. It trusts the source/runtime-bound encoder
and filesystem during that serial call; it is not an independent mathematical
proof of lossy codec correctness or protection against malicious concurrent
rewrites/all-evidence forgery. It does not compare raw RGB to decoded AV1 hashes.

Final admission requires 24×528 rows, 20/4 split, one task, 48 streams and 48 unique
successful replays. Per-sequence action/state hashes and pairwise arm RMS (rad)
and gripper RMS (0–255 command units) are saved. Exact duplicate state+action
sequences immediately block further collection/training admission/design; they
are preserved, never deduplicated or replaced.

`require_fixed_layout_pass(root)` rechecks fresh source/runtime/reset, all persisted
rows/media/traces/contact evidence, replay reports, duplicate evidence and the
whole-root fingerprint (including validation reports). It does **not** spend new
replay tokens or advance physics. Any changed spec/helper/data/report, missing
media/replay, partial root or stale seal blocks admission. The seal detects drift;
it is not a cryptographic signature against an attacker rewriting all evidence.
Future training must call this gate and compute normalization using IDs0–19 only;
LeRobot's automatically aggregated statistics include validation rows and are
**not** authorized train-only statistics. No trainer is included here.

## Safe read-only preflight / CPU tests

Run from the repository root. No Python on the host, no installs or secret-file
loading. The preflight constructs/restores the simulator only; it never runs the
expert, steps physics, creates a renderer or creates the collection root.

```bash
docker run --rm --runtime runc --network none --read-only \
  --memory 16g --cpus 4 --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  --mount type=bind,src="$PWD",dst=/vla,readonly -w /vla \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/huggingface -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  --entrypoint python \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m experiments.fixed_layout.record --preflight
```

For focused CPU tests, use the same sandbox with the final arguments changed to
`-m unittest discover -s experiments/fixed_layout/tests -v`. Implementation logs
include a stricter profiling guard rejecting native physics steps, real expert
runs, renderer construction and full policy loads. Synthetic media tests only
encode/decode small generated RGB arrays in the temporary filesystem.

Preflight reports `reset_valid`, `storage_eligible` and `output_exists` separately.
Its `collection_ready` remains false: reset validity is not expert/data admission.

## Future collection command — documentation only, NOT current authorization

**Do not execute until explicitly approved and the storage reserve is restored.**
30 GiB is a remaining-free-space floor, not a data-size allowance. Raw RGB alone
would be about 18.6 GiB uncompressed across the batch, plus media/trace overhead;
actual compression is unknown. Starting at exactly the floor will promptly stop.
No automatic deletion, reserve reduction or time extension is allowed.

The single-use directory must not be precreated: Docker must instead bind the
**empty, dedicated new parent namespace** for directory creation. This is the
minimal writable directory envelope; all existing data/output roots stay on the
read-only repository bind. The recorder itself accepts only the fixed child.
If that parent already contains another artifact, stop for mount-layout review;
do not expose old outputs writable, move them, or invent a different run root.

For a newly provisioned, empty `outputs/fixed_layout` parent, after authorization:

```bash
# Reference only: probe the nearest existing destination parent, not repo disk.
destination="$PWD/outputs/fixed_layout/fixed_layout_20260918"
probe="$destination"
while test ! -e "$probe"; do probe="$(dirname "$probe")"; done
test "$(df -B1 --output=avail "$probe" | tail -1)" -ge 32212254720 || exit 1
test ! -e outputs/fixed_layout/fixed_layout_20260918 || exit 1
# Provision the dedicated empty parent only if it does not exist.
test -d outputs/fixed_layout || mkdir outputs/fixed_layout
test -z "$(find outputs/fixed_layout -mindepth 1 -maxdepth 1 -print -quit)" || exit 1
docker run --rm --runtime runc --network none --read-only \
  --memory 16g --cpus 4 --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  --mount type=bind,src="$PWD",dst=/vla,readonly \
  --mount type=bind,src="$PWD/outputs/fixed_layout",dst=/vla/outputs/fixed_layout \
  -w /vla -e MUJOCO_GL=osmesa \
  -e HF_HOME=/tmp/huggingface -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  --entrypoint python \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc \
  -m experiments.fixed_layout.record --collect
```

The three-hour monotonic budget starts before setup and includes recording,
encoding, both replays and final validation. It is checked at each sink/replay/
decode/hash boundary, with a Linux process alarm around the operation. Python
signal delivery can be delayed by an uninterruptible native/kernel call; once
control returns no expired run can continue/admit. Failed evidence is retained.
There is deliberately no standalone CLI that regenerates validation replays.
Full end-to-end simulator/renderer/collection testing and the 24/48 physical gates
remain unexecuted. Independent review occurred and identified interval,
persistence-coverage, destination-storage and media-binding findings. The Task 2
review follow-up addresses those findings with CPU-only synthetic regressions;
this is not collection authorization or a physical-success result. Further
review before the future launch should use the implementation report and tests,
not the earlier inaccurate claim that no independent reviewer was available.