# Local varied data (Task 1 adapter)

Approved specification: [local varied data](../../docs/superpowers/specs/2026-09-19-local-varied-data-design.md).
Implementation does not authorize this worker to collect. Main must independently
review, rerun CPU tests and the non-stepping preflight before its single launch.

Module `experiments.local_varied_data.record` accepts exactly one of `--preflight`
or `--collect`. No count, source, output, seed, deadline, reserve, retry or resume
overrides. Preflight derives all 77 states in memory and emits JSON on stdout;
it creates no experiment output. Collection exclusively claims the canonical
output (an empty bind mount is permitted), freezes the plan, source, configuration,
all 77 states and their hashes before dataset creation or any expert execution.

60 expert slots: four center, eight marginal-extreme training candidates, four
center, remaining 32 training candidates, then 12 validation. All 12 preparation
episodes must be accepted before slot 13. Each accepted episode receives exactly
two saved-control replays; no expert regeneration. The 24 future policy layouts
are never executed. All source/full-state, media and replay evidence is freshly
audited at completion and by `require_local_varied_pass` before future admission.

Only red/green free-joint XY changes. Warmstart, time, velocities, controls, other
objects, cube Z/quaternions and cameras remain unchanged. No settling. Three future
policy seeds are frozen with the expert/layout seeds; independent RandomState
permutation and jitter streams are derived per split and coordinate via SHA256.

Unchanged shared APIs: fixed-layout `record_attempt`, `TickSink` (through recorder),
`BoundPairedDataset`, row/media readers, trace/contact gates, saved-control replay
and audit functions. The metadata helper is used only in its count-parametric,
non-final mode; own 48/12 final metadata, schedule, snapshot and admission wrappers
avoid old root/24-count/single-reset contracts. No production monkeypatches.

40 GiB at start, 30 GiB floor, six hours including source checks, encoding, replay,
decoding and hashing. Checks run in the existing sink and encoder hooks as well
as loops/boundaries. SIGALRM supplies a process deadline; Python/native libraries
may delay delivery during a native call. The deadline is not a claim of hard
real-time cancellation inside native code. Fail-first evidence is retained;
no automatic deletion or replacement. CPU metrics are not GPU utilization.

Use only pinned image 08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
CPU/runc, network none, repository read-only, temporary cache directories. Tests
write only their own implementation evidence mount and container temporary files.
Never mount old output directories writable. No installs, credentials or training.

Training statistics must later use episode IDs 0–47 only, not LeRobot's default
global statistics. Validation is same-domain interpolation, not held-out regions.
Data quantity and coverage both change relative to the old baseline. Evaluation
scenes have not been expert-success screened. Success alone within a 10mm domain
cannot demonstrate visual grounding; reports retain release/placement deltas.