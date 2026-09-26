# Broad workspace data (Task 1 adapter)

Approved specification: [broad workspace data](../../docs/superpowers/specs/2026-09-19-broad-workspace-design.md).
Implementation does not authorize this worker to collect. Main must independently
review, rerun CPU tests and the non-stepping preflight before its single launch.

Module `experiments.b1_language_data.record` accepts exactly one of
`--preflight` or `--collect`. No count, source, output, seed, deadline, reserve,
retry or resume overrides. Preflight derives all 281 bank identities in memory
and emits JSON on stdout; it creates no experiment output. Collection
exclusively claims the canonical output (an empty bind mount is permitted),
freezes the plan, source, configuration, all 281 bank identities and their
hashes before dataset creation or any expert execution.

200 admitted expert slots: 144 LHS-varied broad training layouts, 16
exact-base-XY training slots, and 40 broad validation layouts. Each admitted
episode receives per-episode home-pose jitter (±1° on each of 7 arm joints)
and per-episode hover-Z jitter (±3 cm on the Approach Pick and Move to Place
waypoints); values are SHA-derived. Infeasible home poses are redrawn up to
eight times per episode; if none succeed the episode is rejected and consumes
one of a 100-rejection budget. The 80 reserved evaluation layouts are never
executed at this stage — their bank arrays are frozen with their own SHA-derived
jitter for downstream policy evaluation only.

Envelope F7 splits the cube XY workspace into two disjoint half-rectangles that
cannot overlap by construction:

* Red XY ∈ [0.20, 0.35] × [−0.45, −0.20]
* Green XY ∈ [0.20, 0.35] × [+0.20, +0.45]

Each admitted episode receives exactly two saved-control replays; no expert
regeneration. Two future policy seeds are frozen with the expert/layout seeds
and are disjoint from `local_varied`'s three policy seeds. Independent
RandomState streams are derived per split, coordinate, joint, phase, redraw
and policy identity via SHA256.

Per-episode reset materialization: because home-pose jitter changes the initial
arm qpos, each admitted episode writes its own `resets/episode_{i:03d}.npz`
(`initial_state` + `qpos` only). Their SHA256s are recorded in
`meta/info.json:episode_resets`. Fixed-parity evaluation continues to use the
shared base source with no jitter.

`GraspController` is unchanged. Hover-Z jitter is injected inside the
`experiments.b1_language_data.expert.Expert.run` override by mutating the
z-offset of exactly two `generate_task_states` entries — "Approach Pick" and
"Move to Place" — after deep-copying the shared generator output. All other
`states` entries and every other line of `pose_snapshot_v2`'s `run` remain
byte-identical.

Unchanged shared APIs: fixed-layout `record_attempt`, `TickSink` (through
recorder), `BoundPairedDataset`, row/media readers, trace/contact gates,
saved-control replay and audit functions, and the count-parametric `_metadata`
non-final mode. Own 160/40 final metadata, schedule, snapshot, feasibility
gate and admission wrappers avoid old root/60-count/single-reset contracts.
No production monkeypatches.

60 GiB at start, 40 GiB floor, twelve hours including source checks, encoding,
replay, decoding and hashing. Checks run in the existing sink and encoder
hooks as well as loops/boundaries. SIGALRM supplies a process deadline;
Python/native libraries may delay delivery during a native call. The deadline
is not a claim of hard real-time cancellation inside native code. Fail-first
evidence is retained; no automatic deletion or replacement. CPU metrics are
not GPU utilization.

Use only pinned image
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`,
CPU/runc, network none, repository read-only, temporary cache directories.
Tests write only their own implementation evidence mount and container
temporary files. Never mount old output directories writable. No installs,
credentials or training.

Training statistics must later use episode IDs 0–159 only, not LeRobot's
default global statistics. Validation is same-domain interpolation across the
F7 envelope, not held-out regions. Data quantity, coverage AND jitter all
change relative to prior baselines. Evaluation scenes have not been
expert-success screened. Success alone within F7 cannot demonstrate visual
grounding; reports retain release/placement deltas.
