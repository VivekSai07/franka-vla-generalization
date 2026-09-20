# Paired instruction probe implementation plan

> **For agentic workers:** Use the executing-plans and test-driven-development
> workflows task by task. The user approved this design and inline implementation;
> do not create a worktree, commit, launch GPU work, or expand the three-file scope.

**Goal:** Implement (not launch) a deterministic, fail-closed 48-rollout diagnostic
of instruction-conditioned object choice and training-scene reproduction.

**Architecture:** Call the unchanged `validation.evaluate_pilot.preflight` and
`run_trial`; do not clone the policy/controller loop or edit any bound sources.
Pure selection, comparison and summary helpers surround that loop. Replay each
new saved action trace on CPU for all-present-cube contacts and all ordered-pair
stack diagnostics. Replay never calls a policy.

**Tech stack:** Python, unittest, NumPy, MuJoCo, existing Torch/LeRobot and PyAV,
the existing offline Docker image; no new dependencies and no host Python.

## Global constraints and predeclared design

- Only add this plan, `validation/paired_instruction_probe.py` and
  `tests/test_paired_instruction_probe.py`. No credentials, source changes,
  training, new demonstration data, final-scene rollouts, commits or GPU jobs.
- Require the completed 120-trial tuning baseline, its unchanged-post-run marker,
  exact selected-checkpoint/data/runtime/source/processor fingerprints from a
  fresh existing evaluator preflight, and matching episode metadata. Hash every
  baseline file before and after; retain the hash inventory in the new report.
- The evaluator's bank fingerprint includes sealed final files for integrity;
  never restore/select/evaluate them. No new full 120-trial audit.
- Select the lowest baseline ordinal **two-cube** tuning entry for each of the
  12 ordered color pairs, independent of outcomes. Order selected layouts by
  baseline ordinal. For each, run original, reversed, seedcontrol in that order:
  original and reversed use `20260916 + baseline_ordinal`; seedcontrol uses +1.
- Reverse only the copied entry's `pick`/`place`. Keep original `file`, `sha256`,
  `scene_hash`, `seed`, `pair_id`, `colors` and physical layout unchanged. Pass
  baseline ordinal (or ordinal+1) to `run_trial`, not the new trial index.
- Then run 12 train conditions, one lowest episode-index two-cube scene per
  ordered pair, from explicit manifest train IDs only. Require two cubes;
  no fallback to validation or three/four cubes. Sort by episode index and use
  `20260916 + episode_index` as the fixed train rollout seed.
- Train adapter: hash-check original replay NPZ first, restore `initial_state`,
  check `initial_qpos` if present (otherwise initial positions from the integration
  state, checking the first saved `qpos` observation when present), verify
  `scene_hash`/geometry. Never mistake an entire `qpos` trajectory for initialization.
  Exclusively create a derived NPZ with just `initial_state` and `qpos` in the new
  output. Preserve original source path/hash plus derived path/hash in the plan.
- Write all 48 fully selected entries, roles, seeds, derived-state provenance,
  metrics definitions and fixed tolerances to `plan.json` before loading the
  policy. No resume, overwrite, retries or outcome-dependent replacements.
- Same baseline environment: offline image/runtime, OSMesa rendering,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, Torch threads=8, deterministic algorithms,
  cuDNN benchmark disabled. Frozen selected 020000, FP32-preserving loader and
  saved processors; original 50/10 cadence, 30 Hz, 20 substeps, 40 s/1200 ticks,
  unchanged controller, action adapter, cameras and 30-tick success stopping.
- Original trace comparison: identical key sets, shapes and dtypes; prefer
  `np.array_equal`. Otherwise zero relative tolerance, action absolute tolerance
  `1e-7`, state/time/integration/other floating metrics `1e-8`; discrete arrays
  exact. Compare every saved array including initial/terminal integration,
  terminal qpos/qvel/time and diagnostics, not merely final success. Nonfinite
  action failures may match only at identical nonfinite positions/signs.
  Stop at the first mismatched array or metadata; retain failed report, no retry.
- After each trial, replay saved applied actions on CPU and verify pre/post
  states/times, action bounding, terminal integration and requested-pair success.
  Record per-cube left/right/any/bilateral finger contacts and lift per tick,
  first contact and bilateral ticks; sustained grasp proxy = 10 consecutive
  bilateral ticks, and sustained grasp+lift = 10 consecutive bilateral ticks
  each above 5 cm relative to that cube's initial Z. These are diagnostic
  proxies, not proof of a physical grasp. Record first completion ticks.
- Record all ordered pairs' strict released-stack predicates each tick, longest
  streak and first 30-tick completion (including wrong-pair stacks). No extension
  past the original evaluator's success/timeout stop.
- First-object choice uses first bilateral contact; report sustained and
  sustained+lift choices as well. Simultaneous first contacts are ties: expose
  tied colors and use null for singular choice rather than color-order bias.
- Summaries remain separate for original/reversed/seedcontrol/train. Paired
  switch = two nonnull different choices; null/tie/nonfinite/timeout outcomes
  stay in the 12-layout denominator (also report missing, both-null and
  one-null pairs). A null-to-object transition is not a switch. Include requested
  choice rates, task success, clipping-free success and wrong stacks; no causal
  or guaranteed-success claims.
- Mark complete only after 48 trial records, 48 CPU replay records, 12 original
  comparisons, unchanged final preflight, baseline inventory, derived states,
  plan and probe source hashes. Any exception writes status `failed`.

## Tasks and verification

### 1. Deterministic protocol helpers (tests first)

- [x] Add unittest cases for exact 36+12 selection, order, role/seed identity,
  deep-copy isolation, outcome independence, missing pairs, no validation/final
  selection and deterministic two-cube training requirements.
- [x] Add cases for every-array baseline mismatches (shape, dtype, omitted key,
  late raw/applied action, state/time, terminal integration), stringent tolerance
  and null-inclusive paired summaries.
- [x] Run the new tests in CPU-only, network-disabled Docker with `/vla` read-only;
  observe assertions that the new module/helpers are not implemented.
- [x] Implement selection/comparison/summary helpers; run their tests green.

### 2. Initial-state adapter and CPU replay

- [x] First add real small-MuJoCo tests for initial-state derivation with
  `initial_qpos`, trajectory `qpos`, missing optional qpos, corrupt hashes/states,
  exclusive output and untouched source. Add saved-action replay tests including
  nonfinite terminal attempts, changed action/state/terminal traces, contact
  streak resets, ties and all-pair 30-tick success.
- [x] Implement using existing restore, action, timing and scoring functions;
  run these CPU tests green. No renderer/policy is needed for replay tests.

### 3. Fail-closed orchestration

- [x] First test baseline gates/fingerprints, missing artifacts, immutable
  inventory, no overwrite, exception reporting, original mismatch stops further
  calls, exact 48 completion and postflight changes (policy boundary doubles
  only; no actual GPU policy).
- [x] Implement CLI with `--baseline` and `--output`; infer bound input paths
  from baseline result, not new defaults. Gate before loader, exclusively create
  outputs, persist plan before policy calls, freeze the FP32-loaded policy and
  use saved processors. Run tests green.
- [x] Run full CPU unittest discovery in Docker (previous suite: 69 tests),
  inspect new-file diagnostics and diff whitespace, optionally run existing
  read-only preflight and real selection without inference or output creation.
- [x] Recheck the tracked-file digest and scope. Record actual checks and
  limitations here. No commit and no rollout launch.

## Verification record

- Test-first evidence: nine helper tests failed for the absent module, then
  passed. Eight additional replay/runner tests failed for missing behaviors,
  then passed. Additional regression coverage tests frozen deterministic
  inference, stale preflight, infrastructure errors, plan and derived-file
  mutations. Standalone CLI test first reproduced a missing repository import;
  adding the same root-path setup as the baseline evaluator fixed it.
- Final full suite: **90 tests passed (69 existing + 21 new)** in **14.804 s**,
  no failures/errors. Executed unittest discovery in `smolvla:rtc-offline`,
  `--runtime runc`, network disabled, repository read-only, bytecode disabled,
  OSMesa and GPU visibility disabled. Explicit `torch.cuda.is_available()`
  check returned false. The libav missing-moov message is expected from the
  existing deliberately truncated-video rejection test, not a new failure.
- Fresh **actual** `evaluate_pilot.preflight` passed and its fingerprints matched
  the baseline result. All 120 baseline episode/artifact metadata bindings
  passed. Hashed all **361 baseline files** before/after: unchanged. This did
  not rerun the 120-rollout audit or decode/replay the baseline videos.
- Actual selection verified: tuning ordinals **0, 10, 20, 30, 40, 50, 60, 70,
  80, 90, 100, 110**; train episode IDs **0, 25, 50, 75, 100, 125, 150, 175,
  200, 225, 250, 275**. Every selected initialization has exactly two cubes.
- All **12 actual training initializations** were hash-verified, adapted and
  restored with the unchanged evaluator in temporary container-only output.
  The actual NPZ files have full integration `initial_state` and trajectory
  `qpos`; first-observation qpos matched the recovered initial positions exactly.
  Synthetic tests also cover `initial_qpos` and integration-only variants.
- All previously tracked file bytes remained unchanged: pre/post aggregate
  SHA-256 `f581cade660da18b225faca573107f5d1acd63740ab82c87a55a46e88d3b783c`.
  New-file whitespace checks pass. Only the three authorized files were added;
  pre-existing local offline setup files were left alone. No commit.

## Entry point and output contract (not launched)

The standalone script and module entry point both accept only `--baseline` and
`--output`. Use the existing pinned offline image and baseline OSMesa/GPU
environment for any separately launched policy run. Mount the checkout and all
bound inputs read-only at `/vla` because saved baseline paths/configuration use
that prefix. Mount only a new diagnostic parent writable and select a **nonexistent
child** as `--output` (not an already-existing mount point). No credentials or
network are needed. `--help` is tested without policy loading.

The output contains the predeclared plan, rolling/final result, 12 derived train
initializations and 48 exclusive trial directories. Existing evaluator outputs
are `episode.json`, `trace.npz`, `cameras.mp4`; replay adds `contacts.npz` and
`diagnostics.json`. The aggregate report links trial conditions and original
comparison results. Finger contact arrays are `[post-control tick, cube, left/right]`;
lift and all-pair stack arrays share that tick axis. Event ticks are zero-based
completion ticks, not inferred substep contacts.

**Limitations:** GPU execution and the actual 48-rollout results are outside
this implementation session. CPU tests use policy boundary doubles; they do
not establish seeded GPU action reproduction or learned task success. The
original-condition gate must establish reproduction during execution and must
not be relaxed if it fails. Contact/grasp/lift are sampled diagnostic proxies,
not proof of load-bearing grasp or causal language use. Traces end at the
unchanged evaluator's requested-pair success/timeout/nonfinite stopping point.
No new training, collection, final-scene rollout, policy loading or persistent
diagnostic output occurred during verification.