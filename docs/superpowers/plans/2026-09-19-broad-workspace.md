# Broad workspace data implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. TDD, independent review, then main-only real collection.

**Goal:** Collect 200 admitted broad-workspace demos in envelope F7 with per-episode home-pose and hover-Z jitter, freezing 80 held-out evaluation layouts and 2 policy seeds before execution.
**Architecture:** Isolated new package `experiments/broad_workspace_data/` adapts the local_varied record/validate lifecycle; per-episode reset materialization replaces the shared source.npz assumption; GraspController gains a `hover_z_offset` argument. No monkeypatch or source rewriting of old modules.
**Tech Stack:** Pinned Docker `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`, Python 3.10 / NumPy / MuJoCo / LeRobot / PyAV CPU runc / OSMesa only. No new packages.

## Global constraints

Exact approved spec docs/superpowers/specs/2026-09-19-broad-workspace-design.md.
200 admits max (up to 300 attempts, 100 rejections tolerated); 160 train (144 LHS-varied + 16 base XY) + 40 val + 80 reserved eval; 281 unique episode identities.
Envelope F7: red XY ∈ [0.20, 0.35] × [−0.45, −0.20]; green XY ∈ [0.20, 0.35] × [+0.20, +0.45]; disjoint halves, no overlap possible.
Home-pose jitter ±1° on each of 7 arm joints, per episode, SHA-seeded. Hover-Z jitter ±3 cm on pre-grasp and pre-place, per episode, SHA-seeded, clamped ≥ 8 cm above cube top.
Per-episode reset materialization (no shared source.npz for the 200 demos); each reset saved and hashed into meta/info.json.
First-failure-stop within an admitted attempt; rejected attempts (IK-infeasible / planner-failed) count against the rejection budget and do not stop collection.
60 GiB start / 40 GiB remaining, 12 h cap, 400 control replays max, no training / GPU / policy calls.
All old bound code / spec / data / cache / checkpoints / READMEs immutable. New package only.
Existing feature branch / in-place canonical /vla; main-only commits; sole user; no push.
NEVER read .env / .env.example or host Python / install. Worker CPU tests / no actual expert / physics stepping / renderer; main launches only after tests / review / real preflight.

## Task 1: integrated bounded collector with per-episode reset and jitter

Files new `experiments/broad_workspace_data/{__init__, plan, scenes, record, validate}.py`, `tests/test_plan.py`, `test_scenes.py`, `test_record.py`, `test_validate.py`, `README.md`. Small additional modules only if needed. Output `outputs/broad_workspace_data/broad_workspace_20260920`; CPU logs `outputs/implementation/broad_workspace_data_20260920`. CLI `--preflight` readonly and `--collect` exclusive. No arbitrary paths / seeds / counts / timeout / retry overrides.

- [ ] Write RED tests before implementation. Interfaces `make_plan() -> dict`, `validate_plan(plan)`, `derive_scene(model, base_source, layout, jitter) -> arrays`, `preflight() -> dict`, `collect() -> report`, `require_broad_workspace_pass(root) -> report`.
```python
p = make_plan()
assert len(p['episodes']) == 200 and len(p['layouts']) == 264
assert p['train_ids'] == list(range(160)) and p['validation_ids'] == list(range(160, 200))
assert len(p['evaluation_layout_ids']) == 80 and len(set(p['policy_seeds'])) == 2
for e in p['episodes']:
    assert len(e['home_pose_jitter_deg']) == 7 and all(abs(v) <= 1.0 for v in e['home_pose_jitter_deg'])
    assert len(e['hover_z_jitter_m']) == 2 and all(abs(v) <= 0.03 for v in e['hover_z_jitter_m'])
```
- [ ] Immutable plan exact types / counts; separate 144 / 40 / 80 LHS permutations + jitter per coordinate using SHA seed namespace / master / identity from spec. The 16 exact-base-XY training slots use `layout_id='base'` with the base cube XY; jitter (home + Z) still applies. Freeze all resolved seeds, offset values, jitter values, candidate-to-episode mapping, 2 policy seeds disjoint from expert / base. No automatic redraw of the plan even on geometry failure. Tests: marginal bin coverage / extrema uniqueness / disjoint splits / envelope containment / float bool spoof / reproducibility / seed collision detection / no cross-split layout overlap / no XY overlap between red and green rectangles.
- [ ] Scene adapter reuses old source reset loader / geometry helpers if generic; alters only 4 free-joint XY values AND arm qpos[0..6] with jitter; asserts full integration mask exactly for cubes and arm, cameras / Z / etc unchanged. `derive_scene(model, base_source, layout, jitter)` returns arrays where `initial_state` and `qpos` reflect BOTH the layout XY change AND the arm home-pose jitter. Real non-step CPU check all 281 geometry valid, no duplicates, all 200 demo reset refs resolve, 80 evaluation never in expert list. Save plan / all 281 states / hash manifest BEFORE dataset or trajectory, compare old source hashes before / after. No exposure oracle policy fields.
- [ ] Feasibility gate implemented as `feasible(model, layout, jitter) -> bool`: checks (a) jittered arm qpos within joint limits and (b) IK for pick and place converges from jittered pose. Redraw jitter up to 8 times per episode using `homepose:{i}:redraw:{k}` streams. Test: synthetic infeasibility → 8 redraws → rejection; feasibility on first try → no redraws; deterministic redraw sequence.
- [ ] Reuse `record_attempt` / `TickSink` / media bounded encoder from `local_varied_data` unchanged when independent of root / count; own `run_schedule` / provenance / check-output / resources. Per-attempt actual unchanged pose-snapshot-v2 Expert with new `hover_z_offset` argument plumbed from the plan. Preserve partial raw frames / terminal, first-failure-stop, step / resource check in sink. Store per-attempt elapsed CPU metrics, stage / contact / release / placement diagnostics and command saturation. No GPU util claim.
- [ ] `GraspController.__init__` gains `hover_z_offset: dict[str, float] | None = None`; when None, behavior is byte-identical to prior calls. Test: default call produces identical waypoints as pre-change; keyed call shifts pre_grasp / pre_place waypoints by the requested Δz and clamps to floor. Old callers (`local_varied_data.record`, `fixed_layout.record`) remain call-site-unchanged.
- [ ] Reuse fixed `validate.read_rows` / `load_attempt` / `replay_controls` / `audit_replay` / `validate_video` / `contact_gates` / media binding only when no 60 count / shared reset assumptions. New metadata / schedule wrapper 200 count / 160-40 split / per-layout+jitter reset; no global patching. Per-episode reset .npz saved to `resets/episode_{i:03d}.npz` with SHA into `meta/info.json`. Exactly two saved-control replays per accepted episode, reused exclusive-mode budget tokens; do NOT call old expert regeneration helper. Governing tolerances unchanged.
- [ ] Final audit independently: all 200 persisted episode entries / 105,600 rows / 211,200 video frames and 400 unique successful replay evidences, no extra / missing files, duplicate full action + state rejection, all 281 reset bindings and 80 reserved unexecuted scenes unchanged, first-100-rejection budget observed, full fingerprint / seal. Per-episode jitter values persisted and re-derivable from plan. Future pass gate must compare fresh source / runtime / actual data, not trust saved success. No training stats global.
- [ ] Failures: preserve original exception despite finalize error; root claim exclusive empty bind-mount allowed as reviewed old pattern; no resume / delete / clear failed states; twelve-hour monotonic + alarm and disk-40 floor inside encoding / replay / checks as well as boundary.
- [ ] Tests synthetic fake expert to prove 200 count / all frozen before calls / feasibility gate / rejection budget / 80 bank not run / no extra on failure / disk floor / no override / replay mode counts. Real 2-episode synthetic LeRobot persistence with new adapter / AV1 and media bindings, no physics / model; final full 200 scheduler / metadata gates synthetic arrays, not 200 actual encodes if unnecessary. Independently test shared generic validators for layout-specific reset injection AND arm-jitter injection, real LHS 264 non-stepping valid. No new actual experts.
- [ ] Run GREEN new + relevant old unit suites separate Python processes pinned CPU readonly repo, tmp cache and own implementation RW only. Report RED / GREEN / old hashes / APIs / limitations. All `local_varied_data` tests must still pass unchanged; test that `GraspController` default call is byte-identical.
- [ ] Independent review spec + quality, main rerun CPU tests + non-stepping preflight, commit.

## Task 2: main collection

- [ ] Check 60 GiB start space, running jobs / source uniqueness and no pending changes.
- [ ] Launch one CPU collector, no GPU / network / credentials, readonly repo / bank inputs, new output only RW; retain logs / container.
- [ ] Inspect startup and register dockerwait notification once, no polling / duplicate.
- [ ] On completion verify counts / replays / seal / holdout untouched; stop on failure. No training / broad evaluation / new attempts without next decision.

## Task 3: training (deferred to next session)

Not implemented in this session's plan. Recorded here for reference:
- Package `experiments/smolvla_broadworkspace/` mirrors `experiments/smolvla_local_varied/` with the new dataset root, seal, splits (160 / 40), 15000 updates, checkpoints at step_000001 / 003000 / 006000 / 009000 / 012000 / 015000, seed 20260920. Recipe otherwise identical.

## Task 4: evaluation (deferred to next session)

Not implemented in this session's plan. Recorded here for reference:
- Package `experiments/smolvla_broadworkspace_eval/` mirrors `experiments/smolvla_local_varied_eval/`. Modes: `broad_holdout` (80 layouts × 2 policy seeds = 160 trials, gate 128/160 = 80 %) and `fixed_parity` (base pose, no jitter, 20 seeds, gate 18/20 = 90 %). Per-trial reset materialized from the plan's held-out layouts (which carry frozen home-pose and hover-Z jitter values).

Self-review: validation snapshots use each planned reset not old center; base geometry duplication intentional (16 slots); trajectory duplication blocked; all 281 states freeze prior calls, future policy seeds fixed, only 200 demo IDs allowed experts. GraspController default byte-identical. No blocked legacy rewrites.
