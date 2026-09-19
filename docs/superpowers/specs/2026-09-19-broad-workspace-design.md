# Broad workspace data: envelope + jitter collection design

Date: 2026-09-19

## Status, purpose and alternatives

The user approved moving toward broad workspace randomization after the
`local_varied` 60-demo dataset produced a fine-tuned SmolVLA that scored
64/72 = 88.9 % on the reserved 24-layout × 3-seed holdout and 20/20 on the
base-pose fixed-parity comparison. All eight failures were `timeout`
(never `nonfinite_action`); layout `evaluation_012` failed on all three
policy seeds and layout `evaluation_014` failed on two of three, indicating
memorized-opening-sequence behavior that shifted geometries expose.

Goal: produce a verified dataset that widens the spatial support far beyond
±10 mm per axis, and additionally attacks the "cached opening" failure mode
by adding per-episode home-pose jitter (±1° on each arm joint) and per-episode
hover-Z jitter (±3 cm on pre-grasp and pre-place). Separately train and
evaluate; comparison to `smolvla_franka_stacking_localvaried_v1_20260919`
is planned but not automatic.

Alternatives considered:

1. **Chosen proposal: 200 broad demonstrations, two-halves envelope, home-pose
   + hover-Z jitter, single instruction.**
   Directly attacks the three overfitting modes visible in local_varied:
   position sensitivity, cached opening, absolute-Z anchor.
2. Slightly-larger local envelope (±20 mm) alone: cheaper but does not
   address the two known non-spatial memorization failures.
3. Multi-instruction (LIBERO v2 twelve color pairs): broader still, but
   introduces language-grounding and admission-rate risks on top of spatial
   generalization; deferred.
4. Recovery-flub synthesis: interesting but not sim-native for a scripted
   waypoint expert; requires its own research design; deferred.

## Exact proposed budget

| Set | Count | Geometry | Purpose |
|---|---:|---|---|
| Training, LHS-varied |144| 144 distinct four-coordinate broad layouts | Local + broad behavior learning |
| Training, exact base XY |16| Base cube XY + per-episode jitter (H,Z) | Retain base-pose capability under jitter |
| Validation |40| 40 distinct broad layouts | Offline fitting diagnostics only |
| Reserved policy evaluation bank |80| 80 separately generated broad layouts | Frozen final-checkpoint comparison later |

Total admitted expert attempts: **200**, one per slot. Attempt budget for
IK/planner failures: up to 100 rejected attempts (300 total attempts). No
retries against an admitted attempt.
Total initializations to freeze: **281 unique episode identities** (200
admitted + 80 reserved + 1 base). Only the 200 admitted attempts execute
experts. The 80 reserved layouts receive no expert or policy execution in
this data stage. Their physical validity is initialization-check only.

Expected data: 200 × 528 = 105,600 rows; train 160 × 528 = 84,480 rows;
validation 40 × 528 = 21,120 rows. Both cameras together 211,200 decoded
frames. Exactly two saved-control replays per admitted demonstration, up to
400 total; no regenerated-expert replay consuming hidden extra attempts.
Failed/partial episodes remain preserved.

These counts are resource limits and coverage choices, not statistically
established minimums. An eventual comparison to `local_varied` changes data
volume AND coverage AND jitter simultaneously; it must not be described as
a coverage-only ablation without a matched control.

## Envelope, jitter and deterministic sampling

Base pose from the sealed pilot episode 7 reset used by both fixed_layout
baselines: red cube XY = (0.282, −0.383), green cube XY = (0.268, +0.351).

### Cube XY envelope F7

Two disjoint half-rectangles, guaranteed non-overlap by construction:

- Red XY ∈ [0.20, 0.35] × [−0.45, −0.20] (15 cm × 25 cm)
- Green XY ∈ [0.20, 0.35] × [+0.20, +0.45] (15 cm × 25 cm)

Both rectangles contain their respective base-pose cube. 4-D Latin
Hypercube sampling over `(red_x, red_y, green_x, green_y)` with independent
axes; each axis uses a SHA-derived RandomState stream keyed
`layout:{split}:{axis}:permutation` and `layout:{split}:{axis}:jitter`
following the local_varied construction. Splits are drawn independently:
training-varied (144 samples), validation (40), evaluation (80). No cross-
split overlap by construction; identity is the LHS index within the split.

### Home-pose jitter (per episode)

Seven-dimensional deterministic jitter on the initial qpos of the Panda
arm joints. Each of the 7 arm joints receives an independent uniform
sample from the interval **±1.0°** (≈ ±0.01745 rad) via a SHA-derived
RandomState stream keyed `homepose:{episode_index}:{joint_index}` where
`joint_index ∈ {0,…,6}`. The gripper `finger_joint` position is **not**
jittered.

Feasibility gate per draw: verify (a) the jittered arm qpos is within the
Panda joint limits and (b) IK for the pick target and IK for the place
target both converge from the jittered pose. On infeasibility, redraw the
seven-vector at most 8 times per episode using additional stream indices
`homepose:{episode_index}:redraw:{k}`. If no feasible draw is found within
8 tries, the episode is rejected (counts against the 100-rejection budget)
and the next episode index is attempted.

### Hover-Z jitter (per episode)

Two-dimensional deterministic jitter on the grasp controller's pre-grasp
and pre-place hover heights. Each of the two phases receives an
independent uniform sample from the interval **±3.0 cm** via SHA-derived
streams keyed `hoverz:{episode_index}:pre_grasp` and
`hoverz:{episode_index}:pre_place`. Clamped so the effective hover-Z above
the target cube top is at least 8 cm and no greater than 20 cm.

### Master seed and namespace

- `version = 'broad_workspace_data_v1'`
- `master = 20260920`
- `seed = int.from_bytes(SHA256(f'{version}:{master}:{identity}').digest()[:4], 'big')`
- Identities:
  - `layout:{split}:{axis}:{kind}` (as above), `{split} ∈ {training_varied, validation, evaluation}`, `{axis} ∈ {red_x, red_y, green_x, green_y}`, `{kind} ∈ {permutation, jitter}`
  - `homepose:{episode_index}:{joint_index}` and `homepose:{episode_index}:redraw:{k}`
  - `hoverz:{episode_index}:{phase}`
  - `expert:{episode_index}` — expert planner RNG seed
  - `policy:{i}` for `i ∈ {0, 1}` — two frozen policy rollout seeds for
    broad-holdout evaluation; distinct from the three `local_varied`
    policy seeds

Seed collision check across all identities is a plan-time invariant; on
collision, the plan is rejected without redraw.

## Per-episode reset materialization

Because home-pose jitter changes the initial arm qpos, the reset artifact
is now **per-episode**, not shared. For each admitted episode `i`:

- Compute the initial 37-dim qpos as: base source qpos with cube XY
  replaced by the LHS-drawn values (base pose for the 16 exact-XY training
  episodes) and arm qpos[0..6] replaced by base qpos[0..6] + jitter.
- Compute the full 242-dim `initial_state` by packing this qpos, base
  qvel, base ctrl, base act, and current sim time = 0.
- Save `resets/episode_{i:03d}.npz` inside the dataset root with only the
  two fields `initial_state` (242,) and `qpos` (37,); no timestep history.
- Record the SHA256 of each reset .npz into `meta/info.json` under
  `episode_resets: {episode_index: sha256}`.
- Also record `home_pose_jitter_deg` (7-vector) and `hover_z_jitter_m`
  (2-vector) per episode into `meta/info.json`.

Reset determinism check: for each admitted episode, verify that a fresh
`derive_scene(model, base_source, layout, jitter)` reproduces the saved
reset .npz byte-exactly.

## Grasp controller integration

`GraspController` gains a `hover_z_offset: dict[str, float]` argument with
keys `'pre_grasp'` and `'pre_place'`, applied additively to the relevant
waypoint z-coordinates. Existing callers that pass no argument continue
using the default (all zeros), so `local_varied` and `fixed_layout` code
paths are byte-unchanged.

Collision guard: pre-grasp hover z + jitter must remain ≥ (cube top + 8 cm)
and pre-place hover z + jitter must remain ≥ (green top + block height
+ 8 cm). Enforced at plan time by the clamp described above; verified at
episode start.

## Failure-stop policy

Same first-failure-stop policy as `local_varied` *within* an admitted
attempt: any physics NaN, replay mismatch, or admission-gate failure aborts
the collection. **Different from `local_varied`:** rejected attempts
(IK-infeasible home pose or planner failure) do not stop the collection;
they consume from the 100-rejection budget. If the budget is exhausted
before 200 admissions, the collection halts.

Byte budgets: start 60 GiB free, floor 40 GiB, cap 12 h.

## Training

Fresh `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
Trainable 99,880,992 of 450,046,176; frozen VLM; FP32 trainable + FP32 Adam;
BF16 training autocast. AdamW β=(0.9, 0.95), eps=1e-8, wd=1e-10, clip=10.0.
LR: 500-step linear warmup 0→1e-4, cosine decay to 1e-5 at update 15000.
micro 8 × accum 4 = effective 32. Chunk 50, action steps 10. Seed 20260920.
Stock LeRobot flow-matching loss; complete chunks only.

Effective epochs: 15000 × 32 / (160 × 479) ≈ 6.26.

Checkpoints at updates `1, 3000, 6000, 9000, 12000, 15000`. Filenames inside
each `checkpoints/step_{update:06d}/` are unchanged LeRobot names.

Output directory: `outputs/train/smolvla_franka_stacking_broadworkspace_v1_20260920`.
Preflight and admission siblings follow the `_preflight` and `_admission`
naming.

## Evaluation on step_015000

- **Broad holdout** (single mode, 160 trials): 80 held-out layouts × 2
  frozen policy seeds. Each layout carries its own frozen home-pose and
  hover-Z jitter values from the plan (same distributions as training,
  deterministically frozen per layout ID). Success gate 128 / 160 = **80 %**.
- **Fixed parity** (20 trials): base pose (base cube XY, base arm qpos, no
  hover-Z jitter), seeds `20260918..20260937`. Preserves the historical
  parity comparison. Success gate 18 / 20 = **90 %**.

Offline probes: 60 rows per episode (3 rows × 20 sampled episodes) as in
`local_varied`; observation-only, never oracle-fed.

## Failure/success dispositions

- Collection failure at any point: `failure.json` written to the dataset
  root; no partial dataset publishable.
- Training failure: `failure.json` in the training output; no
  auto-relaunch.
- Evaluation failure (per-trial): recorded in the trial's `episode.json`;
  aggregate `capability_passed` remains false if any single trial has
  `nonfinite_action`.

No automatic session extension beyond the 12 h collection cap.

## Explicit non-goals for this stage

- No multi-color instructions.
- No distractor cubes.
- No teleop-style recovery flubs.
- No comparison-with-baseline evaluation code (comparison protocol will be
  a separately reviewed session).
- No push to origin or HF Hub; local commits only under
  `VivekSai07 <viveksaisurya07@gmail.com>`.
