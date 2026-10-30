# VLA inputs S0: extra-signal sidecar for the sealed P2 episodes — 2026-10-30

**Status: built, sealed and re-verified in a second container.** CPU only, robosuite image
(`sha256:54d861aa…27e`, MuJoCo 3.4.0). Branch `feat/vla-inputs`, package `experiments/vla_inputs_signals/`
(hash-bound from now on: fork, do not edit). Spec:
[design](superpowers/specs/2026-10-30-vla-inputs-s0-signals-design.md); plan:
[plan](superpowers/plans/2026-10-30-vla-inputs-s0-signals.md).

## Result

| | |
|---|---|
| sidecar | `outputs/vla_inputs_signals/p2_signals_20261030/` (87 MB), 540 episodes × 452 frames = 244,080 frames |
| seal fingerprint | `7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c` |
| bound to | P2 seal `b1d642b3…f4d898`, `provenance()` sha256 `577d94e3…c65e07` (this package, P2's 142 sealed files, four extra imports), runtime sha256 `f41bde6f…eff5f` |
| build | 6 shards pinned to 4 CPUs each, 3 min 31 s (`s0_shard_0..5`, all exit 0); merge, report, seal in fresh containers; `--verify` in a second container printed `S0 VERIFIED` |
| per frame | `wrist_wrench` (6), `finger_force` (2), `target_box` (2×5), `props_box` (n×2×5), `segment`, `phase_index` |

## Gates

| gate | result |
|---|---|
| G1 replay exact | 540/540 episodes replayed with `qpos`, prop poses and `final_qpos` byte-equal at every tick (a `ReplayMismatch` would have aborted the shard; none did). The remaining clause, sealed `observation.state == qpos[:, :9]` as float32, was checked afterwards for all 540 by a one-off read-only script that is **not part of the sealed package**: 0 differences (`replay.load_episode(i)`, `np.array_equal(state, qpos[:, :9].astype(np.float32))`). |
| G2 non-perturbation | episode 0, full integration state identical at all 452 ticks with and without every signal computed |
| G3 wrench vs MuJoCo's sensors | 3 episodes (0, 271, 539), every 10th frame (138 frames), maximum absolute difference below 1e-9; plus static known-load, zero-gravity, zero-contact and known-contact fixtures |
| G4 box containment | 60 frames × 2 cameras (10 episodes, 6 frames each): all 120 renders contain target pixels and every pixel lies inside the box expanded by 1 px; IoU between the segmentation box and the projected box, reported not gated: mean 0.949, minimum 0.626 |
| G5 seal | fingerprint recomputed in a second container: equal |

70 tests pass in the robosuite image. Every guard was mutation-checked (flip v, swap u/v, wrong fovy, collision-geom-only,
constant full-image box, no near-plane clip, ignored rotation, world frame, sign flip, robot-robot contacts counted, data
instead of the scratch copy, fingerprint skipping a file type, missing finite/shape/index/segment/identity checks, merge
without provenance or coverage checks, overwrite, training-only percentiles, a stepped simulation). Three tests were
**vacuous until a mutation survived**: a finger-on-finger fixture whose bodies were welded (fixed bodies never collide), a
no-write test whose fields `mj_forward` had already filled, and rewritten-file tests that were rejected by the index digest
before the structural check they claimed to test. All three were fixed and the mutants then died (FAILURES #65).

## What the sidecar says (descriptive, demonstrations only, no policy)

Training rows are the 480 `train` episodes = 216,960 frames. These are the M2 clip bounds, fixed here before M2 trains:

| signal | p0.1 | p99.9 | max \|z\| (mean/std over training rows) |
|---|---|---|---|
| fx, fy, fz (N) | −8.24, −17.82, −13.61 | 13.36, 15.03, 47.15 | 54.6, 58.3, 45.8 |
| tx, ty, tz (N·m) | −2.82, −1.03, −2.69 | 2.10, 1.76, 2.84 | 52.7, 28.0, 25.3 |
| finger left, right (N) | 0.00, 0.00 | 17.39, 17.79 | 17.6, 27.2 |

- **The tails are heavy** (max \|z\| 17–58 against a 0.1/99.9 percentile range of a few N): without clipping, a single
  contact transient would dominate normalisation. M2 must clip to the percentiles above, as designed.
- **Wrist force is dominated by the hand's own weight, not the prop's.** `fz` averages about −11 N before the close and
  in every object's lifted hold (about −9 N after release), against a hand-and-fingers subtree of 1.02 kg (10.0 N, in the
  hand frame) and props of 0.05–0.20 kg (0.5–2 N). The mean
  hold-minus-before-close `fz` is not ordered by mass (bread 0.05 kg: +1.13 N; bottle 0.20 kg: −0.57 N; can 0.15 kg: +0.33 N),
  because the pose and the accelerations of the two segments differ. So the object's weight is **not** readable from the
  mean wrist force on these demonstrations; whatever M2 learns from the wrench has to come from contact transients and
  pose-dependent loading. This is a limit on what M2 can be expected to show, recorded before M2 trains.
- **Finger force separates grasps.** Mean per-finger normal force in the lifted hold: can 6.6 N, bread 5.3 N, lemon 4.9 N,
  milk 4.6 N, bottle 4.4 N, cereal 4.1 N; at most 0.02 N before the close and 0.03–0.04 N after release (not diagnosed: a light residual contact).
- **`in_frame` is 1 in every segment of every object, in both cameras (minimum rate over all cells: exactly 1.0).** It is projected geometric validity, not
  visibility, so as a feature it carries no information on these data; the information is in the four coordinates. M3 should
  not expect the bit to flag a lost target.

## Deviations from the plan, and why

- P2's sealed `provenance.json` lists **142** files, not the 76 the spec and plan quoted (corrected in both and in the
  programme design). Four modules imported by frozen P2 code are **not** in it: `data_collection/collect_libero_dataset.py`,
  `data_collection/pilot_integrity.py`, `data_collection/pilot_plan.py`, `experiments/paired_demonstrations/plan.py`
  (imported by `paired_demonstrations/expert_v1.py`). The S0 contract binds them as `EXTRA_FILES`, and a test fails if any
  repository module the package loads is outside `provenance()`. P2's own seal is unchanged and still does not cover them.
- The scratch copy is `copy.copy(data)`: the Python bindings have no `mj_copyData`.
- The sealed NPZ has no `phase_index`; `replay.frames` rebuilds it from the simulation clock and checks it against the
  sealed gripper command at every tick, and `segment` (before close, closing, closed and lifted, after release) is derived
  from the sealed actions and the target's height (lift threshold 2 cm). Both are stored in the sidecar.
- Shard layout follows P2 (`<work>/shard_KK/` then a hard-linked merge); `build_shard` refuses an existing shard
  directory, so an interrupted shard is deleted and rebuilt, never resumed.

## What S0 does not show

That a real sensor would read these values (exact simulator quantities, contact chatter included); anything about
policies; box visibility under occlusion.

## Files

Sealed output and shard scratch: `outputs/vla_inputs_signals/p2_signals_20261030/` (sealed) and
`…/p2_signals_20261030_shards/` (the same episode files by hard link plus per-shard records), `pilot_20261030/` (one
episode through the CLI, 7.7 s). Not committed, not under `datasets/`.
