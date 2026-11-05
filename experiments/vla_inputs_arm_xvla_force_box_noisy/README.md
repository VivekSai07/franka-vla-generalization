# vla_inputs_arm_xvla_force_box_noisy: arms A6-A9, A3's frozen policy fed a corrupted oracle box

M3b, [spec D2/D3](../../docs/superpowers/specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md). **No policy is trained.** Each arm is
A3 (X-VLA at checkpoint 101,520, trained on M2's 17-value force state with the target's box drawn on both camera images) run on M0's
comparison tier, with one change: the box that is drawn is the oracle box (`signals.target_box`, exactly what A3 draws) put through one
controlled corruption. M0 constructs an adapter from the checkpoint path alone, so each condition is a class in `adapter.py`:

| arm | class | condition | what is drawn |
|---|---|---|---|
| A6 | `XVLAJitterSmall` | `jitter_small` | each of the four coordinates of each in-frame box plus Gaussian noise, sigma = 6 px |
| A7 | `XVLAJitterLarge` | `jitter_large` | the same, sigma = 18 px |
| A8 | `XVLADropout` | `dropout` | the gripper-camera box is not drawn (`in_frame` 0) with probability 0.5 per `act`; the front box unchanged |
| A9 | `XVLASwap` | `swap` | on both cameras the live box of the distractor nearest the target in 3D (chosen at the first tick of the trial, fixed for the episode; the first of equal distances in name order) instead of the target's |

`XVLANoCorruption` draws the oracle box unchanged. It is **not an arm**: it exists for the live equality gate (a trial with no corruption
reproduces A3's recorded trace bit for bit) and for the replay test that proves the only difference from A3 is the box.

## Constants (fixed before any trial; pinned by `tests/test_corruption.py`)

`corruption.CONDITIONS`: sigma 6 px, sigma 18 px, dropout probability 0.5; `MIN_SIDE_PX` 2. Boxes are S0's `(u_min, v_min, u_max, v_max,
in_frame)` per camera, divided by 512. After the noise a box is ordered, grown to a minimum side of 2 px about its centre, clipped to the
image and, if clipping left it thinner than 2 px, shifted inward; a box that is not in frame stays absent. The noise is seeded by sha256
of (condition, the oracle pair's float64 bytes, the tick), `tick` being the number of state-builder calls since `reset()`, so a trial
reproduces whatever the worker order or a resume.

## Mean IoU with the true box (measured 2026-11-05, S0's sealed boxes, one frame in eight of every third episode, 10,260 boxes per camera)

| condition | front camera | gripper camera | share of boxes below IoU 0.5 (front / gripper) |
|---|---|---|---|
| `jitter_small` | 0.585 | 0.840 | 28.0% / 1.2% |
| `jitter_large` | 0.254 | 0.622 | 89.4% / 27.2% |
| `dropout` | 1.000 | 0.500 (one absent box counts 0) | 0% / 50% |
| `swap` | the box of another object; near 0 by construction | | |

The front-camera box is small (median 27 px wide, minimum 14 px), so 6 px of noise already costs 0.4 of IoU there; the gripper-camera
box is large (median 128 px wide). The spec's "small" and "large" are sizes of noise, not of IoU loss on both cameras.

## What stays privileged

The box is still the oracle's, corrupted. **A3's state still carries M2's exact simulator forces** (wrist wrench, finger forces): these
arms say what box error a policy that needs a force sensor and a camera tolerates, not what a deployable system would do.

## Hash binding

M0's `prepare` binds this package (every top-level `.py` and this README), A3's arm module (`experiments/vla_inputs_arm_xvla_force_box`),
every repository module they load and the inputs the adapter reads (`INPUT_DIRS`: the X-VLA base bundle and the sealed S0 sidecar). Do not
edit any of it after `prepare`; a change is a new arm. `tests/test_closure.py` checks the closure, that A2's and A3's recorded closures
are unchanged, and that M0's core hash is `c4f6c303f2ae`.

## Tests

`tests/test_corruption.py` (admission or robosuite image, CPU), `tests/test_adapter.py`, `tests/test_parity.py` (replay of sealed episodes 0,
18, 271, 539: the corruption off equals A3's inputs byte for byte; on, the frames are the render drawn with the corrupted sidecar box)
and `tests/test_closure.py` (robosuite image, CPU). `M3B_FULL_LAYOUTS=1` makes the layout test build every comparison layout.
