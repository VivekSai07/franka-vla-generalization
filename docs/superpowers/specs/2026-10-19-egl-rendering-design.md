# GPU (EGL) rendering image and side-by-side check — design

**Date:** 2026-10-19. **Branch:** `feat/egl-rendering` (worktree `.worktrees/feat-egl-rendering`, from `origin/main` `fa72b56`).
**Status:** criteria below are fixed before any EGL measurement is taken. Follows the deferral recorded in TODO.md
(2026-10-13): *do it at a phase boundary with its own side-by-side check, never mixed into a phase's other changes.*
This branch changes nothing a seal hashes: it adds one new package, `experiments/egl_rendering/`.

## Why

P1 collection spent about 37 s per episode in OSMesa software rendering and 0.6 s in physics. A probe on 2026-10-13
measured EGL (GPU) at 2.2 s per attempt, 17 times faster, with the same expert outcome. Rollout evaluation also renders on
the CPU while the GPU idles. Three things were left open then, and are the content of this check:

1. The robosuite image has no `libEGL.so.1`; adding it means a new image digest.
2. EGL was not byte-deterministic: across 3 identical runs 4 to 6 of 904 images differed, each by 1 pixel by 1 level.
3. EGL and OSMesa differ on 72% of pixels (mean 1.2 levels, maximum 105), so a dataset and the policy trained on it must
   use one renderer.

## What this branch decides, and what it does not

- **Decides:** whether the EGL image is a sound *instrument*: it adds nothing but the library, OSMesa is untouched, the
  physics and the recorded actions do not depend on the renderer, and EGL's own run-to-run noise is small and bounded.
- **Does not decide:** to use EGL for any dataset or evaluation. That is a phase decision (P3 or later) which must use one
  renderer for collection, training and evaluation. Nothing sealed or evaluated so far is re-rendered. **The P2
  checkpoints must not be evaluated under EGL** without a paired test, because they were trained on OSMesa pixels.

## The image

`experiments/egl_rendering/Dockerfile`: `FROM` the robosuite image digest
`sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e`, then `apt-get install --no-install-recommends
libegl1`. The package comes from the Ubuntu archive through the host's proxy, passed as build arguments (not stored). The
NVIDIA EGL driver is injected at run time (`--runtime nvidia --gpus all`, `NVIDIA_DRIVER_CAPABILITIES` including
`graphics`) and `MUJOCO_GL=egl` is set. (An earlier draft of this spec also said `PYOPENGL_PLATFORM=egl` must be set; that
was wrong. Rendering works with and without it; the `EGLError` printed at interpreter exit comes from not calling
`Renderer.close()` explicitly. The check's runs set it anyway; recorded in the report.)

## Measurements (one P1 attempt, plan slot 0, expert draw 0, 904 images; unmodified `props_p1_data.episode`)

| id | measurement | runs |
|---|---|---|
| M1 | OSMesa in the new image against OSMesa in the base image | 1 each, `LP_NUM_THREADS=4` |
| M2 | EGL against EGL | 3 separate processes |
| M3 | EGL against OSMesa | the M2 runs against the M1 run |
| M4 | wall time per attempt, same container flags | each of the above |

For each run the check stores every rendered image, and every non-image per-tick quantity the expert emits (`qpos`,
`ctrl`, `props_pose`, `sim_time`, `phase_index`, `action`, `observation.state`) plus the outcome (success, reason, ticks).

## Criteria (fixed now; none is moved after the numbers are seen)

- **C1, the image only adds the library.** The new image has exactly one layer more than the base, whose command is the
  `libegl1` install, no proxy in its environment, and **M1 is byte-identical in every image and every physics array.**
  Fail: the OSMesa path changed; do not use the image.
- **C2, physics and recorded data do not depend on the renderer.** All physics arrays and the outcome are byte-identical
  across every run in M1, M2 and M3, in particular `qpos`, `ctrl`, `action` and `observation.state` (everything that a
  dataset's parquet holds). Fail: rendering feeds back into the expert; EGL cannot be used at all.
- **C3, EGL is repeatable within a stated tolerance.** Over the three M2 runs, for every pair: maximum absolute difference
  at most 2 levels (of 255), and at most 1% of images differ in any pixel. (The earlier probe: 4 to 6 of 904, 1 level.)
  Fail: the noise is larger than a tolerance gate can carry.
- **C4, the difference from OSMesa is recorded, not gated.** Fraction of pixels differing, mean and maximum absolute
  difference, and the same per camera, for the first, a middle and the last image. This decides nothing; it is the figure a
  later phase decision needs.
- **C5, speed is measured.** Median wall time per attempt for OSMesa and EGL, per run.

**Verdict.** C1, C2 and C3 pass: the EGL image is an admitted instrument and its digest is recorded. A phase that adopts it
must also replace the byte-equal determinism gate of the P0/P1/P2 collections, which cannot hold for EGL and cannot hold for
the AV1 files in any case when one pixel moves: the replacement is exact equality of every parquet column plus the C3
tolerance on decoded frames. That amendment belongs to the adopting phase's spec, not to this branch.

## Out of scope

Re-rendering or re-evaluating any existing dataset or checkpoint; changing `props_scene`, `props_p1_data`,
`props_p2_data` or `robosuite_scene`; the training or evaluation path. A paired EGL-versus-OSMesa rollout evaluation of the
P2 checkpoint is the test that would let a later phase evaluate it faster, and is not run here.
