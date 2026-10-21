# EGL rendering image and side-by-side check — 2026-10-19

Branch `feat/egl-rendering` (worktree, from `origin/main` `fa72b56`). Design and the criteria, fixed before any EGL run:
[spec](superpowers/specs/2026-10-19-egl-rendering-design.md). Package: `experiments/egl_rendering/`. Raw runs and
`report.json` (about 3.6 GB, gitignored): `outputs/egl_rendering_check/`.

## Result

**The EGL image passes C1, C2 and C3 and is admitted as an instrument.** It is *not* adopted for any dataset or evaluation:
that is a phase decision and the P2 checkpoints were trained on OSMesa pixels (see Limits).

- Image: `vla-robosuite:1.5.2-mj3.4.0-egl`, `sha256:fec83621c46bdae862a60dbfa32607226f8e4afb7610f3b85acf1094a757ade5`. The
  base is the robosuite image `sha256:54d861aa…ccf27e` plus one layer (`apt-get install libegl1`), 2.5 MB. No proxy in its
  environment.
- EGL ran on the GPU (`NVIDIA RTX 6000 Ada Generation/PCIe/SSE2`); OSMesa ran on `llvmpipe`. MuJoCo 3.4.0 in both.

One unmodified P1 attempt (plan slot 0, expert draw 0; 452 ticks, 904 images), five runs, one at a time on cores 0 to 3:

| criterion | measured | result |
|---|---|---|
| **C1** one layer added; OSMesa in the new image equals OSMesa in the base | one layer; 0 of 904 images differ, max 0 levels. An independent hash of the whole image stack is the same in both (`573d82cf486d08a9`) | **pass** |
| **C2** physics and recorded data do not depend on the renderer | all 14 per-tick arrays and the outcome are bit-identical across all five runs, including `qpos`, `ctrl`, `action`, `observation.state`, `props_pose` (an independent per-array hash agrees) | **pass** |
| **C3** EGL repeats within 2 levels and 1% of images | three pairs: 1, 2 and 1 of 904 images differ, 1, 2 and 1 pixels, each by 1 level (limit: at most 9 images) | **pass** |
| C4 EGL against OSMesa (recorded, not gated) | all 904 images differ; 50.3% of pixels; mean 0.71 levels, max 111; first/middle/last image: 59.8% / 45.1% / 60.5% of pixels, max 105 | recorded |
| C5 wall time per attempt (recorded) | OSMesa 36.6 s, 36.5 s; EGL 4.7 s, 4.8 s, 4.8 s: **7.7×** | recorded |

## What this changes in earlier notes

The 2026-10-13 probe (TODO) reported 17× (2.2 s against 37.8 s), 4 to 6 of 904 images differing between EGL repeats, and 72%
of pixels differing from OSMesa (mean 1.2, max 105). Here: 7.7×, 1 to 2 images, 50.3% of pixels (mean 0.71, max 111). The
OSMesa time agrees (37.8 s against 36.5 s). I did not isolate why the EGL time is higher here
(different core allocation, the check's per-tick image copying, GPU state: all candidates, none measured). The EGL repeat
noise also differs between then and now, so the C3 tolerance is a bound to carry, not a constant. These measurements cover
all 904 images with a stated protocol and supersede the probe's figures for the questions they answer.

## What it does not show (limits)

- One attempt, one GPU model and driver (570.211.01), one MuJoCo version. C3 holds with a margin here (2 of 904 against a
  limit of 9) but the tolerance gate has not been exercised on many episodes.
- **EGL and OSMesa frames are different.** Half the pixels differ, by about one level on average and up to 111 at edges. A
  policy trained on OSMesa pixels has not been shown to work on EGL pixels, or the reverse. This check does not test it.
  Therefore: **do not evaluate the P2 checkpoints under EGL**, and do not mix renderers inside a dataset or between a
  dataset and its evaluation. A paired rollout test (same checkpoint, same layouts, both renderers) is what would make a
  faster evaluation legitimate; it is not run.
- **A byte-equal determinism gate cannot hold for EGL** (the repeats differ by a pixel), and any 1-pixel change alters the
  AV1 bitstream. A phase that adopts EGL must replace the P0/P1/P2 gate with: every parquet column exactly equal, and decoded
  frames within the C3 tolerance. That belongs in that phase's spec.
- The physics being independent of the renderer (C2) is shown for this attempt. It follows from rendering not feeding back
  into the simulation, but it is checked, not assumed, only here.

## Notes from the work

- The build container cannot reach the Ubuntu mirrors; the host's HTTP proxy, passed as build arguments, can. The first build
  failed with "no installation candidate", which was the mirror, not the package.
- **A cause I stated and then disproved.** My first EGL smoke test printed an `EGLError` at exit and I attributed it to a
  missing `PYOPENGL_PLATFORM=egl`, wrote that into the spec and made `check.py` require it. A four-way test (PyOpenGL imported
  or not, variable set or not) showed all four render and close fine; a second four-way test showed the cause is leaving the
  renderer to its destructor instead of calling `Renderer.close()` (error with no close, regardless of the variable; none
  with it). The requirement was removed from the code, the spec and the README. The runs above set the variable, which is
  harmless.
- One test mutation (a condition on pixels in C1) could not be killed because it is implied by the one on images; it was
  removed, not tested.
