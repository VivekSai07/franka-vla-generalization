# EGL rendering image and side-by-side check

Not a dataset or training package: no contract, no seal, nothing here is hashed by another package. It adds one image and
the check that admits it as an instrument. Design and criteria (fixed before any EGL measurement):
`docs/superpowers/specs/2026-10-19-egl-rendering-design.md`. Result: `docs/EGL_RENDERING_CHECK_20261019.md`.

| file | what |
|---|---|
| `Dockerfile` | the robosuite image (`sha256:54d861aa…ccf27e`) plus `libegl1`, nothing else |
| `check.py` | `run`: one unmodified P1 attempt under a named renderer, storing all images, physics arrays and the outcome. `report`: criteria C1 to C5 |
| `compare.py` | pure numpy comparisons and the C3 limits (at most 2 levels, at most 1% of images differing) |
| `tests/` | 26 tests, admission image or the EGL image; `compare` and the verdict logic were mutation-checked |

**The image** (local tag `vla-robosuite:1.5.2-mj3.4.0-egl`): `sha256:fec83621c46bdae862a60dbfa32607226f8e4afb7610f3b85acf1094a757ade5`.
Build with the classic builder and the host proxy as build arguments (the base reaches no mirror without it; the proxy is not
stored in the image):

```bash
DOCKER_BUILDKIT=0 docker build -q --build-arg http_proxy="$http_proxy" --build-arg https_proxy="$https_proxy" \
  -f experiments/egl_rendering/Dockerfile -t vla-robosuite:1.5.2-mj3.4.0-egl experiments/egl_rendering
```

**Run under EGL:** `--runtime nvidia --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl
-e PYOPENGL_PLATFORM=egl` (the check's runs set it; rendering also works without it). Call `Renderer.close()` explicitly:
a renderer left to its destructor prints an `EGLError` at interpreter exit under EGL (the render itself is fine).
`check.py` refuses to run if the environment is not the renderer its label claims.

**Not adopted for any dataset or evaluation.** Adoption is a phase decision, and the P2 checkpoints were trained on OSMesa
pixels and must not be evaluated under EGL without a paired test.
