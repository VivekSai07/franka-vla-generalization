# The robosuite image plus libEGL, so MuJoCo can render on the GPU (MUJOCO_GL=egl).
# Base: vla-robosuite:1.5.2-mj3.4.0 (sha256:54d861aa...ccf27e). The NVIDIA EGL driver itself is injected at run time by
# the NVIDIA container runtime (--runtime nvidia --gpus all with NVIDIA_DRIVER_CAPABILITIES including graphics); only the
# vendor-neutral dispatch library is missing from the base. OSMesa stays installed and unchanged.
FROM sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 \
 && rm -rf /var/lib/apt/lists/*
