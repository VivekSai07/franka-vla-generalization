#!/usr/bin/env bash
# Run a python script inside the pinned image, read-only repo, writable /out for analysis artefacts.
set -euo pipefail
REPO=/home/utxowqr/smolvla_franka
OUTDIR=/home/utxowqr/b2_analysis_out
mkdir -p "$OUTDIR"
docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=2g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/vla \
  -v "$REPO:/vla:ro" -v "$OUTDIR:/out:rw" -v "$(realpath "$1"):/script.py:ro" \
  -w /vla --entrypoint python \
  sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc /script.py "${@:2}"
