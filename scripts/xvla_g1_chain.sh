#!/usr/bin/env bash
# The G1 gate sequence, then training until a deadline, unattended: stops at the first failing gate and never trains past a failure.
#   setsid nohup scripts/xvla_g1_chain.sh "tomorrow 08:00" > outputs/train/xvla_g1_rand_chain.log 2>&1 &
# Run it from the checkout that holds the sources (the detached worktree .worktrees/g1-run); gitignored trees are read from the main checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=${G1_MAIN:-/home/utxowqr/smolvla_franka}
OUT=outputs/train/xvla_g1_rand_v1
D=outputs/gen_g1_data/g1_20261110
IMAGE=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
deadline=${1:?deadline, e.g. 'tomorrow 08:00'}
say() { echo "$(date '+%F %T') [g1-chain] $*"; }

# gate <name> <runner flag> <cpu|gpu> <rw suffix> <ro suffixes...>
gate() {
  local name=$1 flag=$2 kind=$3 rw=$4; shift 4
  local mounts=(-v "$MAIN/${OUT}_$rw:/vla/${OUT}_$rw:rw") env=() rt=(--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8)
  for ro in "$@"; do mounts+=(-v "$MAIN/${OUT}_$ro:/vla/${OUT}_$ro:ro"); done
  [[ $kind == cpu ]] && { rt=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=4); }
  [[ $name == m2equivalence ]] && mounts+=(-v "$MAIN/outputs/train/xvla_p2_force_v1_admission:/vla/outputs/train/xvla_p2_force_v1_admission:ro")
  mkdir -p "$MAIN/${OUT}_$rw"
  say "gate $name: start"
  docker run --rm --name "xvla-g1-$name" "${rt[@]}" --network none --read-only --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536 \
    -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e MUJOCO_GL=osmesa -e HOME=/tmp -e PYTHONPATH=/vla \
    -v "$PWD:/vla:ro" -v "$MAIN/$D:/vla/$D:ro" -v "$MAIN/.cache:/vla/.cache:ro" \
    -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro" \
    "${mounts[@]}" -w /vla --entrypoint python "$IMAGE" -m experiments.xvla_g1_rand.runner "$flag" \
    || { say "gate $name FAILED; stopping, nothing is trained"; exit 1; }
  say "gate $name: passed"
}

git log --oneline -1
free_gib=$(( $(df --output=avail -B1G "$MAIN" | tail -1) ))
(( free_gib >= 90 )) || { say "only ${free_gib} GiB free (< 90); stopping"; exit 1; }
[[ -z $(docker ps -q) ]] || { say "containers are running; the GPU is not idle: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; exit 1; }
gate admission --check cpu admission
gate preflight --preflight gpu preflight admission
gate resumeproof --resume-proof gpu resumeproof admission preflight
gate equivalence --equivalence gpu equivalence admission
gate m2equivalence --m2-equivalence gpu m2equivalence admission
say "all gates passed; training until $deadline"
scripts/xvla_g1_night.sh fresh "$deadline"
say "training container launched; the chain is done"
