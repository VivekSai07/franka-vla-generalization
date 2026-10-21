#!/usr/bin/env bash
# Props P3 evaluation driver: staged, parallel, resumable. Run from the repo root (or the P3 worktree root).
#
#   scripts/xvla_props_p3_eval.sh feas              # expert feasibility of the 288 reserved layouts (CPU, unrendered)
#   scripts/xvla_props_p3_eval.sh prepare           # verify checkpoints + seal + feasibility once (admission image)
#   scripts/xvla_props_p3_eval.sh smoke [WORKERS]   # 2 trials on checkpoint 126,900: measures cost and proves the path
#   scripts/xvla_props_p3_eval.sh all   [WORKERS]   # screen -> report -> confirm -> final
#
# Rules (fixed in docs/superpowers/plans/2026-10-21-props-p3-training-eval.md) are applied by `evaluate.py decide`;
# this script only launches workers and collects. It polls `docker ps` (docker wait has failed to notify here) and
# resumes an incomplete stage once (a worker that died for infrastructure reasons leaves missing trial folders; a
# resumed trial equals an uninterrupted one because each trial reseeds itself). Checkpoints are evaluated under
# OSMesa, the renderer they were trained on; never under the EGL image.
set -uo pipefail
cd "$(dirname "$0")/.."
# Code is mounted from this tree; data (gitignored outputs, .cache) from OUT_ROOT, the main checkout by default
# when run from a worktree (OUT_ROOT=/path to override).
OUT_ROOT=${OUT_ROOT:-$(cd "$(git rev-parse --git-common-dir)/.." && pwd)}
ROBO=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADMIT=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
EVAL=outputs/xvla_props_p3_eval
LOG=$OUT_ROOT/$EVAL/driver.log
CANDS=50760,76140,88830,101520,114210,126900
LAST=126900
mkdir -p "$OUT_ROOT/$EVAL"
# Bind-mount targets must exist inside the read-only code mount; in a worktree the gitignored dirs do not.
mkdir -p "$PWD/$EVAL" "$PWD/outputs/props_p3_data" "$PWD/outputs/train" "$PWD/.cache"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

common=(--network none --read-only --user "$(id -u):$(id -g)" --ulimit nofile=65536:65536
        -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
        -e PYTHONPATH=/vla -v "$PWD:/vla:ro" -v "$OUT_ROOT/.cache:/vla/.cache:ro"
        -v "$OUT_ROOT/outputs/props_p3_data:/vla/outputs/props_p3_data:ro" -v "$OUT_ROOT/outputs/train:/vla/outputs/train:ro"
        -v "$OUT_ROOT/$EVAL:/vla/$EVAL:rw" -w /vla --entrypoint python)

# Pure steps (prepare, collect, decide, final): no GPU, no simulator.
cpu() {
  docker run --rm --runtime runc "${common[@]}" --tmpfs /tmp:rw,size=2g -e NVIDIA_VISIBLE_DEVICES=void \
    "$ADMIT" -m experiments.xvla_props_p3_eval.evaluate "$@" 2>&1 | grep -v -i warn | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

CORES=5   # dedicated cores per worker: unpinned workers each saw 48 cores and thrashed (FAILURES #57)
launch() { # mode candidates shard shards slot
  local first=$(( $5 * CORES ))
  docker run -d --name "p3eval_${1}_${3}" --runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --cpuset-cpus "${first}-$((first + CORES - 1))" -e OMP_NUM_THREADS=$CORES -e MKL_NUM_THREADS=$CORES \
    -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=$CORES --tmpfs /tmp:rw,size=16g "${common[@]}" "$ROBO" \
    -m experiments.xvla_props_p3_eval.evaluate worker --mode "$1" --candidates "$2" --shard "$3" --shards "$4" >/dev/null
}

# stage "mode:candidates:workers"   run concurrently, then collect.
stage() {
  local spec attempt mode cands workers k names slot
  for attempt in 1 2; do
    names=(); slot=0
    for spec in "$@"; do
      IFS=: read -r mode cands workers <<<"$spec"
      for ((k = 0; k < workers; k++)); do
        launch "$mode" "$cands" "$k" "$workers" "$slot" && names+=("p3eval_${mode}_${k}"); slot=$((slot + 1))
      done
    done
    log "stage attempt $attempt: $* -> ${#names[@]} workers"
    while [ -n "$(docker ps -q -f name=p3eval_)" ]; do
      log "running $(docker ps -q -f name=p3eval_ | wc -l) workers; gpu: $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
      sleep 120
    done
    for n in "${names[@]}"; do
      code=$(docker inspect -f '{{.State.ExitCode}}' "$n"); [ "$code" != 0 ] && log "worker $n exit $code: $(docker logs "$n" 2>&1 | tail -3 | tr '\n' ' ')"
      docker logs "$n" > "$OUT_ROOT/$EVAL/logs_${n}_attempt$attempt.txt" 2>&1; docker rm "$n" >/dev/null
    done
    local ok=1
    for spec in "$@"; do
      IFS=: read -r mode cands workers <<<"$spec"
      cpu collect --mode "$mode" --candidates "$cands" | grep -q " incomplete " && ok=0
    done
    [ $ok = 1 ] && return 0
    log "stage incomplete after attempt $attempt"
  done
  log "STOP: stage still incomplete after one resume"; return 1
}

feas() { # 12 unrendered CPU shards, 4 cores each, then merge
  mkdir -p "$OUT_ROOT/$EVAL/feasibility"
  for k in $(seq 0 11); do
    docker run -d --name "p3feas_$k" --runtime runc --cpuset-cpus "$((4 * k))-$((4 * k + 3))" \
      -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=4 --tmpfs /tmp:rw,size=8g "${common[@]}" "$ROBO" \
      -m experiments.xvla_props_p3_eval.feasibility --out "/vla/$EVAL/feasibility" --shard "$k" --shards 12 >/dev/null
  done
  while [ -n "$(docker ps -q -f name=p3feas_)" ]; do sleep 20; done
  for k in $(seq 0 11); do
    code=$(docker inspect -f '{{.State.ExitCode}}' "p3feas_$k"); [ "$code" != 0 ] && { log "feasibility shard $k exit $code"; return 1; }
    docker rm "p3feas_$k" >/dev/null
  done
  docker run --rm --runtime runc "${common[@]}" --tmpfs /tmp:rw,size=2g "$ROBO" \
    -m experiments.xvla_props_p3_eval.feasibility --out "/vla/$EVAL/feasibility" --merge 2>&1 | grep -v -i warn | tee -a "$LOG"
}

case "${1:-}" in
  feas) feas || exit 1 ;;
  prepare) cpu prepare || exit 1 ;;
  smoke)
    w=${2:-1}
    stage "smoke:$LAST:$w" || exit 1
    log SMOKE_DONE ;;
  all)
    total=${2:-8}
    stage "screen:$CANDS:$total" || exit 1
    cpu decide screen || exit 1
    surv=$(python3 -c "import json;print(','.join(map(str,json.load(open('$OUT_ROOT/$EVAL/decisions/screen.json'))['survivors'])))")
    stage "report:$surv:$total" || exit 1
    cpu decide report || exit 1
    win=$(python3 -c "import json;print(json.load(open('$OUT_ROOT/$EVAL/decisions/report.json'))['winner'])")
    stage "confirm:$win:$total" || exit 1
    cpu final || exit 1
    log ALL_DONE ;;
  *) echo "usage: $0 feas|prepare|smoke|all [workers]"; exit 2 ;;
esac
