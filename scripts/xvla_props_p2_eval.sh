#!/usr/bin/env bash
# Props P2 evaluation driver: staged, parallel, resumable. Run from the repo root.
#
#   scripts/xvla_props_p2_eval.sh smoke [WORKERS_PER_POLICY]   # 2 trials on P2 and on P1 126,900: measures cost
#   scripts/xvla_props_p2_eval.sh all   [TOTAL_WORKERS]        # screen -> report -> control+confirm -> final
#
# Rules (fixed in docs/superpowers/plans/2026-10-16-props-p2-evaluation.md) are applied by `evaluate.py decide`;
# this script only launches workers and collects. It polls `docker ps` (docker wait has failed to notify here), keeps
# going if the user disconnects when started under nohup, and resumes an incomplete stage once (a worker that died
# for infrastructure reasons leaves missing trial folders; a resumed trial equals an uninterrupted one).
set -uo pipefail
cd "$(dirname "$0")/.."
ROBO=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADMIT=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
EVAL=outputs/xvla_props_p2_eval
LOG=$EVAL/driver.log
CANDS=50760,76140,88830,101520,114210,126900
P1=126900
mkdir -p "$EVAL"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

common=(--network none --read-only --user "$(id -u):$(id -g)" --ulimit nofile=65536:65536
        -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
        -e PYTHONPATH=/vla -v "$PWD:/vla:ro" -v "$PWD/.cache:/vla/.cache:ro"
        -v "$PWD/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$PWD/outputs/train:/vla/outputs/train:ro"
        -v "$PWD/$EVAL:/vla/$EVAL:rw" -w /vla --entrypoint python)

# Pure steps (collect, decide, final): no GPU, no simulator.
cpu() {
  docker run --rm --runtime runc "${common[@]}" --tmpfs /tmp:rw,size=2g -e NVIDIA_VISIBLE_DEVICES=void \
    -v "$PWD/outputs/props_p1_data:/vla/outputs/props_p1_data:ro" "$ADMIT" -m experiments.xvla_props_p2_eval.evaluate "$@" 2>&1 \
    | grep -v -i warn | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

CORES=5   # dedicated cores per worker: 8 unpinned workers each saw 48 cores, spawned ~48 threads each and
          # thrashed (load 142 on 48 cores, no trial finished in 6 min); pinned, thread pools size to CORES.
launch() { # mode policy candidates shard shards slot
  local first=$(( $6 * CORES ))
  docker run -d --name "p2eval_${1}_${2}_${4}" --runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --cpuset-cpus "${first}-$((first + CORES - 1))" -e OMP_NUM_THREADS=$CORES -e MKL_NUM_THREADS=$CORES \
    -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=$CORES --tmpfs /tmp:rw,size=16g "${common[@]}" "$ROBO" \
    -m experiments.xvla_props_p2_eval.evaluate worker --mode "$1" --policy "$2" --candidates "$3" --shard "$4" --shards "$5" >/dev/null
}

# stage "mode:policy:candidates:workers" ...   all specs run concurrently, then every one is collected.
stage() {
  local specs=("$@") spec attempt
  for attempt in 1 2; do
    local names=() slot=0
    for spec in "${specs[@]}"; do
      IFS=: read -r mode policy cands workers <<<"$spec"
      for ((k = 0; k < workers; k++)); do launch "$mode" "$policy" "$cands" "$k" "$workers" "$slot" && names+=("p2eval_${mode}_${policy}_${k}"); slot=$((slot + 1)); done
    done
    log "stage attempt $attempt: ${specs[*]} -> ${#names[@]} workers"
    while [ -n "$(docker ps -q -f name=p2eval_)" ]; do
      local t; t=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
      log "running $(docker ps -q -f name=p2eval_ | wc -l) workers; gpu: $t"
      sleep 120
    done
    for n in "${names[@]}"; do
      code=$(docker inspect -f '{{.State.ExitCode}}' "$n"); [ "$code" != 0 ] && log "worker $n exit $code: $(docker logs "$n" 2>&1 | tail -3 | tr '\n' ' ')"
      docker logs "$n" > "$EVAL/logs_${n}_attempt$attempt.txt" 2>&1; docker rm "$n" >/dev/null
    done
    local ok=1
    for spec in "${specs[@]}"; do
      IFS=: read -r mode policy cands workers <<<"$spec"
      cpu collect --mode "$mode" --policy "$policy" --candidates "$cands" | grep -q " incomplete " && ok=0
    done
    [ $ok = 1 ] && return 0
    log "stage incomplete after attempt $attempt"
  done
  log "STOP: stage still incomplete after one resume"; return 1
}

case "${1:-}" in
  smoke)
    w=${2:-1}
    stage "smoke:p2:$P1:$w" "smoke:p1:$P1:$w" || exit 1
    log SMOKE_DONE ;;
  all)
    total=${2:-8}
    stage "screen:p2:$CANDS:$total" || exit 1
    cpu decide screen || exit 1
    surv=$(python3 -c "import json;print(','.join(map(str,json.load(open('$EVAL/decisions/screen.json'))['survivors'])))")
    n=$(awk -F, '{print NF}' <<<"$surv")
    p1w=$(( total / (n + 1) )); [ "$p1w" -lt 1 ] && p1w=1
    stage "report:p2:$surv:$((total - p1w))" "report:p1:$P1:$p1w" || exit 1
    cpu decide report || exit 1
    win=$(python3 -c "import json;print(json.load(open('$EVAL/decisions/report.json'))['winner'])")
    q=$(( total / 3 )); [ "$q" -lt 1 ] && q=1
    stage "control:p2:$win:$q" "control:p1:$P1:$q" "confirm:p2:$win:$q" || exit 1
    cpu final || exit 1
    log ALL_DONE ;;
  *) echo "usage: $0 smoke|all [workers]"; exit 2 ;;
esac
