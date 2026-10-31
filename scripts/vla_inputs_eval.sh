#!/usr/bin/env bash
# M0 driver. Run from the worktree that holds the code (it is mounted read-only), with:
#   OUT=<campaign dir, writable>  BUNDLE=<P2 bundle>  M0_NO_RUN=1 to print the docker commands instead of running them
#
#   plan                                   layouts (admission image)
#   feasibility <shards> <cores-each>      unrendered expert on every layout, sharded, then merged (robosuite image, CPU)
#   prepare <arm> <module:Class> <upd=path[,upd=path]>
#   stage <arm> <tier> <checkpoint> <workers> <cores-each> [first-cpu]
#                                          one pinned worker pool for ONE checkpoint, polled with `docker ps`, collected,
#                                          resumed once if incomplete, loud if still incomplete
#   select <arm> | compare <arm> <checkpoint> | final
#
# A worker never switches checkpoint: one pool = one checkpoint (FAILURES #57). `docker wait` is never used.
set -euo pipefail
WT="${WT:-$PWD}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
BUNDLE="${BUNDLE:-$MAIN/outputs/downloads_20261017/props_p2}"
OUT="${OUT:?set OUT to the campaign directory}"
ROB=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADM=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc

docker_run() {  # docker_run <image> <gpu:0|1> <name-or-""> <detach:0|1> <cpuset-or-""> <cores> -- <python args>
  local image="$1" gpu="$2" name="$3" detach="$4" cpuset="$5" cores="$6"; shift 7
  local args=(run --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
    -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/vla -e BUNDLE=/bundle
    -e "M0_IMAGE=$image" -e "OMP_NUM_THREADS=$cores" -e "MKL_NUM_THREADS=$cores" -e "LP_NUM_THREADS=$cores"
    -v "$WT:/vla:ro" -v "$BUNDLE:/bundle:ro" -v "$BUNDLE/runtime/xvla_base_bundle:/vla/.cache/xvla_trial/bundle:ro"
    -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro"
    -v "$OUT:/out:rw" -w /vla --entrypoint python)
  [ "$gpu" = 1 ] && args+=(--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8) || args+=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void)
  [ -n "$name" ] && args+=(--name "$name")
  [ -n "$cpuset" ] && args+=(--cpuset-cpus "$cpuset")
  [ "$detach" = 1 ] && args+=(-d) || args+=(--rm)
  args+=("$image" "$@")
  if [ "${M0_NO_RUN:-0}" = 1 ]; then echo docker "${args[@]}"; else docker "${args[@]}"; fi
}

cli() { docker_run "$ROB" 0 "" 0 "" 1 -- -m experiments.vla_inputs_eval.evaluate --root /out "$@"; }

cmd="${1:?command}"; shift
case "$cmd" in
  plan)
    docker_run "$ADM" 0 "" 0 "" 1 -- -m experiments.vla_inputs_eval.evaluate --root /out plan ;;
  feasibility)
    shards="$1"; cores="$2"
    for k in $(seq 0 $((shards - 1))); do
      docker_run "$ROB" 0 "m0-feas-$k" 1 "$((k * cores))-$((k * cores + cores - 1))" "$cores" -- \
        -m experiments.vla_inputs_eval.feasibility --out /out/feasibility --plan /out/plan --shard "$k" --shards "$shards"
    done
    echo "poll: docker ps --filter name=m0-feas; then: $0 feasibility-merge" ;;
  feasibility-merge)
    docker_run "$ROB" 0 "" 0 "" 1 -- -m experiments.vla_inputs_eval.feasibility --out /out/feasibility --plan /out/plan --merge ;;
  prepare)
    arm="$1"; adapter="$2"; ckpts="$3"
    docker_run "$ROB" 1 "" 0 "" 1 -- -m experiments.vla_inputs_eval.evaluate --root /out prepare --arm "$arm" --adapter "$adapter" --checkpoints "$ckpts" ;;
  stage)
    arm="$1"; tier="$2"; ckpt="$3"; workers="$4"; cores="$5"; first="${6:-0}"
    for k in $(seq 0 $((workers - 1))); do
      lo=$((first + k * cores))
      docker_run "$ROB" 1 "m0-$arm-$tier-$ckpt-$k" 1 "$lo-$((lo + cores - 1))" "$cores" -- \
        -m experiments.vla_inputs_eval.evaluate --root /out worker --arm "$arm" --tier "$tier" --checkpoint "$ckpt" --shard "$k" --shards "$workers"
    done
    [ "${M0_NO_RUN:-0}" = 1 ] && exit 0
    for attempt in 1 2; do
      while [ -n "$(docker ps -q --filter "name=m0-$arm-$tier-$ckpt-")" ]; do sleep 20; done
      for k in $(seq 0 $((workers - 1))); do
        code="$(docker inspect -f '{{.State.ExitCode}}' "m0-$arm-$tier-$ckpt-$k")"
        [ "$code" = 0 ] || echo "worker $k exited $code"
        docker rm "m0-$arm-$tier-$ckpt-$k" >/dev/null
      done
      if cli collect --arm "$arm" --tier "$tier" --checkpoint "$ckpt"; then echo "STAGE COMPLETE $arm $tier $ckpt"; exit 0; fi
      [ "$attempt" = 2 ] && break
      echo "stage incomplete: resuming the missing trials once"
      for k in $(seq 0 $((workers - 1))); do
        lo=$((first + k * cores))
        docker_run "$ROB" 1 "m0-$arm-$tier-$ckpt-$k" 1 "$lo-$((lo + cores - 1))" "$cores" -- \
          -m experiments.vla_inputs_eval.evaluate --root /out worker --arm "$arm" --tier "$tier" --checkpoint "$ckpt" --shard "$k" --shards "$workers"
      done
    done
    echo "STAGE STILL INCOMPLETE $arm $tier $ckpt" >&2; exit 1 ;;
  select)  cli select --arm "$1" ;;
  compare) cli compare --arm "$1" --checkpoint "$2" ;;
  final)   cli final ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
