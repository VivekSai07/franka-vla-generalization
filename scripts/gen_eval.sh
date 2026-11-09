#!/usr/bin/env bash
# G0 driver: the generalization benchmark, one step at a time. Run from a DEDICATED worktree at the merge commit (it is mounted
# read-only into every container; never switch it, never edit a bound file in it), with OUT = the campaign directory:
#
#   WT=<worktree> OUT=<campaign dir, writable> scripts/gen_eval.sh <step> ...
#
#   plan                                   layouts and their conditions (admission image)
#   feasibility <shards> <cores-each>      the unrendered expert on every job, sharded, polled with `docker ps`, merged (robosuite, CPU)
#   prepare <arm>                          A0 | A2 | A3: binds core, textures, arm, manifests, checkpoint (robosuite image)
#   smoke <arm>                            the 2 smoke layouts (12 trials) with 6 workers: several workers on purpose (FAILURES #71)
#   selection <arm>                        the selection tier, 6 workers x 7 cores (for a later arm's checkpoint choice, never for claims)
#   select <arm>                           the selection winner (arms with several checkpoints)
#   benchmark <arm>                        the benchmark tier, 6 workers x 7 cores, polled, collected, resumed once if incomplete
#   compare <arm>                          one arm's paired read-out per axis and level
#   final                                  the per-arm table
#   identity_gate <tag>                    gate D9 (iii): A2's M0 trials 1 and 291 through gen_eval, nominal, one GPU container
#   stage <arm> <tier> <ckpt> <workers> <cores-each> [first-cpu]    the generic worker pool the steps above use
#   module <python module and args>        any module of the repository, robosuite image, CPU
#
# A worker never switches checkpoint: one pool = one checkpoint (FAILURES #57). `docker wait` is never used.
# Environment: WT, OUT (required); MAIN (the main checkout, holds outputs/ and .cache/), M0 (M0's campaign, read only, for the
# identity gate), BUNDLE (the P2 bundle), TEXTURES (the texture folder), WORKERS (6), CORES (7), POLL_SECONDS (20),
# GEN_NO_RUN=1 prints the docker commands and runs nothing, DOCKER (the docker binary; tests put a fake here).
set -euo pipefail
WT="${WT:-$PWD}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
BUNDLE="${BUNDLE:-$MAIN/outputs/downloads_20261017/props_p2}"
OUT="${OUT:?set OUT to the campaign directory}"
M0="${M0:-$MAIN/outputs/vla_inputs_eval/m0_20261030}"
TEXTURES="${TEXTURES:-$MAIN/.cache/gen_textures}"
WORKERS="${WORKERS:-6}"
CORES="${CORES:-7}"
POLL_SECONDS="${POLL_SECONDS:-20}"
DOCKER="${DOCKER:-docker}"
read -ra DOCKER_CMD <<< "$DOCKER"
ROB=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADM=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc

# arm -> adapter, checkpoint update, where the checkpoint lives on the host and in the container
arm_adapter() {
  case "$1" in
    A0) echo "experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter" ;;
    A2) echo "experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter" ;;
    A3) echo "experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter" ;;
    *) echo "arm must be A0, A2 or A3, got '$1'" >&2; return 2 ;;
  esac
}
arm_update() { case "$1" in A0) echo 101520 ;; A2) echo 88830 ;; A3) echo 101520 ;; *) return 2 ;; esac; }
arm_checkpoint_host() {
  case "$1" in
    A0) echo "$BUNDLE/checkpoints/step_101520" ;;
    A2) echo "$MAIN/outputs/train/xvla_p2_force_v1/checkpoints/step_088830" ;;
    A3) echo "$MAIN/outputs/train/xvla_p2_force_box_v1/checkpoints/step_101520" ;;
  esac
}
arm_checkpoint_container() {
  case "$1" in
    A0) echo "/bundle/checkpoints/step_101520" ;;
    A2) echo "/ckpt/step_088830" ;;
    A3) echo "/ckpt/step_101520" ;;
  esac
}
arm_ckpts_dir() {   # the host directory mounted at /ckpt (A0's checkpoint comes with the bundle)
  case "$1" in
    A0) echo "" ;;
    A2) echo "$MAIN/outputs/train/xvla_p2_force_v1/checkpoints" ;;
    A3) echo "$MAIN/outputs/train/xvla_p2_force_box_v1/checkpoints" ;;
  esac
}
need_prepared() { [ -f "$OUT/arms/$1/prepared.json" ] || { echo "run prepare first: $OUT/arms/$1/prepared.json is missing" >&2; exit 2; }; }
need_checkpoint() { [ -f "$(arm_checkpoint_host "$1")/checkpoint_result.json" ] || { echo "missing checkpoint $(arm_checkpoint_host "$1")" >&2; exit 2; }; }

mountpoints() {   # the worktree is mounted read-only: every mount target must exist in it first
  [ "${GEN_NO_RUN:-0}" = 1 ] && return 0
  mkdir -p "$WT/.cache/gen_textures" "$WT/.cache/xvla_trial/bundle" "$WT/outputs/props_p2_data" "$WT/outputs/vla_inputs_signals" \
    "$WT/outputs/vla_inputs_eval"
}

docker_run() {  # docker_run <image> <gpu:0|1> <name-or-""> <detach:0|1> <cpuset-or-""> <cores> <ckpts-dir-or-""> -- <python args>
  local image="$1" gpu="$2" name="$3" detach="$4" cpuset="$5" cores="$6" ckpts="$7"; shift 8
  local args=(run --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
    -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/vla -e BUNDLE=/bundle
    -e "M0_IMAGE=$image" -e "OMP_NUM_THREADS=$cores" -e "MKL_NUM_THREADS=$cores" -e "LP_NUM_THREADS=$cores"
    -v "$WT:/vla:ro" -v "$BUNDLE:/bundle:ro" -v "$BUNDLE/runtime/xvla_base_bundle:/vla/.cache/xvla_trial/bundle:ro"
    -v "$TEXTURES:/vla/.cache/gen_textures:ro"
    -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro"
    -v "$MAIN/outputs/vla_inputs_eval:/vla/outputs/vla_inputs_eval:ro" -v "$M0:/m0:ro"
    -v "$OUT:/out:rw" -w /vla --entrypoint python)
  if [ "$gpu" = 1 ]; then args+=(--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8); else args+=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void); fi
  [ -n "$name" ] && args+=(--name "$name")
  [ -n "$ckpts" ] && args+=(-v "$ckpts:/ckpt:ro")
  [ -n "$cpuset" ] && args+=(--cpuset-cpus "$cpuset")
  if [ "$detach" = 1 ]; then args+=(-d); else args+=(--rm); fi
  args+=("$image" "$@")
  mountpoints
  if [ "${GEN_NO_RUN:-0}" = 1 ]; then echo docker "${args[@]}"; else "${DOCKER_CMD[@]}" "${args[@]}"; fi
}

cli() { docker_run "$ROB" 0 "" 0 "" 1 "" -- -m experiments.gen_eval.evaluate --root /out "$@"; }

pool_running() { [ -n "$("${DOCKER_CMD[@]}" ps -q --filter "name=$1")" ]; }
wait_for_pool() { while pool_running "$1"; do sleep "$POLL_SECONDS"; done; }

launch_workers() {  # launch_workers <arm> <tier> <ckpt> <workers> <cores> <first>
  local arm="$1" tier="$2" ckpt="$3" workers="$4" cores="$5" first="$6" k lo
  for k in $(seq 0 $((workers - 1))); do
    lo=$((first + k * cores))
    docker_run "$ROB" 1 "g0-$arm-$tier-$ckpt-$k" 1 "$lo-$((lo + cores - 1))" "$cores" "$(arm_ckpts_dir "$arm")" -- \
      -m experiments.gen_eval.evaluate --root /out worker --arm "$arm" --tier "$tier" --checkpoint "$ckpt" --shard "$k" --shards "$workers"
  done
}

stage() {  # one pinned worker pool for ONE checkpoint, polled with docker ps, collected, resumed once if incomplete, loud if still incomplete
  local arm="$1" tier="$2" ckpt="$3" workers="$4" cores="$5" first="${6:-0}" attempt k code
  launch_workers "$arm" "$tier" "$ckpt" "$workers" "$cores" "$first"
  [ "${GEN_NO_RUN:-0}" = 1 ] && return 0
  for attempt in 1 2; do
    wait_for_pool "g0-$arm-$tier-$ckpt-"
    for k in $(seq 0 $((workers - 1))); do
      code="$("${DOCKER_CMD[@]}" inspect -f '{{.State.ExitCode}}' "g0-$arm-$tier-$ckpt-$k")"
      [ "$code" = 0 ] || echo "worker $k exited $code"
      "${DOCKER_CMD[@]}" rm "g0-$arm-$tier-$ckpt-$k" >/dev/null
    done
    if cli collect --arm "$arm" --tier "$tier" --checkpoint "$ckpt"; then echo "STAGE COMPLETE $arm $tier $ckpt"; return 0; fi
    [ "$attempt" = 2 ] && break
    echo "stage incomplete: resuming the missing trials once"
    launch_workers "$arm" "$tier" "$ckpt" "$workers" "$cores" "$first"
  done
  echo "STAGE STILL INCOMPLETE $arm $tier $ckpt" >&2
  return 1
}

cmd="${1:?command}"; shift
case "$cmd" in
  plan)
    docker_run "$ADM" 0 "" 0 "" 1 "" -- -m experiments.gen_eval.evaluate --root /out plan ;;
  feasibility)
    shards="$1"; cores="$2"
    for k in $(seq 0 $((shards - 1))); do
      docker_run "$ROB" 0 "g0-feas-$k" 1 "$((k * cores))-$((k * cores + cores - 1))" "$cores" "" -- \
        -m experiments.gen_eval.feasibility --out /out/feasibility --plan /out/plan --shard "$k" --shards "$shards"
    done
    [ "${GEN_NO_RUN:-0}" = 1 ] && exit 0
    wait_for_pool g0-feas-
    bad=0
    for k in $(seq 0 $((shards - 1))); do
      code="$("${DOCKER_CMD[@]}" inspect -f '{{.State.ExitCode}}' "g0-feas-$k")"
      [ "$code" = 0 ] || { echo "feasibility shard $k exited $code" >&2; bad=1; }
      "${DOCKER_CMD[@]}" rm "g0-feas-$k" >/dev/null
    done
    [ "$bad" = 0 ] || exit 1
    docker_run "$ROB" 0 "" 0 "" 1 "" -- -m experiments.gen_eval.feasibility --out /out/feasibility --plan /out/plan --merge ;;
  prepare)
    arm="$1"; adapter="$(arm_adapter "$arm")"; update="$(arm_update "$arm")"; need_checkpoint "$arm"
    docker_run "$ROB" 1 "" 0 "" 1 "$(arm_ckpts_dir "$arm")" -- -m experiments.gen_eval.evaluate --root /out prepare --arm "$arm" \
      --adapter "$adapter" --checkpoints "$update=$(arm_checkpoint_container "$arm")" ;;
  smoke)
    arm="$1"; arm_adapter "$arm" >/dev/null; need_prepared "$arm"
    stage "$arm" smoke "$(arm_update "$arm")" "$WORKERS" "$CORES" ;;
  selection)
    arm="$1"; arm_adapter "$arm" >/dev/null; need_prepared "$arm"
    stage "$arm" selection "$(arm_update "$arm")" "$WORKERS" "$CORES" ;;
  benchmark)
    arm="$1"; arm_adapter "$arm" >/dev/null; need_prepared "$arm"
    stage "$arm" benchmark "$(arm_update "$arm")" "$WORKERS" "$CORES" ;;
  stage)
    need_prepared "$1"
    stage "$@" ;;
  select)  need_prepared "$1"; cli select --arm "$1" ;;
  compare) need_prepared "$1"; cli compare --arm "$1" --checkpoint "$(arm_update "$1")" ;;
  final)   cli final ;;
  identity_gate)
    tag="${1:?a tag for the scratch directory, e.g. 20261110}"
    [ -d "$M0/arms/A2" ] || { echo "M0's campaign $M0 has no arms/A2" >&2; exit 2; }
    need_checkpoint A2
    # M0's comparison workers ran with 7 cores and 7 CPU threads: the gate reproduces both (FAILURES #70)
    docker_run "$ROB" 1 "g0-identity-$tag" 0 "0-$((CORES - 1))" "$CORES" "$(arm_ckpts_dir A2)" -- \
      -m experiments.gen_eval.identity_gate --campaign /m0 --out "/out/identity_$tag" ;;
  module)  docker_run "$ROB" 0 "" 0 "" 1 "" -- -m "$@" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
