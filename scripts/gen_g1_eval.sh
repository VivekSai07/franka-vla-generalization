#!/usr/bin/env bash
# G1 driver: arm A1 (X-VLA trained on G1's randomized data) on G0's frozen evaluator, plus the secondary M0 comparison.
# A fork of scripts/gen_eval.sh, which names only A0/A2/A3 (its tests refuse any other arm and live in the frozen gen_eval package).
# Run from a DEDICATED worktree at the merge commit (mounted read-only into every container; never switch it), with OUT = G0's
# campaign directory (outputs/gen_g0/g0_20261109: plan, feasibility, arms/A0, arms/A2):
#
#   WT=<worktree> OUT=<G0 campaign, writable> [M0C=<M0 campaign>] scripts/gen_g1_eval.sh <step> ...
#
#   prepare                      arm A1, the six candidates 50,760 ... 126,900 (robosuite image, GPU)
#   stage <tier> <ckpt> <workers> <cores-each> [first-cpu]   one worker pool for ONE checkpoint; skipped if already complete
#   selection                    the selection tier, the six candidates at once, one 8-core worker each
#   select                       the selection winner (selection.json); `winner` prints it
#   compare <ckpt>               G0's own per-arm read-out, then G1 against A2 (experiments/gen_g1_compare)
#   m0prepare | m0comparison | m0compare    secondary: M0's comparison tier at the winner (M0's driver, campaign M0C)
#
# A worker never switches checkpoint (FAILURES #57). `docker wait` is never used. GEN_NO_RUN=1 prints the docker commands.
set -euo pipefail
WT="${WT:-$PWD}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
BUNDLE="${BUNDLE:-$MAIN/outputs/downloads_20261017/props_p2}"
OUT="${OUT:?set OUT to the G0 campaign directory}"
M0C="${M0C:-$MAIN/outputs/vla_inputs_eval/m0_20261030}"
TEXTURES="${TEXTURES:-$MAIN/.cache/gen_textures}"
CK="${CK:-$MAIN/outputs/train/xvla_g1_rand_v1/checkpoints}"
G1DATA="${G1DATA:-$MAIN/outputs/gen_g1_data}"
WORKERS="${WORKERS:-6}"
CORES="${CORES:-7}"
POLL_SECONDS="${POLL_SECONDS:-20}"
DOCKER="${DOCKER:-docker}"
read -ra DOCKER_CMD <<< "$DOCKER"
ROB=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ARM=A1
ADAPTER=experiments.vla_inputs_arm_xvla_g1.adapter:XVLAG1Adapter
CANDIDATES=(50760 76140 88830 101520 114210 126900)
PHASE=101520

ckpt_dir() { printf 'step_%06d' "$1"; }
need_prepared() { [ -f "$OUT/arms/$ARM/prepared.json" ] || { echo "run prepare first: $OUT/arms/$ARM/prepared.json is missing" >&2; exit 2; }; }
stage_dir() { echo "$OUT/arms/$ARM/$1/$(ckpt_dir "$2")"; }
stage_done() { grep -q '"status": "complete"' "$(stage_dir "$1" "$2")/result.json" 2>/dev/null; }
winner() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['winner'])" "$OUT/arms/$ARM/selection.json"; }

mountpoints() {   # the worktree is mounted read-only: every mount target must exist in it first
  [ "${GEN_NO_RUN:-0}" = 1 ] && return 0
  mkdir -p "$WT/.cache/gen_textures" "$WT/.cache/xvla_trial/bundle" "$WT/outputs/props_p2_data" "$WT/outputs/vla_inputs_signals" \
    "$WT/outputs/vla_inputs_eval" "$WT/outputs/gen_g1_data"
}

docker_run() {  # docker_run <gpu:0|1> <name-or-""> <detach:0|1> <cpuset-or-""> <cores> -- <python args>
  local gpu="$1" name="$2" detach="$3" cpuset="$4" cores="$5"; shift 6
  local args=(run --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
    -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/vla -e BUNDLE=/bundle
    -e "M0_IMAGE=$ROB" -e "OMP_NUM_THREADS=$cores" -e "MKL_NUM_THREADS=$cores" -e "LP_NUM_THREADS=$cores"
    -v "$WT:/vla:ro" -v "$BUNDLE:/bundle:ro" -v "$BUNDLE/runtime/xvla_base_bundle:/vla/.cache/xvla_trial/bundle:ro"
    -v "$TEXTURES:/vla/.cache/gen_textures:ro"
    -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro"
    -v "$MAIN/outputs/vla_inputs_eval:/vla/outputs/vla_inputs_eval:ro" -v "$G1DATA:/vla/outputs/gen_g1_data:ro"
    -v "$CK:/ckpt:ro" -v "$OUT:/out:rw" -w /vla --entrypoint python)
  if [ "$gpu" = 1 ]; then args+=(--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8); else args+=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void); fi
  [ -n "$name" ] && args+=(--name "$name")
  [ -n "$cpuset" ] && args+=(--cpuset-cpus "$cpuset")
  if [ "$detach" = 1 ]; then args+=(-d); else args+=(--rm); fi
  args+=("$ROB" "$@")
  mountpoints
  if [ "${GEN_NO_RUN:-0}" = 1 ]; then echo docker "${args[@]}"; else "${DOCKER_CMD[@]}" "${args[@]}"; fi
}

cli() { docker_run 0 "" 0 "" 1 -- -m experiments.gen_eval.evaluate --root /out "$@"; }
pool_running() { [ -n "$("${DOCKER_CMD[@]}" ps -q --filter "name=$1")" ]; }
wait_for_pool() { while pool_running "$1"; do sleep "$POLL_SECONDS"; done; }

launch_workers() {  # launch_workers <tier> <ckpt> <workers> <cores> <first>
  local tier="$1" ckpt="$2" workers="$3" cores="$4" first="$5" k lo
  for k in $(seq 0 $((workers - 1))); do
    lo=$((first + k * cores))
    docker_run 1 "g1-$ARM-$tier-$ckpt-$k" 1 "$lo-$((lo + cores - 1))" "$cores" -- \
      -m experiments.gen_eval.evaluate --root /out worker --arm "$ARM" --tier "$tier" --checkpoint "$ckpt" --shard "$k" --shards "$workers"
  done
}

stage() {  # one pinned worker pool for ONE checkpoint, polled with docker ps, collected, resumed once if incomplete, loud if still incomplete
  local tier="$1" ckpt="$2" workers="$3" cores="$4" first="${5:-0}" attempt k code
  if [ "${GEN_NO_RUN:-0}" != 1 ] && stage_done "$tier" "$ckpt"; then echo "STAGE ALREADY COMPLETE $ARM $tier $ckpt"; return 0; fi
  launch_workers "$tier" "$ckpt" "$workers" "$cores" "$first"
  [ "${GEN_NO_RUN:-0}" = 1 ] && return 0
  for attempt in 1 2; do
    wait_for_pool "g1-$ARM-$tier-$ckpt-"
    for k in $(seq 0 $((workers - 1))); do
      code="$("${DOCKER_CMD[@]}" inspect -f '{{.State.ExitCode}}' "g1-$ARM-$tier-$ckpt-$k")"
      [ "$code" = 0 ] || echo "worker $k exited $code"
      "${DOCKER_CMD[@]}" rm "g1-$ARM-$tier-$ckpt-$k" >/dev/null
    done
    if cli collect --arm "$ARM" --tier "$tier" --checkpoint "$ckpt"; then echo "STAGE COMPLETE $ARM $tier $ckpt"; return 0; fi
    [ "$attempt" = 2 ] && break
    echo "stage incomplete: resuming the missing trials once"
    launch_workers "$tier" "$ckpt" "$workers" "$cores" "$first"
  done
  echo "STAGE STILL INCOMPLETE $ARM $tier $ckpt" >&2
  return 1
}

m0_driver() {  # M0's own driver against M0's campaign, with G1's sealed data and checkpoints mounted
  [ "${GEN_NO_RUN:-0}" = 1 ] || mkdir -p "$WT/outputs/gen_g1_data" "$WT/.cache/xvla_trial/bundle" "$WT/outputs/props_p2_data" \
    "$WT/outputs/vla_inputs_signals"
  local g0_campaign="$OUT"
  M0_NO_RUN="${GEN_NO_RUN:-0}" WT="$WT" MAIN="$MAIN" OUT="$M0C" BUNDLE="$BUNDLE" CKPTS="$CK" PHASE="${PHASE_M0:-}" \
    EXTRA_MOUNTS="-v $G1DATA:/vla/outputs/gen_g1_data:ro -v $g0_campaign:/g0:ro" "$WT/scripts/vla_inputs_eval.sh" "$@"
}

cmd="${1:?command}"; shift
case "$cmd" in
  prepare)
    list=""
    for c in "${CANDIDATES[@]}"; do
      [ -f "$CK/$(ckpt_dir "$c")/checkpoint_result.json" ] || { echo "missing checkpoint $CK/$(ckpt_dir "$c")" >&2; exit 2; }
      list="$list${list:+,}$c=/ckpt/$(ckpt_dir "$c")"
    done
    docker_run 1 "" 0 "" 1 -- -m experiments.gen_eval.evaluate --root /out prepare --arm "$ARM" --adapter "$ADAPTER" --checkpoints "$list" ;;
  stage)
    need_prepared
    stage "$@" ;;
  selection)
    need_prepared
    pids=()
    for i in "${!CANDIDATES[@]}"; do
      c="${CANDIDATES[$i]}"
      mkdir -p "$OUT/arms/$ARM"
      "$0" stage selection "$c" 1 8 $((i * 8)) > "$OUT/arms/$ARM/selection_stage_$c.log" 2>&1 &
      pids+=($!)
    done
    failed=0
    for k in "${!pids[@]}"; do wait "${pids[$k]}" || { echo "selection stage ${CANDIDATES[$k]} failed" >&2; failed=1; }; done
    [ "${GEN_NO_RUN:-0}" = 1 ] && { cat "$OUT"/arms/$ARM/selection_stage_*.log; exit 0; }
    tail -qn 1 "$OUT"/arms/$ARM/selection_stage_*.log
    exit $failed ;;
  select)  need_prepared; cli select --arm "$ARM" ;;
  winner)  winner ;;
  compare)
    need_prepared
    ckpt="${1:?checkpoint}"
    [ -f "$OUT/arms/$ARM/compare_step_$(printf '%06d' "$ckpt").json" ] || cli compare --arm "$ARM" --checkpoint "$ckpt"
    [ -f "$OUT/arms/$ARM/compare_A2_step_$(printf '%06d' "$ckpt").json" ] || \
      docker_run 0 "" 0 "" 1 -- -m experiments.gen_g1_compare.compare g0 --root /out --checkpoint "$ckpt" ;;
  m0prepare)
    c="$(winner)"
    [ -f "$M0C/arms/$ARM/prepared.json" ] || PHASE_M0="$c" m0_driver prepare "$ARM" "$ADAPTER" "$c=/ckpt/$(ckpt_dir "$c")" ;;
  m0comparison)
    m0_driver stage "$ARM" comparison "$(winner)" "$WORKERS" "$CORES" 0 ;;
  m0compare)
    c="$(winner)"
    [ -f "$M0C/arms/$ARM/compare_A2_step_$(printf '%06d' "$c")_vs_088830.json" ] || \
      m0_driver module experiments.gen_g1_compare.compare m0 --root /out --g0-root /g0 --checkpoint "$c" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
