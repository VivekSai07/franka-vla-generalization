#!/usr/bin/env bash
# G1 evaluation chain, detached, supervised, resumable. Run a COPY from the dedicated evaluation worktree (a script that is edited or
# merged while bash executes it misbehaves):
#
#   cp scripts/gen_g1_chain.sh /tmp/g1_chain.sh
#   WT=/home/utxowqr/smolvla_franka/.worktrees/g1-eval OUT=<G0 campaign> setsid nohup /tmp/g1_chain.sh >/dev/null 2>&1 &
#
# prepare A1 (six candidates) -> smoke -> selection tier (six candidates at once) -> select -> benchmark tier at the winner and at
# 101,520 -> G1 against A2 per axis (experiments/gen_g1_compare) -> secondary: M0's comparison tier at the winner against A2.
# Every step is skipped when its result file exists, so re-running the same command resumes. It stops at the first failing step,
# never kills or deletes anything, polls with `docker ps` and result files (never `docker wait`), waits while another job holds the
# GPU, and stops when free disk falls below MIN_FREE_GB (ask the user). Cancel before the next step: touch "$OUT/chain_g1.cancel".
# Progress:  tail -f "$OUT/chain_g1.log"      state: cat "$OUT/chain_g1.state"      done: "$OUT/chain_g1.done"
set -u
MAIN=${MAIN:-/home/utxowqr/smolvla_franka}
WT=${WT:?set WT to the dedicated evaluation worktree}
OUT=${OUT:-$MAIN/outputs/gen_g0/g0_20261109}
M0C=${M0C:-$MAIN/outputs/vla_inputs_eval/m0_20261030}
DRIVER="$WT/scripts/gen_g1_eval.sh"
MIN_FREE_GB=${MIN_FREE_GB:-40}
STEP=${CHAIN_POLL_SECONDS:-120}
MAX_GPU_WAIT=${MAX_GPU_WAIT:-21600}
PHASE=101520
LOG="$OUT/chain_g1.log"
export WT OUT M0C MAIN
mkdir -p "$OUT"

log() { echo "$(date '+%F %T') [g1-chain] $*" | tee -a "$LOG"; }
state() { echo "$(date '+%F %T') $*" > "$OUT/chain_g1.state"; }
die() { log "STOP: $*"; state "stopped: $*"; exit "${2:-1}"; }
cancelled() { [ -e "$OUT/chain_g1.cancel" ] && die "cancelled by $OUT/chain_g1.cancel" 0; }
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
mine_running() { [ -n "$(docker ps -q --filter name=g1-A1- --filter name=m0-A1-)" ]; }
free_gb() { df -BG --output=avail "$MAIN" | tail -1 | tr -dc 0-9; }
done_json() { grep -q '"status": "complete"' "$1" 2>/dev/null; }

reap() {   # finished worker containers of an interrupted run keep their names: remove them so a resumed stage can start
  local id name code
  for id in $(docker ps -aq --filter name=g1-A1- --filter name=m0-A1- --filter status=exited); do
    name=$(docker inspect -f '{{.Name}}' "$id"); code=$(docker inspect -f '{{.State.ExitCode}}' "$id")
    log "removing finished container $name (exit $code)"; docker rm "$id" >/dev/null
  done
}

before_gpu_step() {
  cancelled
  while mine_running; do log "our containers are still running; waiting"; sleep "$STEP"; done
  reap
  local waited=0
  while gpu_busy; do
    [ "$waited" -lt "$MAX_GPU_WAIT" ] || die "GPU held by another job for ${MAX_GPU_WAIT}s; re-run the chain to resume" 3
    log "GPU in use by someone else; waiting"; sleep "$STEP"; waited=$((waited + STEP))
  done
  [ "$(free_gb)" -ge "$MIN_FREE_GB" ] || die "only $(free_gb) GB free (< $MIN_FREE_GB): ask the user" 4
}

step() {   # step <name> <skip-file-or-"" > <command...>: skip when the result file is complete, stop on failure
  local name="$1" skip="$2"; shift 2
  if [ -n "$skip" ] && { [ -s "$skip" ] && { [[ "$skip" != */result.json ]] || done_json "$skip"; }; }; then log "skip $name (done: ${skip#$OUT/})"; return 0; fi
  before_gpu_step
  log "start $name"; state "running $name"
  local t0; t0=$(date +%s)
  "$@" >> "$LOG" 2>&1 || die "$name failed (exit $?); see $LOG; re-run the chain to resume"
  log "done $name in $(( ($(date +%s) - t0) / 60 )) min; free $(free_gb) GB"
}

stage_result() { echo "$OUT/arms/A1/$1/step_$(printf '%06d' "$2")/result.json"; }
D() { "$DRIVER" "$@"; }

log "chain start: WT=$WT OUT=$OUT M0C=$M0C; free $(free_gb) GB; $(nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)"
[ -x "$DRIVER" ] || die "missing driver $DRIVER"
[ -f "$OUT/arms/A2/prepared.json" ] || die "G0's A2 arm is not prepared in $OUT"

step "prepare A1 (six candidates)" "$OUT/arms/A1/prepared.json" D prepare
step "smoke at 126900" "$(stage_result smoke 126900)" D stage smoke 126900 6 7
CANDIDATES="50760 76140 88830 101520 114210 126900"
selection_done() { local c; for c in $CANDIDATES; do done_json "$(stage_result selection "$c")" || return 1; done; }
if selection_done; then log "skip selection tier (all six candidates complete)"; else step "selection tier, six candidates" "" D selection; fi
for c in $CANDIDATES; do
  done_json "$(stage_result selection "$c")" || die "selection stage $c is not complete"
done
step "select" "$OUT/arms/A1/selection.json" D select
WIN=$(D winner) || die "no selection winner"
log "selection winner: $WIN (candidates and counts: $(tr -d '\n ' < "$OUT/arms/A1/selection.json" | head -c 300))"

step "benchmark tier at the winner $WIN" "$(stage_result benchmark "$WIN")" D stage benchmark "$WIN" 6 7
step "compare at the winner $WIN" "$OUT/arms/A1/compare_A2_step_$(printf '%06d' "$WIN").json" D compare "$WIN"
if [ "$WIN" != "$PHASE" ]; then
  step "benchmark tier at $PHASE" "$(stage_result benchmark "$PHASE")" D stage benchmark "$PHASE" 6 7
  step "compare at $PHASE" "$OUT/arms/A1/compare_A2_step_$(printf '%06d' "$PHASE").json" D compare "$PHASE"
fi

step "M0 prepare (secondary)" "$M0C/arms/A1/prepared.json" D m0prepare
step "M0 comparison tier at $WIN (secondary)" "$M0C/arms/A1/comparison/step_$(printf '%06d' "$WIN")/result.json" D m0comparison
step "M0 compare (secondary)" "$M0C/arms/A1/compare_A2_step_$(printf '%06d' "$WIN")_vs_088830.json" D m0compare

state "complete"; log "chain complete"
date '+%F %T' > "$OUT/chain_g1.done"
