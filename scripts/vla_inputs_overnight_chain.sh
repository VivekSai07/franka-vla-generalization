#!/usr/bin/env bash
# Overnight chain after the M2 training: break, dry night, pilot night 1 (force), M2 selection tier. Run detached:
#   cp scripts/vla_inputs_overnight_chain.sh /tmp/m2_chain.sh && nohup /tmp/m2_chain.sh > outputs/train/m2_overnight_chain.log 2>&1 &
# (run a COPY: a script that is edited or merged while bash executes it misbehaves)
# Cancel at any time (it stops before the next phase; running containers are not touched): touch outputs/train/m2_overnight_chain.cancel
#
# 1. wait for the M2 training to finish (result.json complete, no training container); abort if it died and nobody is resuming it
# 2. a one-hour break (the GPU cools down), then check the GPU is free and the disk has room
# 3. dry night: the pilot driver with PILOT_SMOKE_ONLY=1 and 40-update continuations: plan, feasibility, control, zero-equivalence gate,
#    force training, prepare, selection ceremony and the 2 smoke layouts on the GPU. Judged from its artefacts, not its exit code.
# 4. pilot night 1, force only: control continuation, the control on the pilot tier, force gate + continuation + pilot tier + compare,
#    then the driver's own summary (SUMMARY.md). The target-box pilot (3 h) does not fit in the window and moves to the next night.
# 5. M2 `prepare` and the selection tier (six candidates at once, about 2 h). The comparison tier is NEVER started here: it needs the
#    pilot decision to be recorded and an explicit PILOT_DECISION_RECORDED=1.
# If the dry night fails, the pilots are skipped and step 5 runs instead. A phase is started only if its estimate fits before DEADLINE.
# Nothing is ever deleted or killed. Polling is by `docker ps`, never `docker wait`.
set -u
MAIN=${MAIN:-/home/utxowqr/smolvla_franka}
STEP=${CHAIN_POLL_SECONDS:-60}
OUT_TRAIN=$MAIN/outputs/train/xvla_p2_force_v1
LOG_PREFIX="[chain]"
CANCEL=$MAIN/outputs/train/m2_overnight_chain.cancel
PILOT_WT=$MAIN/.worktrees/pilot-run
EVAL_WT=$MAIN/.worktrees/m2-eval
CAMPAIGN=$MAIN/outputs/vla_inputs_eval/m0_20261030
STAMP=$(date +%Y%m%d)
DRY=$MAIN/outputs/vla_inputs_eval/m0_pilot_dry_$STAMP
NIGHT=$MAIN/outputs/vla_inputs_eval/m0_pilot_night1_$STAMP
BREAK_SECONDS=${BREAK_SECONDS:-3600}
DEADLINE=${DEADLINE:-$(date -d "tomorrow 08:00" +%s)}
# Estimates in seconds, from measured rates (docs/BENCHMARK_STATUS.md): dry night, pilot night (force only), M2 selection tier.
EST_DRY=${EST_DRY:-2700}; EST_NIGHT=${EST_NIGHT:-22500}; EST_SELECTION=${EST_SELECTION:-7500}

log() { echo "$(date '+%F %T') $LOG_PREFIX $*"; }
left() { echo $((DEADLINE - $(date +%s))); }
cancelled() { [ -e "$CANCEL" ] && { log "cancelled by $CANCEL"; return 0; }; return 1; }
status() { python3 - "$OUT_TRAIN/result.json" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", "?"))
except Exception:
    print("none")
EOF
}
training_running() { [ -n "$(docker ps --filter name=xvla-p2-force-train --filter status=running -q)" ]; }
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
artefact() { python3 - "$1" <<'EOF'
import json, sys
try:
    sys.exit(0 if json.load(open(sys.argv[1])).get("status") in ("complete", "passed") else 1)
except Exception:
    sys.exit(1)
EOF
}
no_failures() { ! grep -q '"event": "fail"' "$1/events.jsonl" 2>/dev/null; }

log "start; deadline $(date -d @"$DEADLINE" '+%F %T'); break ${BREAK_SECONDS}s"

# 1. wait for the training
while :; do
  cancelled && exit 0
  if ! training_running; then
    case "$(status)" in
      complete) break ;;
      *) if pgrep -f xvla_p2_force_supervisor > /dev/null; then :; else log "training is not running, not complete and nobody resumes it (status $(status)); aborting"; exit 1; fi ;;
    esac
  fi
  sleep "$STEP"
done
[ -f "$OUT_TRAIN/checkpoints/step_126900/checkpoint_result.json" ] || { log "training says complete but step_126900 has no marker; aborting"; exit 1; }
log "training complete: $(python3 -c "import json; r=json.load(open('$OUT_TRAIN/result.json')); print(r.get('completed_updates'), 'updates, elapsed_h', round(r.get('elapsed_seconds',0)/3600,2))")"

# 2. break, then checks
end_break=$(( $(date +%s) + BREAK_SECONDS ))
while [ "$(date +%s)" -lt "$end_break" ]; do cancelled && exit 0; sleep "$STEP"; done
log "break over; GPU $(nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader)"
for i in $(seq 1 10); do gpu_busy || break; log "GPU in use by someone else; waiting"; sleep "$STEP"; done
gpu_busy && { log "GPU still in use; aborting"; exit 1; }
free_gb=$(df -BG --output=avail "$MAIN" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "${MIN_FREE_GB:-60}" ] || { log "only ${free_gb} GB free; aborting"; exit 1; }

selection_phase() {
  cancelled && return 0
  [ "$(left)" -ge "$EST_SELECTION" ] || { log "M2 selection needs ~${EST_SELECTION}s, $(left)s left: not started"; return 0; }
  gpu_busy && { log "GPU busy: M2 selection not started"; return 0; }
  log "M2 selection: prepare"
  export WT=$EVAL_WT OUT=$CAMPAIGN
  "$EVAL_WT/scripts/vla_inputs_m2_eval.sh" prepare 2>&1 | grep -E "PREPARED|Error|Traceback|missing|refus" ; [ -f "$CAMPAIGN/arms/A2/prepared.json" ] || { log "M2 prepare failed"; return 1; }
  log "M2 selection: six candidates, one worker each"
  "$EVAL_WT/scripts/vla_inputs_m2_eval.sh" selection; log "selection stages ended with code $?"
  "$EVAL_WT/scripts/vla_inputs_m2_eval.sh" select 2>&1 | tail -5; log "M2 select ended with code $?"
  [ -f "$CAMPAIGN/arms/A2/selection.json" ] && log "selection.json: $(cat "$CAMPAIGN/arms/A2/selection.json" | head -c 400)"
}

# 3. dry night
cancelled && exit 0
if [ "$(left)" -lt $((EST_DRY + EST_NIGHT)) ]; then
  log "not enough time for the dry night and the pilot night ($(left)s left, need $((EST_DRY + EST_NIGHT))s): M2 selection only"
  selection_phase; exit 0
fi
[ ! -e "$DRY" ] || DRY="${DRY}_$(date +%H%M)"
log "dry night in $DRY"
PILOT_SMOKE_ONLY=1 PILOT_SKIP_CONTEXT=1 UPDATES=40 WT=$PILOT_WT "$PILOT_WT/scripts/vla_inputs_pilot_night.sh" "$DRY" force > "$MAIN/outputs/train/m2_overnight_chain_dry.log" 2>&1
log "dry night driver exit code $?"
ok=1
for f in training/control training/force_zeroeq training/force; do artefact "$DRY/$f/result.json" || { log "dry night: $f not complete"; ok=0; }; done
for f in arms/A0/smoke/step_000040 arms/A1/smoke/step_000040; do artefact "$DRY/$f/result.json" || { log "dry night: $f not complete"; ok=0; }; done
no_failures "$DRY" || { log "dry night: events.jsonl holds a failure"; ok=0; }

if [ "$ok" = 1 ] && ! cancelled; then
  log "dry night passed"
  # 4. pilot night 1 (force only), reusing the dry night's plan and feasibility (CPU work, identical)
  if [ "$(left)" -lt "$EST_NIGHT" ]; then log "pilot night needs ~${EST_NIGHT}s, $(left)s left"; else
    mkdir -p "$NIGHT" && cp -r "$DRY/plan" "$DRY/feasibility" "$NIGHT/"
    log "pilot night 1 in $NIGHT (force only)"
    PILOT_SKIP_CONTEXT=1 WT=$PILOT_WT "$PILOT_WT/scripts/vla_inputs_pilot_night.sh" "$NIGHT" force > "$MAIN/outputs/train/m2_overnight_chain_night.log" 2>&1
    log "pilot night driver exit code $?; summary: $NIGHT/SUMMARY.md"
    [ -f "$NIGHT/SUMMARY.md" ] && sed -n 1,40p "$NIGHT/SUMMARY.md"
  fi
else
  log "dry night failed: the pilots are skipped; see $DRY/logs and $MAIN/outputs/train/m2_overnight_chain_dry.log"
fi

# 5. M2 selection tier (never the comparison tier)
selection_phase
log "chain finished"
