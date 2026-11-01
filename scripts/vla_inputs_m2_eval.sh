#!/usr/bin/env bash
# M2 (arm A2) evaluation on M0's frozen evaluator, one step at a time. Run from a DEDICATED worktree (it is mounted read-only
# into every container; never switch it, never edit a bound file in it) and with OUT set to M0's campaign directory, which holds
# plan/, feasibility/ and arms/A0/:
#
#   WT=<worktree> OUT=/home/utxowqr/smolvla_franka/outputs/vla_inputs_eval/m0_20261030 scripts/vla_inputs_m2_eval.sh <step>
#
#   prepare     arm A2: the six candidate checkpoints (50,760 ... 126,900) and the matched-phase checkpoint 101,520
#   selection   the six candidates on the selection tier, one worker each, all at once (79 feasible trials each)
#   select      the selection winner; selection.json lists it and 101,520 as the comparison checkpoints
#   comparison  the comparison tier at each checkpoint of selection.json, one 6-worker pool after the other.
#               Refuses unless PILOT_DECISION_RECORDED=1: M2 is not run on the comparison tier before pilot night 1's decision.
#   compare     A2 against A0 at each checkpoint of selection.json
#
# Environment: CK (checkpoint directory, default the M2 training output), WORKERS (comparison pool size, 6), CORES (cores per
# comparison worker, 7), M0_NO_RUN=1 (print the docker commands, run nothing). Polling is by `docker ps`, never `docker wait`.
set -euo pipefail
step="${1:?prepare|selection|select|comparison|compare}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
WT="${WT:?set WT to the dedicated worktree}"
OUT="${OUT:?set OUT to the M0 campaign directory}"
CK="${CK:-$MAIN/outputs/train/xvla_p2_force_v1/checkpoints}"
WORKERS="${WORKERS:-6}"
CORES="${CORES:-7}"
ARM=A2
ADAPTER=experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter
CANDIDATES=(50760 76140 88830 101520 114210 126900)
PHASE=101520
DRIVER="$WT/scripts/vla_inputs_eval.sh"
export WT MAIN OUT
export CKPTS="$CK"

ckpt_dir() { printf 'step_%06d' "$1"; }
decided() { python3 - "$OUT/arms/$ARM/selection.json" <<'EOF'
import json, sys
print(" ".join(str(c) for c in json.load(open(sys.argv[1]))["comparison_checkpoints"]))
EOF
}

case "$step" in
  prepare)
    list=""
    for c in "${CANDIDATES[@]}"; do
      [ -f "$CK/$(ckpt_dir "$c")/checkpoint_result.json" ] || { echo "missing checkpoint $(ckpt_dir "$c") in $CK" >&2; exit 1; }
      list="$list${list:+,}$c=/ckpt/$(ckpt_dir "$c")"
    done
    PHASE="$PHASE" "$DRIVER" prepare "$ARM" "$ADAPTER" "$list" ;;
  selection)
    [ -f "$OUT/arms/$ARM/prepared.json" ] || { echo "run prepare first" >&2; exit 1; }
    pids=()
    for i in "${!CANDIDATES[@]}"; do
      c="${CANDIDATES[$i]}"
      "$DRIVER" stage "$ARM" selection "$c" 1 8 $((i * 8)) > "$OUT/arms/$ARM/selection_stage_$c.log" 2>&1 &
      pids+=($!)
    done
    [ "${M0_NO_RUN:-0}" = 1 ] && { wait; cat "$OUT"/arms/$ARM/selection_stage_*.log; exit 0; }
    failed=0
    for k in "${!pids[@]}"; do wait "${pids[$k]}" || { echo "selection stage ${CANDIDATES[$k]} failed" >&2; failed=1; }; done
    tail -n 1 "$OUT"/arms/$ARM/selection_stage_*.log
    exit $failed ;;
  select)
    "$DRIVER" select "$ARM" ;;
  comparison)
    [ "${PILOT_DECISION_RECORDED:-0}" = 1 ] || { echo "refusing: pilot night 1's decision is not recorded (set PILOT_DECISION_RECORDED=1 once it is)" >&2; exit 1; }
    for c in $(decided); do "$DRIVER" stage "$ARM" comparison "$c" "$WORKERS" "$CORES" 0; done ;;
  compare)
    for c in $(decided); do "$DRIVER" compare "$ARM" "$c"; done ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
