#!/usr/bin/env bash
# M3 (arm A3: force state + the target's box drawn on both cameras) evaluation on M0's frozen evaluator, one step at a time. Run
# from a DEDICATED worktree (it is mounted read-only into every container; never switch it, never edit a bound file in it) and with
# OUT set to M0's campaign directory, which holds plan/, feasibility/, arms/A0/ and arms/A2/ (A3's control):
#
#   WT=<worktree> OUT=/home/utxowqr/smolvla_franka/outputs/vla_inputs_eval/m0_20261030 scripts/vla_inputs_m3_eval.sh <step>
#
#   prepare     arm A3: the six candidate checkpoints (50,760 ... 126,900) and the matched-phase checkpoint 101,520
#   smoke       the 2 smoke layouts at the matched-phase checkpoint, one worker (proves the GPU path of the adapter)
#   selection   the six candidates on the selection tier, one worker each, all at once (79 feasible trials each)
#   select      the selection winner; selection.json lists it and 101,520 as the comparison checkpoints
#   comparison  the comparison tier at each checkpoint of selection.json, one 6-worker pool after the other
#   compare     A3 against A0 at each checkpoint of selection.json (M0's own comparison, secondary)
#   compare_a2  A3 against A2 (the primary test, spec D10): the winners, and the matched-phase pair 101,520 against 101,520
#   final       M0's Holm over A1-A4 and the amended family {A1 vs A0, A2 vs A0, A3 vs A2, A4 vs A0} with the criterion
#
# There is no pilot gate: M2's calibration showed a pilot may promote but must not drop, so a box pilot cannot spare the full run.
# Environment: CK (checkpoint directory, default the M3 training output), WORKERS (comparison pool size, 6), CORES (cores per
# comparison worker, 7), M0_NO_RUN=1 (print the docker commands, run nothing). Polling is by `docker ps`, never `docker wait`.
set -euo pipefail
step="${1:?prepare|smoke|selection|select|comparison|compare|compare_a2|final}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
WT="${WT:?set WT to the dedicated worktree}"
OUT="${OUT:?set OUT to the M0 campaign directory}"
CK="${CK:-$MAIN/outputs/train/xvla_p2_force_box_v1/checkpoints}"
WORKERS="${WORKERS:-6}"
CORES="${CORES:-7}"
ARM=A3
CONTROL=A2
ADAPTER=experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter
COMPARE_MODULE=experiments.xvla_p2_force_box.eval_compare
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
# "<A3 checkpoint> <A2 checkpoint>" per line: the winners, then the matched-phase pair (once if they coincide)
pairs() { python3 - "$OUT" <<'EOF'
import json, sys
root = sys.argv[1]
a3 = json.load(open(f"{root}/arms/A3/selection.json"))
a2 = json.load(open(f"{root}/arms/A2/selection.json"))
out = [(a3["winner"], a2["winner"])]
phase = (a3["phase_checkpoint"], a2["phase_checkpoint"])
if phase not in out:
    out.append(phase)
for a, b in out:
    print(a, b)
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
  smoke)
    [ -f "$OUT/arms/$ARM/prepared.json" ] || { echo "run prepare first" >&2; exit 1; }
    "$DRIVER" stage "$ARM" smoke "$PHASE" 1 8 0 ;;
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
    [ -f "$OUT/arms/$ARM/selection.json" ] || { echo "run select first" >&2; exit 1; }
    for c in $(decided); do "$DRIVER" stage "$ARM" comparison "$c" "$WORKERS" "$CORES" 0; done ;;
  compare)
    for c in $(decided); do "$DRIVER" compare "$ARM" "$c"; done ;;
  compare_a2)
    [ -f "$OUT/arms/$CONTROL/selection.json" ] || { echo "$CONTROL has no selection.json" >&2; exit 1; }
    pairs | while read -r a b; do
      "$DRIVER" module "$COMPARE_MODULE" --root /out compare --checkpoint "$a" --control-checkpoint "$b"
    done ;;
  final)
    "$DRIVER" final
    "$DRIVER" module "$COMPARE_MODULE" --root /out final ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
