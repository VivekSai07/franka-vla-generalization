#!/usr/bin/env bash
# M3b evaluation on M0's frozen evaluator: arms A5-A9, no policy is trained, comparison tier at checkpoint 101,520 only (no selection
# tier, no `select`, no M0 `compare`). One step at a time. Run from a DEDICATED worktree (it is mounted read-only into every container;
# never switch it, never edit a bound file in it) and with OUT set to M0's campaign directory, which holds plan/, feasibility/ and the
# prepared arms A0, A2 and A3:
#
#   WT=<worktree> OUT=/home/utxowqr/smolvla_franka/outputs/vla_inputs_eval/m0_20261030 scripts/vla_inputs_m3b_eval.sh <step> [arm]
#
#   identity_gate        the live equality gate (spec D7 iv): a zero-corruption adapter re-runs A3's comparison trials 1 and 291 and
#                        every trace array must equal A3's recorded one bit for bit. OUT is a SCRATCH directory (never the campaign);
#                        CAMPAIGN (default M0's campaign) is mounted read-only at /campaign and is only read. It runs under M0's
#                        worker runtime (deterministic torch algorithms, CUBLAS_WORKSPACE_CONFIG) and the workers' CPU thread settings
#   prepare <arm>        arm A5..A9 at checkpoint 101,520 (the single and matched-phase checkpoint)
#   smoke <arm>          the 2 smoke layouts, one worker
#   comparison <arm>     the comparison tier: one pool of WORKERS workers on CORES cores each (6 x 7)
#   analyse              R, the paired tests, failure classes and the detector's error (the m3b analysis module; writes m3b_analysis.json)
#
# A5 (the predicted box) needs DETECTOR_DIR, the host directory of the trained detector's final weights, mounted read-only at
# /vla/$DETECTOR_REL for prepare, smoke, comparison and analyse (an arm input directory bound by `prepare`).
# Environment: CK (checkpoint directory, default the M3 training output), WORKERS (6), CORES (7), DETECTOR_DIR, DETECTOR_REL,
# CAMPAIGN, M0_NO_RUN=1 (print the docker commands, run nothing). Polling is by `docker ps`, never `docker wait`.
set -euo pipefail
step="${1:?identity_gate|prepare|smoke|comparison|analyse}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
WT="${WT:?set WT to the dedicated worktree}"
OUT="${OUT:?set OUT to the campaign directory (a scratch directory for identity_gate)}"
CK="${CK:-$MAIN/outputs/train/xvla_p2_force_box_v1/checkpoints}"
CAMPAIGN="${CAMPAIGN:-$MAIN/outputs/vla_inputs_eval/m0_20261030}"
WORKERS="${WORKERS:-6}"
CORES="${CORES:-7}"
CHECKPOINT=101520
DETECTOR_REL="${DETECTOR_REL:-outputs/vla_inputs_box_detector/run_v1/final}"
DRIVER="$WT/scripts/vla_inputs_eval.sh"
NOISY=experiments.vla_inputs_arm_xvla_force_box_noisy.adapter
export WT MAIN OUT
export CKPTS="$CK"

adapter_of() {
  case "$1" in
    A5) echo experiments.vla_inputs_arm_xvla_force_box_pred.adapter:XVLAPredictedBoxAdapter ;;
    A6) echo "$NOISY:XVLAJitterSmall" ;;
    A7) echo "$NOISY:XVLAJitterLarge" ;;
    A8) echo "$NOISY:XVLADropout" ;;
    A9) echo "$NOISY:XVLASwap" ;;
    *) echo "arm must be one of A5 A6 A7 A8 A9, not '$1'" >&2; exit 2 ;;
  esac
}
need_detector() {
  [ -n "${DETECTOR_DIR:-}" ] && [ -d "$DETECTOR_DIR" ] || { echo "arm A5 needs DETECTOR_DIR, the trained detector's directory" >&2; exit 1; }
  export EXTRA_MOUNTS="-v $DETECTOR_DIR:/vla/$DETECTOR_REL:ro"
}
need_prepared() { [ -f "$OUT/arms/$1/prepared.json" ] || { echo "run prepare first" >&2; exit 1; }; }

case "$step" in
  identity_gate)
    [ -d "$CAMPAIGN" ] || { echo "campaign $CAMPAIGN does not exist" >&2; exit 1; }
    [ "$(cd "$CAMPAIGN" && pwd -P)" != "$(mkdir -p "$OUT" && cd "$OUT" && pwd -P)" ] || { echo "OUT must be a scratch directory, not the campaign" >&2; exit 1; }
    EXTRA_MOUNTS="-v $CAMPAIGN:/campaign:ro" MODULE_CORES="$CORES" "$DRIVER" gpu_module experiments.vla_inputs_m3b_analysis.identity_gate \
      --campaign /campaign --out /out --checkpoint "$CHECKPOINT" --ordinals 1 291 ;;
  prepare)
    arm="${2:-}"; adapter="$(adapter_of "$arm")"
    [ "$arm" != A5 ] || need_detector
    [ -f "$CK/step_$CHECKPOINT/checkpoint_result.json" ] || { echo "missing checkpoint step_$CHECKPOINT in $CK" >&2; exit 1; }
    PHASE="$CHECKPOINT" "$DRIVER" prepare "$arm" "$adapter" "$CHECKPOINT=/ckpt/step_$CHECKPOINT" ;;
  smoke)
    arm="${2:-}"; adapter_of "$arm" >/dev/null
    [ "$arm" != A5 ] || need_detector
    need_prepared "$arm"
    "$DRIVER" stage "$arm" smoke "$CHECKPOINT" 1 8 0 ;;
  comparison)
    arm="${2:-}"; adapter_of "$arm" >/dev/null
    [ "$arm" != A5 ] || need_detector
    need_prepared "$arm"
    "$DRIVER" stage "$arm" comparison "$CHECKPOINT" "$WORKERS" "$CORES" 0 ;;
  analyse)
    [ -z "${DETECTOR_DIR:-}" ] || need_detector
    "$DRIVER" module experiments.vla_inputs_m3b_analysis.analyse --root /out analyse ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
