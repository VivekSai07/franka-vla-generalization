#!/usr/bin/env bash
# One batch of xvla_g1_rand training up to a deadline, or a status read. Never needs polling.
#   scripts/xvla_g1_night.sh fresh  "tomorrow 08:00"   # first batch
#   scripts/xvla_g1_night.sh resume "tomorrow 08:00"   # every later batch, with a new deadline
#   scripts/xvla_g1_night.sh check  "tomorrow 08:00"   # all pre-launch checks, prints the command, launches nothing
#   scripts/xvla_g1_night.sh status                    # progress, pace and GPU temperatures
# The run pauses itself at the last snapshot that fits before the deadline (exit 0, status "paused").
# `timeout` is only a backstop 45 min after the deadline; snapshots are atomic, so a kill loses at most the updates since the last
# one. The runner sleeps when the GPU reaches 83 C (back to work at 75 C).
# Run it from the checkout that holds the gates' source (the detached worktree .worktrees/g1-run): that tree is mounted as /vla.
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=${G1_MAIN:-/home/utxowqr/smolvla_franka}
OUT=outputs/train/xvla_g1_rand_v1
D=outputs/gen_g1_data/g1_20261110
IMAGE=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
mode=${1:?fresh|resume|check|status}
# gitignored trees live in the main checkout, not in a worktree
OUTR=$MAIN/$OUT

if [[ $mode == status ]]; then
  [[ -s $OUTR/result.json ]] || { echo "no run yet at $OUTR"; exit 0; }
  docker ps -a --filter name=xvla-g1-train --format '{{.Names}}  {{.Status}}'
  python3 - "$OUTR" <<'EOF'
import json, sys
out = sys.argv[1]
r = json.load(open(f"{out}/result.json"))
done, total = r.get("completed_updates", 0), r.get("max_optimizer_updates", 126900)
print(f"status={r.get('status')} stage={r.get('stage')} updates={done}/{total} ({100 * done / total:.1f}%)")
print("checkpoints:", [m.get("update") for m in r.get("checkpoints", [])])
print("snapshots written:", (r.get("snapshot_updates_written") or [])[-3:], "resumes:", len(r.get("resumes", [])))
rows = [json.loads(l) for l in open(f"{out}/metrics.jsonl")]
recent = rows[-2000:]
pace = sum(x["seconds"] for x in recent) / len(recent)
gpu = [x["gpu"] for x in rows if x.get("gpu")]
print(f"pace={pace:.3f} s/update  remaining~{(total - done) * pace / 3600:.1f} h  last loss={rows[-1]['loss']:.4f}")
if gpu:
    print(f"GPU max {max(g['temperature_c'] for g in gpu)} C, max {max(g['power_w'] for g in gpu):.0f} W of "
          f"{gpu[-1]['power_limit_w']:.0f} W, thermal slowdowns {sum(g['thermal_slowdown'] for g in gpu)}, "
          f"cool-down pauses {sum(1 for x in rows if x.get('thermal_pause_seconds'))}")
EOF
  exit 0
fi

stop_at=$(date -d "${2:?stop time, e.g. 'tomorrow 08:00'}" +%s)
now=$(date +%s)
(( stop_at > now + 3600 )) || { echo "deadline less than 1 h away: $(date -d @$stop_at)"; exit 1; }
backstop=$(( stop_at - now + 2700 ))
case $mode in
  fresh)  args=(--train --stop-at "$stop_at") ;;
  resume) args=(--train --resume --stop-at "$stop_at") ;;
  check)  if [[ -e $OUTR/run_claim.json ]]; then mode=check-resume; args=(--train --resume --stop-at "$stop_at")
          else args=(--train --stop-at "$stop_at"); fi ;;
  *) echo "mode must be fresh, resume, check or status"; exit 1 ;;
esac
name=xvla-g1-train-$(date +%m%d-%H%M)
for g in admission preflight resumeproof equivalence m2equivalence; do
  [[ -s ${OUTR}_$g/result.json ]] || { echo "missing gate ${OUTR}_$g"; exit 1; }
done
[[ -z $(docker ps -q --filter name=xvla-g1-train) ]] || { echo "a G1 training container is already running"; exit 1; }
if [[ $mode == resume || $mode == check-resume ]]; then
  [[ -s $OUTR/run_claim.json ]] || { echo "nothing to resume at $OUTR"; exit 1; }
  status=$(python3 -c "import json; print(json.load(open('$OUTR/result.json'))['status'])")
  [[ $status != complete ]] || { echo "training already complete"; exit 1; }
  ls -d "$OUTR"/resume/snap_* >/dev/null 2>&1 || { echo "no resume snapshot in $OUTR/resume"; exit 1; }
elif [[ -e $OUTR/run_claim.json ]]; then
  echo "$OUTR already holds a run; use resume"; exit 1
fi
free_gib=$(( $(df --output=avail -B1G "$MAIN" | tail -1) ))
(( free_gib >= 70 )) || { echo "only ${free_gib} GiB free; the runner needs 65 GiB"; exit 1; }
cmd=(docker run -d --name "$name" --runtime nvidia --gpus all --network none --read-only
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
  -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
  -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONPATH=/vla
  -v "$PWD:/vla:ro" -v "$MAIN/$D:/vla/$D:ro" -v "$MAIN/.cache:/vla/.cache:ro"
  -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro"
  -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro"
  -v "${OUTR}_admission:/vla/${OUT}_admission:ro"
  -v "${OUTR}_preflight:/vla/${OUT}_preflight:ro"
  -v "${OUTR}_resumeproof:/vla/${OUT}_resumeproof:ro"
  -v "${OUTR}_equivalence:/vla/${OUT}_equivalence:ro"
  -v "${OUTR}_m2equivalence:/vla/${OUT}_m2equivalence:ro"
  -v "$OUTR:/vla/$OUT:rw"
  -w /vla --entrypoint timeout "$IMAGE"
  --signal=TERM "$backstop" python -m experiments.xvla_g1_rand.runner "${args[@]}")
echo "mode=$mode stop_at=$(date -d @$stop_at) backstop=${backstop}s free=${free_gib}GiB head=$(git log --oneline -1)"
if [[ $mode == check || $mode == check-resume ]]; then
  printf '%q ' "${cmd[@]}"; echo; echo "check only: nothing launched"; exit 0
fi
mkdir -p "$OUTR"
"${cmd[@]}"
echo "launched $name; status: scripts/xvla_g1_night.sh status"
