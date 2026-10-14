#!/usr/bin/env bash
# One overnight batch of xvla_props_p2 training, or a morning status read. Never needs polling.
#   scripts/xvla_props_p2_night.sh fresh  "tomorrow 06:00"   # first night
#   scripts/xvla_props_p2_night.sh resume "tomorrow 06:00"   # every later night
#   scripts/xvla_props_p2_night.sh check  "tomorrow 06:00"   # all pre-launch checks, prints the command, launches nothing
#   scripts/xvla_props_p2_night.sh status                    # progress, pace and GPU temperatures
# The run pauses itself at the last snapshot that fits before the deadline (exit 0, status "paused").
# `timeout` is only a backstop 45 min after the deadline; snapshots are atomic, so a kill loses at most
# the updates since the last one. The runner sleeps when the GPU reaches 83 C (back to work at 75 C).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/train/xvla_props_p2_v1
D=outputs/props_p2_data/props_p2_20261013
IMAGE=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
mode=${1:?fresh|resume|check|status}

if [[ $mode == status ]]; then
  [[ -s $OUT/result.json ]] || { echo "no run yet at $OUT"; exit 0; }
  docker ps -a --filter name=xvla-props-p2-train --format '{{.Names}}  {{.Status}}'
  python3 - "$OUT" <<'EOF'
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

stop_at=$(date -d "${2:?stop time, e.g. 'tomorrow 06:00'}" +%s)
now=$(date +%s)
(( stop_at > now + 3600 )) || { echo "deadline less than 1 h away: $(date -d @$stop_at)"; exit 1; }
backstop=$(( stop_at - now + 2700 ))
case $mode in
  fresh|check) args=(--train --stop-at "$stop_at") ;;
  resume)      args=(--train --resume --stop-at "$stop_at") ;;
  *) echo "mode must be fresh, resume, check or status"; exit 1 ;;
esac
name=xvla-props-p2-train-$(date +%m%d-%H%M)
for g in admission preflight resumeproof equivalence; do
  [[ -s ${OUT}_$g/result.json ]] || { echo "missing gate ${OUT}_$g"; exit 1; }
done
[[ -z $(docker ps -q --filter name=xvla-props-p2-train) ]] || { echo "a P2 training container is already running"; exit 1; }
if [[ $mode == resume ]]; then
  [[ -s $OUT/run_claim.json ]] || { echo "nothing to resume at $OUT"; exit 1; }
  status=$(python3 -c "import json; print(json.load(open('$OUT/result.json'))['status'])")
  [[ $status != complete ]] || { echo "training already complete"; exit 1; }
elif [[ -e $OUT/run_claim.json ]]; then
  echo "$OUT already holds a run; use resume"; exit 1
fi
free_gib=$(( $(df --output=avail -B1G . | tail -1) ))
(( free_gib >= 70 )) || { echo "only ${free_gib} GiB free; the runner needs 65 GiB"; exit 1; }
cmd=(docker run -d --name "$name" --runtime nvidia --gpus all --network none --read-only
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
  -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
  -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONPATH=/vla
  -v "$PWD:/vla:ro" -v "$PWD/$D:/vla/$D:ro" -v "$PWD/.cache:/vla/.cache:ro"
  -v "$PWD/${OUT}_admission:/vla/${OUT}_admission:ro"
  -v "$PWD/${OUT}_preflight:/vla/${OUT}_preflight:ro"
  -v "$PWD/${OUT}_resumeproof:/vla/${OUT}_resumeproof:ro"
  -v "$PWD/${OUT}_equivalence:/vla/${OUT}_equivalence:ro"
  -v "$PWD/$OUT:/vla/$OUT:rw"
  -w /vla --entrypoint timeout "$IMAGE"
  --signal=TERM "$backstop" python -m experiments.xvla_props_p2.runner "${args[@]}")
echo "mode=$mode stop_at=$(date -d @$stop_at) backstop=${backstop}s free=${free_gib}GiB head=$(git log --oneline -1)"
if [[ $mode == check ]]; then
  printf '%q ' "${cmd[@]}"; echo; echo "check only: nothing launched"; exit 0
fi
mkdir -p "$OUT"
"${cmd[@]}"
echo "launched $name; morning: scripts/xvla_props_p2_night.sh status"
