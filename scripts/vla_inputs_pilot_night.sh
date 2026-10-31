#!/usr/bin/env bash
# Pilot night driver: continuation pilots of the VLA-inputs programme, judged on the 192-layout pilot tier.
#
#   WT=<dedicated worktree> scripts/vla_inputs_pilot_night.sh <night_root> <variant> [<variant> ...]
#
# Run it from a DEDICATED worktree (it is mounted read-only into every container; never check out, switch or reset there
# while the night runs, and never edit a bound file in it). <night_root> is a new directory or the one of an interrupted
# night: it is also the campaign root of M0's frozen evaluation core (plan/, feasibility/, arms/), so the pilot checkpoints
# under <night_root>/training/ are visible to the evaluator through its /out mount.
#
# Arms of the pilot campaign: A0 the shared continuation control, A1.. the variants in the order given, A9 the untouched
# bundle checkpoint 101,520 (context: what continuation alone does).
#
# Order: plan and expert feasibility (once) -> control continuation, evaluation of A0 and A9 on the pilot tier -> per variant:
# zero-equivalence gate -> continuation -> prepare -> (empty selection ceremony) -> pilot-tier evaluation (WORKERS workers, one
# checkpoint per worker) -> compare against the control. Strictly sequential: training and evaluation do not fit on the GPU
# together. A failed gate, training or evaluation skips that variant, is recorded in events.jsonl and reported by the summary,
# and is NEVER retried silently: a failed training directory is kept (move it aside to run again). An interrupted evaluation
# resumes (M0's `stage` re-runs only the missing trials once); a finished step is skipped on a second invocation.
#
# Environment: WT (required), MAIN (the checkout holding outputs/, default /home/utxowqr/smolvla_franka), BUNDLE (the P2
# bundle, default $MAIN/outputs/downloads_20261017/props_p2; its runtime/xvla_base_bundle holds the X-VLA base files for training
# and for the evaluator alike), WORKERS (6), CORES
# (8 each), FEAS_SHARDS (12) FEAS_CORES (4), UPDATES (a dry run: that many updates, checkpoint step_<N>), CONTROL_DIR (a
# trained control of this recipe version to reuse), PILOT_SKIP_CONTEXT=1, PILOT_ALLOW_BUSY=1 (start although the GPU is
# in use), PILOT_NO_RUN=1 (print the docker commands, run nothing).
# Mountpoints: the driver creates the empty gitignored directories the read-only mounts need inside WT, before anything starts.
# Polling: containers are watched with `docker ps`; `docker wait` is never used (it has failed to notify here).
set -uo pipefail
WT="${WT:?set WT to a dedicated worktree}"
MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"
BUNDLE="${BUNDLE:-$MAIN/outputs/downloads_20261017/props_p2}"
BASE_BUNDLE="$BUNDLE/runtime/xvla_base_bundle"   # the one M0 mounts: the arm binds its files at prepare, workers re-check them
WORKERS="${WORKERS:-6}"; CORES="${CORES:-8}"; FEAS_SHARDS="${FEAS_SHARDS:-12}"; FEAS_CORES="${FEAS_CORES:-4}"
ROB=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADM=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
PILOT_ADAPTER=experiments.vla_inputs_arm_xvla_pilot.adapter:XVLAPilotAdapter
BUNDLE_ADAPTER=experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter
BUNDLE_CKPT=101520
POLL="${PILOT_POLL_SECONDS:-30}"
M0="$WT/scripts/vla_inputs_eval.sh"

[ $# -ge 2 ] || { echo "usage: WT=<worktree> $0 <night_root> <variant> [<variant> ...]" >&2; exit 2; }
NIGHT="$1"; shift
VARIANTS=("$@")
case "$NIGHT" in /*) ;; *) echo "night root must be an absolute path" >&2; exit 2 ;; esac
[ "${#VARIANTS[@]}" -le 8 ] || { echo "at most 8 variants (arms A1..A8)" >&2; exit 2; }
for v in "${VARIANTS[@]}"; do
  [[ "$v" =~ ^[a-z][a-z0-9_]{0,31}$ ]] || { echo "variant '$v' is not a name" >&2; exit 2; }
  [ "$v" != control ] || { echo "'control' is the shared control, not a pilot variant" >&2; exit 2; }
  [ -f "$WT/experiments/xvla_p2_pilot/variants/$v.json" ] || { echo "no variant file for '$v'" >&2; exit 2; }
done
[ -f "$M0" ] || { echo "$M0 is missing" >&2; exit 2; }
[ "$(printf '%s\n' "${VARIANTS[@]}" | sort -u | wc -l)" = "${#VARIANTS[@]}" ] || { echo "duplicate variant" >&2; exit 2; }
NO_RUN="${PILOT_NO_RUN:-0}"
[ "$NO_RUN" = 1 ] && export M0_NO_RUN=1

if [ "$NO_RUN" != 1 ]; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$busy" ] && [ "${PILOT_ALLOW_BUSY:-0}" != 1 ]; then
    echo "the GPU is in use (pids: $(echo $busy)); a pilot night needs it alone. PILOT_ALLOW_BUSY=1 to override." >&2; exit 4
  fi
  mkdir -p "$NIGHT/training" "$NIGHT/logs"
  # Docker cannot create a mountpoint inside a read-only bind mount: the (gitignored, empty) directories must exist in WT.
  for d in outputs/props_p2_data outputs/vla_inputs_signals outputs/downloads_20261017/props_p2 .cache/xvla_trial/bundle; do
    [ -d "$WT/$d" ] || mkdir -p "$WT/$d" || { echo "cannot create the mountpoint $WT/$d" >&2; exit 2; }
  done
fi
EVENTS="$NIGHT/events.jsonl"
FAILED=0

ev() {  # ev <event> <phase> <variant> [detail]: one JSON line; values are plain words
  [ "$NO_RUN" = 1 ] && { echo "EVENT $*"; return 0; }
  printf '{"t": %s, "event": "%s", "phase": "%s", "variant": "%s", "detail": "%s"}\n' "$(date +%s)" "$1" "$2" "$3" "${4:-}" >> "$EVENTS"
}

common_args() {  # common_args <gpu:0|1> <cores>
  COMMON=(--network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536
    -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/vla
    -e "OMP_NUM_THREADS=$2" -e "MKL_NUM_THREADS=$2"
    -v "$WT:/vla:ro" -v "$MAIN/outputs/props_p2_data:/vla/outputs/props_p2_data:ro"
    -v "$MAIN/outputs/vla_inputs_signals:/vla/outputs/vla_inputs_signals:ro"
    -v "$BUNDLE:/vla/outputs/downloads_20261017/props_p2:ro" -v "$BUNDLE:/bundle:ro"
    -v "$BASE_BUNDLE:/vla/.cache/xvla_trial/bundle:ro" -w /vla --entrypoint python)
  if [ "$1" = 1 ]; then COMMON+=(--runtime nvidia --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8); else COMMON+=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void); fi
}

docker_foreground() {  # docker_foreground <image> <gpu> -- <python args>: a short CPU job in the night root
  local image="$1" gpu="$2"; shift 3
  common_args "$gpu" 1
  if [ "$NO_RUN" = 1 ]; then echo docker run --rm "${COMMON[@]}" -v "$NIGHT:/out:rw" -e "M0_IMAGE=$image" "$image" "$@"; return 0; fi
  docker run --rm "${COMMON[@]}" -v "$NIGHT:/out:rw" -e "M0_IMAGE=$image" "$image" "$@"
}

run_job() {  # run_job <container name> <host out dir mounted at /out/<base>> <python args...>: detached, polled with `docker ps`
  local name="$1" out="$2"; shift 2
  common_args 1 4
  local base; base="$(basename "$out")"
  if [ "$NO_RUN" = 1 ]; then echo docker run -d --name "$name" "${COMMON[@]}" -v "$out:/out/$base:rw" "$ADM" "$@"; return 0; fi
  mkdir -p "$out"
  docker run -d --name "$name" "${COMMON[@]}" -v "$out:/out/$base:rw" "$ADM" "$@" > "$NIGHT/logs/$name.id" || return 1
  while [ -n "$(docker ps -q --filter "name=^${name}\$")" ]; do sleep "$POLL"; done
  docker logs "$name" > "$NIGHT/logs/$name.log" 2>&1 || true
  local code; code="$(docker inspect -f '{{.State.ExitCode}}' "$name")"
  docker rm "$name" > /dev/null
  [ "$code" = 0 ]
}

result_status() {  # complete | missing | <other status>
  python3 - "$1" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status', 'unknown'))
except FileNotFoundError:
    print('missing')
except Exception:
    print('unreadable')
EOF
}

checkpoint_of() {  # the update number of the single step_* directory of a finished training
  local dirs=("$1"/checkpoints/step_*)
  [ "${#dirs[@]}" = 1 ] && [ -d "${dirs[0]}" ] || return 1
  echo $((10#${dirs[0]##*/step_}))
}

train_step() {  # train_step <variant> <mode: train|zeroeq>; prints nothing; returns 0 iff complete
  local v="$1" mode="$2" dir flag out extra=()
  if [ "$mode" = zeroeq ]; then dir="$NIGHT/training/${v}_zeroeq"; flag=--zero-equivalence; else dir="$NIGHT/training/$v"; flag=--train
    [ -z "${UPDATES:-}" ] || extra=(--updates "$UPDATES"); fi
  local phase=train; [ "$mode" = zeroeq ] && phase=zero_equivalence
  if [ "$NO_RUN" != 1 ]; then
    case "$(result_status "$dir/result.json")" in
      complete) echo "skip $phase $v: complete"; return 0 ;;
      missing) [ ! -e "$dir/run_claim.json" ] || { ev fail "$phase" "$v" "a previous run left no result; not retried (move $dir aside)"; return 1; } ;;
      *) ev fail "$phase" "$v" "a previous $phase did not complete; not retried (move $dir aside)"; return 1 ;;
    esac
  fi
  ev start "$phase" "$v"
  if run_job "pilot-$phase-$v" "$dir" -m experiments.xvla_p2_pilot.runner "$flag" --variant "$v" --out "/out/$(basename "$dir")" ${extra[@]+"${extra[@]}"}; then
    ev end "$phase" "$v" ok; return 0
  fi
  ev fail "$phase" "$v" "exit non-zero; see logs/pilot-$phase-$v.log and $dir/failure.json"; return 1
}

prepare_arm() {  # prepare_arm <arm> <adapter> <ckpt> <path in /out or /bundle> [phase]
  local arm="$1" adapter="$2" ckpt="$3" path="$4" phase="${5:-}" args=()
  [ "$NO_RUN" = 1 ] || [ ! -f "$NIGHT/arms/$arm/prepared.json" ] || { echo "skip prepare $arm: prepared"; return 0; }
  [ -z "$phase" ] || args=(--phase-checkpoint "$phase")
  docker_foreground "$ROB" 0 -- -m experiments.vla_inputs_eval.evaluate --root /out prepare --arm "$arm" --adapter "$adapter" \
    --checkpoints "$ckpt=$path" ${args[@]+"${args[@]}"}
}

stage_done() {  # stage_done <arm> <tier> <ckpt>
  [ "$NO_RUN" != 1 ] && [ "$(result_status "$NIGHT/arms/$1/$2/step_$(printf '%06d' "$3")/result.json")" = complete ]
}

evaluate_arm() {  # evaluate_arm <arm> <ckpt>: the pilot tier at one checkpoint, six workers
  if stage_done "$1" comparison "$2"; then echo "skip stage $1 comparison: complete"; return 0; fi
  OUT="$NIGHT" WT="$WT" MAIN="$MAIN" BUNDLE="$BUNDLE" bash "$M0" stage "$1" comparison "$2" "$WORKERS" "$CORES"
}

selection_ceremony() {  # selection_ceremony <arm> <ckpt>: the pilot has no selection tier; M0's select needs a (empty) stage
  [ "$NO_RUN" = 1 ] || [ ! -f "$NIGHT/arms/$1/selection.json" ] || return 0
  if ! stage_done "$1" selection "$2"; then OUT="$NIGHT" WT="$WT" MAIN="$MAIN" BUNDLE="$BUNDLE" bash "$M0" stage "$1" selection "$2" 1 "$CORES" || return 1; fi
  OUT="$NIGHT" WT="$WT" MAIN="$MAIN" BUNDLE="$BUNDLE" bash "$M0" select "$1"
}

write_night() {  # write_night <ckpt>: what the summary needs to know
  [ "$NO_RUN" = 1 ] && return 0
  python3 - "$NIGHT" "$1" "${VARIANTS[@]}" <<'EOF'
import json, sys
root, ckpt, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
record = json.load(open(f'{root}/training/control/result.json'))
json.dump(dict(recipe_version=record['recipe_version'], updates=record['max_optimizer_updates'], checkpoint=ckpt,
               control_arm='A0', context_arm='A9', variants=[[n, f'A{i + 1}'] for i, n in enumerate(names)]),
          open(f'{root}/night.json', 'w'), indent=1)
EOF
}

summarise() {
  [ "$NO_RUN" = 1 ] && { echo "summary: python -m experiments.xvla_p2_pilot.summary /out"; return 0; }
  [ -f "$NIGHT/night.json" ] || return 0
  docker_foreground "$ADM" 0 -- -m experiments.xvla_p2_pilot.summary /out > "$NIGHT/logs/summary.log" 2>&1 || echo "summary failed; see $NIGHT/logs/summary.log" >&2
}

fail_variant() { FAILED=$((FAILED + 1)); summarise; }

# --- plan and feasibility (once) ------------------------------------------------------------------------------------
if [ "$NO_RUN" = 1 ] || [ ! -f "$NIGHT/plan/plan.sha256" ]; then
  ev start plan all
  docker_foreground "$ADM" 0 -- -c 'from experiments.vla_inputs_pilot_plan import plan; print("PLAN", plan.write("/out/plan"))' || { ev fail plan all "plan failed"; exit 2; }
  ev end plan all ok
fi
if [ "$NO_RUN" = 1 ] || [ ! -f "$NIGHT/feasibility/feasibility.json" ]; then
  ev start feasibility all
  export OUT="$NIGHT" WT MAIN BUNDLE
  bash "$M0" feasibility "$FEAS_SHARDS" "$FEAS_CORES" || { ev fail feasibility all "could not start"; exit 2; }
  if [ "$NO_RUN" != 1 ]; then
    while [ -n "$(docker ps -q --filter name=m0-feas)" ]; do sleep "$POLL"; done
    for k in $(seq 0 $((FEAS_SHARDS - 1))); do docker rm "m0-feas-$k" > /dev/null 2>&1 || true; done
  fi
  bash "$M0" feasibility-merge || { ev fail feasibility all "merge failed"; exit 2; }
  ev end feasibility all ok
fi

# --- the shared control and the context arm ------------------------------------------------------------------------
if [ -n "${CONTROL_DIR:-}" ] && [ "$NO_RUN" != 1 ] && [ ! -e "$NIGHT/training/control" ]; then
  [ "$(result_status "$CONTROL_DIR/result.json")" = complete ] || { echo "CONTROL_DIR is not a complete control" >&2; exit 2; }
  mkdir -p "$NIGHT/training/control" && cp -a "$CONTROL_DIR/." "$NIGHT/training/control/"   # a real copy: the evaluator's mounts follow no symlink
fi
train_step control train || { echo "the control did not train; no pilot can be judged without it" >&2; exit 2; }
if [ "$NO_RUN" = 1 ]; then CKPT="${UPDATES:-CKPT}"; else CKPT="$(checkpoint_of "$NIGHT/training/control")" || { echo "the control has no single checkpoint" >&2; exit 2; }; fi
CKPT_DIR="step_$(printf '%06d' "$CKPT" 2>/dev/null || echo "$CKPT")"
write_night "$CKPT"

ev start prepare A0
prepare_arm A0 "$PILOT_ADAPTER" "$CKPT" "/out/training/control/checkpoints/$CKPT_DIR" && ev end prepare A0 ok || { ev fail prepare control "prepare failed"; exit 2; }
ev start evaluate control
evaluate_arm A0 "$CKPT" && ev end evaluate control ok || { ev fail evaluate control "stage incomplete"; summarise; exit 2; }

if [ "${PILOT_SKIP_CONTEXT:-0}" != 1 ]; then
  ev start evaluate context
  if prepare_arm A9 "$BUNDLE_ADAPTER" "$BUNDLE_CKPT" "/bundle/checkpoints/step_$BUNDLE_CKPT" "$BUNDLE_CKPT" && evaluate_arm A9 "$BUNDLE_CKPT"; then ev end evaluate context ok
  else ev fail evaluate context "context arm incomplete (not needed for any decision)"; fi
fi

# --- one variant at a time -------------------------------------------------------------------------------------------
i=0
for v in "${VARIANTS[@]}"; do
  i=$((i + 1)); arm="A$i"
  echo "=== $v ($arm) ==="
  train_step "$v" zeroeq || { fail_variant; continue; }
  train_step "$v" train || { fail_variant; continue; }
  if [ "$NO_RUN" = 1 ]; then vckpt="$CKPT"; else vckpt="$(checkpoint_of "$NIGHT/training/$v")" || { ev fail train "$v" "no single checkpoint"; fail_variant; continue; }; fi
  vdir="step_$(printf '%06d' "$vckpt" 2>/dev/null || echo "$vckpt")"
  [ "$vckpt" = "$CKPT" ] || { ev fail train "$v" "checkpoint $vckpt differs from the control's $CKPT"; fail_variant; continue; }
  ev start evaluate "$v"
  if prepare_arm "$arm" "$PILOT_ADAPTER" "$vckpt" "/out/training/$v/checkpoints/$vdir" "$vckpt" \
     && selection_ceremony "$arm" "$vckpt" && evaluate_arm "$arm" "$vckpt"; then
    if [ "$NO_RUN" = 1 ] || [ -f "$NIGHT/arms/$arm/compare_step_$(printf '%06d' "$vckpt").json" ]; then :; else
      OUT="$NIGHT" WT="$WT" MAIN="$MAIN" BUNDLE="$BUNDLE" bash "$M0" compare "$arm" "$vckpt" || { ev fail evaluate "$v" "compare failed"; fail_variant; continue; }
    fi
    ev end evaluate "$v" ok
  else
    ev fail evaluate "$v" "prepare, selection ceremony or stage incomplete"; fail_variant; continue
  fi
  summarise
done
summarise
echo "night finished: $FAILED variant(s) without a decision; summary: $NIGHT/SUMMARY.md"
[ "$FAILED" = 0 ] || exit 3
