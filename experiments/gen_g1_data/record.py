"""G1 collector: 16 parallel shards of contiguous plan slots, then a verified merge and the seal. A fork of `props_p2_data.record`.

Modes (one per container):
  --gate <dir> --shard k --shards n          the expert gate: 96 first attempts across the ranges, unrendered
  --gate-summary <dir>                       the bars (D4, R3) over the 96 rows
  --determinism <dir> --tag t                12 rendered gate attempts; run twice, compare in --gate-summary
  --pilot <shards_dir> --shard k --slots ... named slots into a scratch shard
  --collect <shards_dir> --shard k [--resume]
  --status <shards_dir>
  --merge <shards_dir> --out <sealed_root>   aggregate, verify against the shards, validate, seal

Each shard: its own journal (one line per attempt), its own LeRobot dataset, finalized per episode without reconstructing the dataset
object (FAILURES #41), resumable alone. Per admitted episode: `episodes/episode_NNNN.npz` (the trace and the scene) and
`signals/episode_NNNN.npz` (S0's arrays), both before the dataset episode, so a crash between them and the journal line leaves only files
the resumed run overwrites. The scheduler (`run_schedule`) imports no simulator and is tested in the admission image.
"""
import argparse
import copy
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import contract
from experiments.gen_g1_data import plan as p

REJECTION_BUDGET = 150
MAX_ATTEMPTS_PER_SLOT = p.ATTEMPTS_PER_LAYOUT * (p.MAX_SUBSTITUTIONS + 1)
CHECK_EVERY = 5
MUJOCO_VERSION = "3.4.0"
TRACE_KEYS = ("qpos", "ctrl", "sim_time", "props_pose", "phase_index", "substeps", "success_tick")
DETERMINISM_ATTEMPTS = 12
# D4 and R3: first-attempt admission of the 96 gate attempts. The unit is the whole admission (expert success, undisturbed distractors,
# exact replay, labels, schema, signals), which is stricter than the expert's success alone; both are reported.
GATE_OVERALL, GATE_PER_STRATUM = 0.95, 0.90
GATE_FIELDS = ("object", "quintile", "distractor_count")


class SlotExhausted(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


# --- journal -------------------------------------------------------------------------------------

def read_journal(path):
    """Journal lines; a truncated final line (killed mid-write) is dropped, any other bad line raises."""
    lines = Path(path).read_text().splitlines()
    records = []
    for n, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise
    return records


def _append(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def substitutions_of(journal):
    return {str(j["slot"]): int(j["r"]) for j in journal if j.get("admitted") and int(j.get("r", 0)) > 0}


# --- scheduler -----------------------------------------------------------------------------------

def run_schedule(plan, execute, journal_path, commit, slots, resume=False, max_attempts=MAX_ATTEMPTS_PER_SLOT,
                 strict=True, budget=REJECTION_BUDGET, check=None, dataset_count=None, log=print, tag="G1"):
    """Admit every slot in `slots` once, in order.

    A rejected attempt retries the same slot: attempt a uses scene r = a // 3 (r > 0 is a deterministic substitute scene of the same
    slot, inside the same stratum) and expert draw k = a % 3. `commit` runs before the journal line, so a journal line never claims
    an episode the dataset lacks. `budget` is per shard.
    """
    journal_path = Path(journal_path)
    existing = read_journal(journal_path) if journal_path.exists() else []
    if existing and not resume:
        raise FileExistsError(f"{journal_path} is not empty; use resume")
    if resume and not existing:
        raise ValueError("Nothing to resume")
    if any(j["slot"] not in slots for j in existing):
        raise ValueError("Journal holds slots outside this shard")
    admitted_slots = {j["slot"] for j in existing if j.get("admitted")}
    if resume and dataset_count is not None and dataset_count() != len(admitted_slots):
        raise ValueError(f"Dataset holds {dataset_count()} episodes but the journal admitted {len(admitted_slots)}")
    rejected = sum(1 for j in existing if not j.get("admitted"))
    if rejected > budget:
        raise BudgetExhausted(f"{rejected} rejected attempts already exceed the budget of {budget}")
    attempts_done = Counter(j["slot"] for j in existing)
    if check:
        check(len(admitted_slots))
    for slot in slots:
        if slot in admitted_slots:
            continue
        episode = plan["episodes"][slot]
        attempt = attempts_done[slot]
        while True:
            if attempt >= max_attempts:
                if strict:
                    raise SlotExhausted(f"Slot {slot} failed {attempt} attempts on {attempt // p.ATTEMPTS_PER_LAYOUT} scenes")
                break
            r, k = divmod(attempt, p.ATTEMPTS_PER_LAYOUT)
            layout = p.resample_start(episode, r)
            draw = p.attempt_draw(episode, k)
            started = time.monotonic()
            result, frames, traces, verdict = execute(layout, draw)
            wall = time.monotonic() - started
            record = {"slot": slot, "attempt": attempt, "r": r, "k": k, "layout_id": layout["layout_id"],
                      "seed": draw["seed"], "admitted": bool(verdict["admitted"]), "reason": verdict["reason"],
                      "gates": verdict["gates"], "expert_reason": verdict.get("details", {}).get("expert_reason"),
                      "distractor_motion": verdict.get("details", {}).get("distractor_motion"),
                      "distractor_touched": verdict.get("details", {}).get("distractor_touched"),
                      "wall_seconds": round(wall, 1)}
            if verdict["admitted"]:
                started = time.monotonic()
                commit(layout, draw, frames, result, traces)
                record["commit_seconds"] = round(time.monotonic() - started, 1)
                _append(journal_path, record)
                admitted_slots.add(slot)
            else:
                _append(journal_path, record)
                rejected += 1
            log(f"{tag} slot={slot} attempt={attempt} r={r} k={k} admitted={record['admitted']} "
                f"reason={record['reason']} wall={wall:.1f}s admitted_total={len(admitted_slots)} rejected_total={rejected}")
            if verdict["admitted"]:
                if check and len(admitted_slots) % CHECK_EVERY == 0:
                    check(len(admitted_slots))
                break
            if rejected > budget:
                raise BudgetExhausted(f"{rejected} rejected attempts exceed the budget of {budget}")
            attempt += 1
    return {"admitted": len(admitted_slots), "rejected": rejected}


# --- real execution --------------------------------------------------------------------------------

def make_execute(render=True):
    from experiments.gen_g1_data.episode import run_attempt
    from experiments.gen_g1_data.validate import admit

    def execute(episode, draw):
        frames, traces = [], []

        def sink(frame, trace):
            frames.append({key: np.copy(v) if isinstance(v, np.ndarray) else v for key, v in frame.items()})
            traces.append({key: np.copy(trace[key]) for key in TRACE_KEYS})

        result = run_attempt(episode, draw, sink, render=render)
        return result, frames, traces, admit(episode, result, frames, traces, images=render)

    return execute


def create_dataset(root, repo_id):
    from experiments.paired_demonstrations.media import ENCODING, FEATURES, PairedDataset
    dataset = PairedDataset.create(
        repo_id=repo_id, root=Path(root) / "dataset", fps=30, features=copy.deepcopy(FEATURES),
        vcodec=ENCODING["vcodec"], encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)
    # One file per episode, so a shard is readable after every finalize and the merge copies videos verbatim.
    dataset.meta.info["data_files_size_in_mb"] = 0
    dataset.meta.info["video_files_size_in_mb"] = 0
    return dataset


def open_dataset(root, repo_id):
    """Reopen a finalized shard for `--resume`; once per process, never per episode."""
    from experiments.paired_demonstrations.media import ENCODING, PairedDataset
    dataset = PairedDataset(repo_id=repo_id, root=Path(root) / "dataset", video_backend="pyav")
    dataset.vcodec = ENCODING["vcodec"]
    dataset._encoder_threads = ENCODING["encoder_threads"]
    dataset.batch_encoding_size = 1
    dataset._streaming_encoder = None
    dataset.meta.info["data_files_size_in_mb"] = 0
    dataset.meta.info["video_files_size_in_mb"] = 0
    return dataset


def episode_name(episode):
    return f"episode_{episode['episode_index']:04d}.npz"


def trace_arrays(episode, draw, result, traces):
    """What `episodes/episode_NNNN.npz` holds: the recorded physics trace and the scene that produced it. `validate.check_recorded_files`
    reads the identity fields back; a replay needs `initial_state`, the condition and the draw."""
    return dict(
        qpos=np.stack([t["qpos"] for t in traces]), ctrl=np.stack([t["ctrl"] for t in traces]),
        props_pose=np.stack([t["props_pose"] for t in traces]), phase_index=np.array([int(t["phase_index"]) for t in traces]),
        substeps=np.array([int(t["substeps"]) for t in traces]), sim_time=np.array([float(t["sim_time"]) for t in traces]),
        success_tick=np.array([bool(t["success_tick"]) for t in traces]),
        initial_state=result["initial_state"], final_qpos=result["final_qpos"],
        object=np.asarray(episode["object"]), bin=np.asarray(episode["bin"]), blue_slot=np.asarray(episode["blue_slot"]),
        start=np.asarray(episode["start"]), distractors=np.asarray([d["object"] for d in episode["distractors"]]),
        distractor_poses=np.asarray([d["pose"] for d in episode["distractors"]], dtype=np.float64).reshape(-1, 3),
        primitives=np.asarray(result["primitives"]), layout_id=np.asarray(episode["layout_id"]),
        condition_sha256=np.asarray(episode["condition_sha256"]), condition=np.asarray(json.dumps(episode["condition"])),
        home_pose_jitter_deg=np.asarray(draw["home_pose_jitter_deg"], dtype=np.float64),
        draw=np.asarray(json.dumps({"seed": int(draw["seed"]), "home_pose_jitter_deg": list(draw["home_pose_jitter_deg"])})))


def write_npz(path, arrays):
    """Atomic; replaces a file a killed earlier run left behind (the journal, not the file, says an episode is admitted)."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


class Committer:
    """Writes an admitted episode and finalizes it immediately, without reconstructing the dataset object."""

    def __init__(self, root, dataset):
        self.root, self.dataset = Path(root), dataset
        (self.root / "episodes").mkdir(exist_ok=True)
        (self.root / "signals").mkdir(exist_ok=True)

    def count(self):
        return self.dataset.meta.total_episodes

    def __call__(self, episode, draw, frames, result, traces):
        write_npz(self.root / "episodes" / episode_name(episode), trace_arrays(episode, draw, result, traces))
        write_npz(self.root / "signals" / episode_name(episode), result["signals"])
        for frame in frames:
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self.dataset.finalize()


def make_check(root, provenance):
    root = Path(root)

    def check(admitted):
        free = shutil.disk_usage(root).free
        if free < contract.MIN_FREE_BYTES:
            raise RuntimeError(f"Only {free / 1024**3:.1f} GB free; need {contract.MIN_FREE_BYTES / 1024**3:.0f}")
        if contract.provenance() != provenance:
            raise RuntimeError("provenance changed during collection; aborting before the seal")

    return check


def _require_pinned_runtime():
    import mujoco
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f"MuJoCo {mujoco.__version__} is not the pinned {MUJOCO_VERSION}; run in the robosuite image")


def shard_root(shards_dir, k):
    return Path(shards_dir) / f"shard_{k:02d}"


# --- modes -----------------------------------------------------------------------------------------

def collect_shard(shards_dir, k, shards, resume=False, slots=None, pilot=False):
    _require_pinned_runtime()
    root = shard_root(shards_dir, k)
    plan = p.make_plan()
    provenance = contract.provenance()
    slots = list(slots) if slots is not None else p.shard_slots(k, shards)
    repo_id = f"gen_g1_shard_{k:02d}"
    if resume:
        if json.loads((root / "plan.json").read_text()) != json.loads(json.dumps(plan)):
            raise SystemExit("plan.json differs from the plan of record; not resuming")
        if json.loads((root / "provenance.json").read_text()) != provenance:
            raise SystemExit("provenance.json differs from the current source; not resuming")
        if json.loads((root / "slots.json").read_text()) != slots:
            raise SystemExit("slots.json differs from this shard's slots; not resuming")
        dataset = open_dataset(root, repo_id)
    else:
        if root.exists():
            raise SystemExit(f"No overwrite: {root} exists")
        if shutil.disk_usage(Path(shards_dir)).free < contract.MIN_FREE_BYTES:
            raise SystemExit("Not enough free disk")
        root.mkdir(parents=True)
        _write_json(root / "plan.json", plan)
        _write_json(root / "provenance.json", provenance)
        _write_json(root / "slots.json", slots)
        dataset = create_dataset(root, repo_id)
    committer = Committer(root, dataset)
    try:
        summary = run_schedule(plan, make_execute(), root / "journal.jsonl", committer, slots, resume=resume,
                               strict=not pilot, check=make_check(root, provenance),
                               dataset_count=committer.count if resume else None, tag=f"G1 shard={k}")
    finally:
        committer.dataset.finalize()
    _write_json(root / "substitutions.json", substitutions_of(read_journal(root / "journal.jsonl")))
    _write_json(root / "done.json", {**summary, "slots": [slots[0], slots[-1]]})
    print(f"G1 SHARD {k} DONE {summary}", flush=True)


def status(shards_dir):
    total = rejected = 0
    for root in sorted(Path(shards_dir).glob("shard_*")):
        journal = read_journal(root / "journal.jsonl") if (root / "journal.jsonl").exists() else []
        admitted = sum(j["admitted"] for j in journal)
        bad = len(journal) - admitted
        total, rejected = total + admitted, rejected + bad
        reasons = Counter(j["reason"] for j in journal if not j["admitted"])
        print(f"{root.name}: admitted={admitted} rejected={bad} done={(root / 'done.json').exists()} {dict(reasons)}")
    print(f"TOTAL admitted={total}/{p.ADMIT_COUNT} rejected={rejected}")


def gate_row(attempt, result, verdict, wall):
    return {"index": attempt["episode_index"], "layout_id": attempt["layout_id"], "object": attempt["object"],
            "bin": attempt["bin"], "blue_slot": attempt["blue_slot"], "quintile": attempt["quintile"],
            "region": attempt["region"], "distractor_count": attempt["distractor_count"],
            "primitive_count": attempt["primitive_count"], "table_offset_m": attempt["condition"]["table_offset_m"],
            "admitted": bool(verdict["admitted"]), "reason": verdict["reason"], "expert_reason": result["reason"],
            "expert_success": bool(verdict["gates"]["expert_success"]), "gates": verdict["gates"],
            "distractor_motion": result["distractor_motion"], "distractor_touched": verdict["details"].get("distractor_touched"),
            "wall_seconds": round(wall, 1)}


def gate_shard(out, k, shards):
    """Unrendered expert gate over this worker's share of the 96 gate attempts (rendering does not feed physics)."""
    _require_pinned_runtime()
    plan = p.make_plan()
    attempts = plan["gate_attempts"][k::shards]
    execute = make_execute(render=False)
    path = Path(out) / f"gate_{k:02d}.jsonl"
    if path.exists():
        raise SystemExit(f"No overwrite: {path}")
    for attempt in attempts:
        started = time.monotonic()
        result, frames, traces, verdict = execute(attempt, p.attempt_draw(attempt, 0))
        _append(path, gate_row(attempt, result, verdict, time.monotonic() - started))
    print(f"G1 GATE SHARD {k} DONE {len(attempts)}", flush=True)


def determinism_indices():
    """12 gate attempts spread over the 96."""
    return list(range(0, p.GATE_ATTEMPTS, p.GATE_ATTEMPTS // DETERMINISM_ATTEMPTS))


def determinism(out, tag):
    _require_pinned_runtime()
    from experiments.gen_g1_data.episode import run_attempt
    from experiments.props_scene.validate import trace_digest
    plan = p.make_plan()
    rows = {}
    for index in determinism_indices():
        attempt = plan["gate_attempts"][index]
        actions, traces = [], []

        def sink(frame, trace):
            actions.append(frame["action"].copy())
            traces.append(trace)

        result = run_attempt(attempt, p.attempt_draw(attempt, 0), sink, render=True)
        rows[str(index)] = {"digest": trace_digest(actions, traces), "success": bool(result["success"])}
    _write_json(Path(out) / f"determinism_{tag}.json", {"lp_num_threads": os.environ.get("LP_NUM_THREADS"), "rows": rows})
    print(f"G1 DETERMINISM {tag} DONE", flush=True)


def gate_verdict(rows):
    """The D4/R3 bars over the gate rows: first-attempt admission overall and for every object, table-height stratum and distractor
    count; the expert's success alone is reported next to it. Pure."""
    rate = lambda subset, key="admitted": sum(r[key] for r in subset) / len(subset)
    out = {"attempts": len(rows), "admitted": sum(r["admitted"] for r in rows), "overall": rate(rows),
           "expert_success_overall": rate(rows, "expert_success")}
    for field in GATE_FIELDS + ("region",):
        values = sorted({r[field] for r in rows})
        out[f"by_{field}"] = {str(v): {"n": sum(r[field] == v for r in rows), "rate": rate([r for r in rows if r[field] == v]),
                                       "expert_success_rate": rate([r for r in rows if r[field] == v], "expert_success")}
                              for v in values}
    out["bars"] = {"overall": GATE_OVERALL, "per_stratum": GATE_PER_STRATUM, "strata": list(GATE_FIELDS)}
    out["failed_bars"] = [f"overall {out['overall']:.3f}"] if out["overall"] < GATE_OVERALL else []
    for field in GATE_FIELDS:
        out["failed_bars"] += [f"{field}={v} {s['rate']:.3f}" for v, s in out[f"by_{field}"].items() if s["rate"] < GATE_PER_STRATUM]
    out["passed"] = not out["failed_bars"]
    out["reasons"] = dict(Counter(r["reason"] for r in rows if not r["admitted"]))
    out["expert_reasons"] = dict(Counter(r["expert_reason"] for r in rows if not r["admitted"]))
    out["max_distractor_displacement_m"] = max((m["displacement_m"] for r in rows for m in r["distractor_motion"]), default=0.0)
    return out


def gate_summary(out):
    out = Path(out)
    rows = [r for path in sorted(out.glob("gate_*.jsonl")) for r in read_journal(path)]
    if sorted(r["index"] for r in rows) != list(range(p.GATE_ATTEMPTS)):
        raise SystemExit(f"gate incomplete: {len(rows)} rows")
    summary = gate_verdict(rows)
    dets = sorted(out.glob("determinism_*.json"))
    det = [json.loads(d.read_text()) for d in dets]
    summary.update(determinism_runs=[d.name for d in dets], determinism_lp_num_threads=[x["lp_num_threads"] for x in det],
                   determinism_equal=len(det) >= 2 and all(d["rows"] == det[0]["rows"] for d in det[1:]))
    _write_json(out / "gate_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("gate", "determinism", "gate_summary", "pilot", "collect", "status", "merge"):
        mode.add_argument(f"--{name.replace('_', '-')}", dest=name, metavar="DIR")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=contract.SHARDS)
    parser.add_argument("--slots", type=int, nargs="+")
    parser.add_argument("--tag")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-merge", action="store_true")
    args = parser.parse_args(argv)
    if args.gate:
        gate_shard(args.gate, args.shard, args.shards)
    elif args.determinism:
        determinism(args.determinism, args.tag)
    elif args.gate_summary:
        gate_summary(args.gate_summary)
    elif args.pilot:
        collect_shard(args.pilot, args.shard, args.shards, slots=args.slots, pilot=True, resume=args.resume)
    elif args.collect:
        collect_shard(args.collect, args.shard, args.shards, resume=args.resume)
    elif args.status:
        status(args.status)
    elif args.merge:
        from experiments.gen_g1_data.merge import merge_and_seal
        merge_and_seal(args.merge, args.out, args.shards, pilot=args.pilot_merge)


if __name__ == "__main__":
    main()
