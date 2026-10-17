"""P2 collector: 12 parallel shards of contiguous plan slots, then a verified merge and the seal.

Modes (one per container):
  --gate <dir> --shard k --shards n          expert re-test, unrendered (rendering does not feed physics)
  --determinism <dir> --tag t                12 rendered gate attempts; run twice, compare in --gate-summary
  --gate-summary <dir>                       bars + determinism verdict
  --pilot <shards_dir> --shard k --slots ... named slots into a scratch shard
  --collect <shards_dir> --shard k [--resume]
  --status <shards_dir>
  --merge <shards_dir> --out <sealed_root>   aggregate, verify against the shards, validate, seal

Each shard: its own journal (one line per attempt), its own LeRobot dataset, finalized per episode
without reconstructing the dataset object (P1's verified pattern; FAILURES #41), resumable alone.
The scheduler (`run_schedule`) imports no simulator and is tested in the admission image.
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

from experiments.props_p2_data import contract
from experiments.props_p2_data import plan as p

REJECTION_BUDGET = 150
MAX_ATTEMPTS_PER_SLOT = p.ATTEMPTS_PER_LAYOUT * (p.MAX_SUBSTITUTIONS + 1)
CHECK_EVERY = 5
MUJOCO_VERSION = "3.4.0"
TRACE_KEYS = ("qpos", "ctrl", "sim_time", "props_pose", "phase_index", "substeps", "success_tick")
DETERMINISM_ATTEMPTS = 12
GATE_OVERALL, GATE_PER_OBJECT, GATE_PER_SLOT = 0.95, 0.90, 0.93


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
                 strict=True, budget=REJECTION_BUDGET, check=None, dataset_count=None, log=print, tag="P2"):
    """Admit every slot in `slots` once, in order.

    A rejected attempt retries the same slot: attempt a uses layout r = a // 3 (r > 0 is a deterministic
    substitute layout of the same slot) and expert draw k = a % 3. `commit` runs before the journal line,
    so a journal line never claims an episode the dataset lacks. `budget` is per shard.
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
                    raise SlotExhausted(f"Slot {slot} failed {attempt} attempts on {attempt // p.ATTEMPTS_PER_LAYOUT} layouts")
                break
            r, k = divmod(attempt, p.ATTEMPTS_PER_LAYOUT)
            layout = p.resample_start(episode, r)
            draw = p.attempt_draw(episode, k)
            started = time.monotonic()
            result, frames, traces, verdict = execute(layout, draw)
            wall = time.monotonic() - started
            record = {"slot": slot, "attempt": attempt, "r": r, "k": k, "layout_id": layout["layout_id"],
                      "seed": draw["seed"], "admitted": bool(verdict["admitted"]), "reason": verdict["reason"],
                      "gates": verdict["gates"], "distractor_motion": verdict.get("details", {}).get("distractor_motion"),
                      "distractor_touched": verdict.get("details", {}).get("distractor_touched"),
                      "wall_seconds": round(wall, 1)}
            if verdict["admitted"]:
                started = time.monotonic()
                commit(layout, frames, result, traces)
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
    from experiments.props_p2_data.episode import run_attempt
    from experiments.props_p2_data.validate import admit

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


class Committer:
    """Writes an admitted episode and finalizes it immediately, without reconstructing the dataset object."""

    def __init__(self, root, dataset):
        self.root, self.dataset = Path(root), dataset
        (self.root / "episodes").mkdir(exist_ok=True)

    def count(self):
        return self.dataset.meta.total_episodes

    def __call__(self, episode, frames, result, traces):
        for frame in frames:
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self.dataset.finalize()
        np.savez(self.root / "episodes" / f"episode_{episode['episode_index']:03d}.npz",
                 qpos=np.stack([t["qpos"] for t in traces]), ctrl=np.stack([t["ctrl"] for t in traces]),
                 props_pose=np.stack([t["props_pose"] for t in traces]),
                 initial_state=result["initial_state"], final_qpos=result["final_qpos"],
                 object=np.asarray(episode["object"]), bin=np.asarray(episode["bin"]),
                 blue_slot=np.asarray(episode["blue_slot"]), start=np.asarray(episode["start"]),
                 distractors=np.asarray([d["object"] for d in episode["distractors"]]),
                 distractor_poses=np.asarray([d["pose"] for d in episode["distractors"]], dtype=np.float64).reshape(-1, 3))


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
    repo_id = f"props_p2_shard_{k:02d}"
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
                               dataset_count=committer.count if resume else None, tag=f"P2 shard={k}")
    finally:
        committer.dataset.finalize()
    _write_json(root / "substitutions.json", substitutions_of(read_journal(root / "journal.jsonl")))
    _write_json(root / "done.json", {**summary, "slots": [slots[0], slots[-1]]})
    print(f"P2 SHARD {k} DONE {summary}", flush=True)


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


def gate_shard(out, k, shards):
    """Unrendered expert re-test over this worker's share of the 288 gate attempts."""
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
        _append(path, {"index": attempt["episode_index"], "object": attempt["object"], "bin": attempt["bin"],
                       "blue_slot": attempt["blue_slot"], "distractor_count": attempt["distractor_count"],
                       "distractors": [d["object"] for d in attempt["distractors"]],
                       "admitted": verdict["admitted"], "reason": verdict["reason"],
                       "expert_reason": result["reason"], "gates": verdict["gates"],
                       "distractor_motion": result["distractor_motion"],
                       "distractor_touched": verdict["details"].get("distractor_touched"),
                       "wall_seconds": round(time.monotonic() - started, 1)})
    print(f"P2 GATE SHARD {k} DONE {len(attempts)}", flush=True)


def determinism_indices():
    """12 gate attempts spread over cells, both counts and both slots."""
    return list(range(0, 288, 24))


def determinism(out, tag):
    _require_pinned_runtime()
    from experiments.props_scene.validate import trace_digest
    plan = p.make_plan()
    rows = {}
    for index in determinism_indices():
        attempt = plan["gate_attempts"][index]
        actions, traces = [], []
        from experiments.props_p2_data.episode import run_attempt

        def sink(frame, trace):
            actions.append(frame["action"].copy())
            traces.append(trace)

        result = run_attempt(attempt, p.attempt_draw(attempt, 0), sink, render=True)
        rows[str(index)] = {"digest": trace_digest(actions, traces), "success": bool(result["success"])}
    _write_json(Path(out) / f"determinism_{tag}.json", {"lp_num_threads": os.environ.get("LP_NUM_THREADS"), "rows": rows})
    print(f"P2 DETERMINISM {tag} DONE", flush=True)


def gate_summary(out):
    out = Path(out)
    rows = [r for path in sorted(out.glob("gate_*.jsonl")) for r in read_journal(path)]
    if sorted(r["index"] for r in rows) != list(range(len(p.make_plan()["gate_attempts"]))):
        raise SystemExit(f"gate incomplete: {len(rows)} rows")
    rate = lambda subset: sum(r["admitted"] for r in subset) / len(subset)
    by_object = {o: rate([r for r in rows if r["object"] == o]) for o in p.OBJECTS}
    by_slot = {s: rate([r for r in rows if r["blue_slot"] == s]) for s in p.SLOTS}
    by_count = {str(n): rate([r for r in rows if r["distractor_count"] == n]) for n in p.DISTRACTOR_COUNTS}
    overall = rate(rows)
    dets = sorted(out.glob("determinism_*.json"))
    det = [json.loads(d.read_text()) for d in dets]
    det_equal = len(det) >= 2 and all(d["rows"] == det[0]["rows"] for d in det[1:])
    passed = (overall >= GATE_OVERALL and min(by_object.values()) >= GATE_PER_OBJECT
              and min(by_slot.values()) >= GATE_PER_SLOT and det_equal)
    summary = {"attempts": len(rows), "admitted": sum(r["admitted"] for r in rows), "overall": overall,
               "by_object": by_object, "by_blue_slot": by_slot, "by_distractor_count": by_count,
               "reasons": dict(Counter(r["reason"] for r in rows if not r["admitted"])),
               "expert_reasons": dict(Counter(r["expert_reason"] for r in rows if not r["admitted"])),
               "max_distractor_displacement_m": max((m["displacement_m"] for r in rows for m in r["distractor_motion"]), default=0.0),
               "determinism_runs": [d.name for d in dets], "determinism_equal": det_equal,
               "determinism_lp_num_threads": [x["lp_num_threads"] for x in det], "passed": passed}
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
        from experiments.props_p2_data.merge import merge_and_seal
        merge_and_seal(args.merge, args.out, args.shards, pilot=args.pilot_merge)


if __name__ == "__main__":
    main()
