"""P1 collector: slots in plan order, a journal line per attempt, frames written only once admitted.

Modes: --pilot (named slots, scratch output, never sealed), --collect, --resume, --status.
Runs in the robosuite image (MuJoCo 3.4.0). The scheduler (`run_schedule`) imports no simulator, so
it is tested in the admission image with fake `execute` / `commit` functions.

The dataset handle is created once and kept open for the whole run; every admitted episode is saved
and finalized immediately (so a kill leaves the completed episodes readable) without reconstructing
the dataset object in between. B3 reconstructed it after every episode and that *reconstruction* —
not the per-episode finalize — triggered an unbounded arrow-cache rebuild each time (FAILURES #41);
finalizing without reconstructing was verified cheap and correct before this file was written
(`outputs/props_p1_data/finalize_probe.py`, `resume_probe.py`). `open_dataset` is used only once, on
`--resume`.
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

from experiments.props_p1_data import contract
from experiments.props_p1_data import plan as p

REJECTION_BUDGET = 150
MAX_ATTEMPTS_PER_SLOT = p.ATTEMPTS_PER_LAYOUT * (p.MAX_SUBSTITUTIONS + 1)
CHECK_EVERY = 25
MUJOCO_VERSION = "3.4.0"
REPO_ID = "props_p1_data"
TRACE_KEYS = ("qpos", "ctrl", "sim_time", "props_pose", "phase_index", "substeps", "success_tick")


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


def write_substitutions(journal_path):
    substitutions = {str(j["slot"]): int(j["r"]) for j in read_journal(journal_path)
                     if j.get("admitted") and int(j.get("r", 0)) > 0}
    _write_json(Path(journal_path).with_name("substitutions.json"), substitutions)
    return substitutions


# --- scheduler -----------------------------------------------------------------------------------

def run_schedule(plan, execute, journal_path, commit, resume=False, slots=None,
                 max_attempts=MAX_ATTEMPTS_PER_SLOT, strict=True, budget=REJECTION_BUDGET,
                 check=None, dataset_count=None, log=print):
    """Admit every slot once, in plan order.

    A rejected attempt retries the same slot: attempt a uses layout r = a // 3 (r > 0 is a
    deterministic substitute start of the same slot) and expert draw k = a % 3. `commit` runs
    before the journal line, so a journal line never claims an episode the dataset lacks.
    """
    journal_path = Path(journal_path)
    existing = read_journal(journal_path) if journal_path.exists() else []
    if existing and not resume:
        raise FileExistsError(f"{journal_path} is not empty; use resume")
    if resume and not existing:
        raise ValueError("Nothing to resume")
    admitted_slots = {j["slot"] for j in existing if j.get("admitted")}
    if resume and dataset_count is not None and dataset_count() != len(admitted_slots):
        raise ValueError(f"Dataset holds {dataset_count()} episodes but the journal admitted {len(admitted_slots)}")
    rejected = sum(1 for j in existing if not j.get("admitted"))
    if rejected > budget:
        raise BudgetExhausted(f"{rejected} rejected attempts already exceed the budget of {budget}")
    attempts_done = Counter(j["slot"] for j in existing)
    if resume:
        write_substitutions(journal_path)
    if check:
        check(len(admitted_slots))
    for slot in (range(len(plan["episodes"])) if slots is None else slots):
        if slot in admitted_slots:
            continue
        episode = plan["episodes"][slot]
        attempt = attempts_done[slot]
        while True:
            if attempt >= max_attempts:
                if strict:
                    raise SlotExhausted(f"Slot {slot} failed {attempt} attempts on {attempt // p.ATTEMPTS_PER_LAYOUT} starts")
                break
            r, k = divmod(attempt, p.ATTEMPTS_PER_LAYOUT)
            start_episode = p.resample_start(episode, r)
            draw = p.attempt_draw(episode, k)
            started = time.monotonic()
            result, frames, traces, verdict = execute(start_episode, draw)
            wall = time.monotonic() - started
            record = {"slot": slot, "attempt": attempt, "r": r, "k": k, "layout_id": start_episode["layout_id"],
                      "seed": draw["seed"], "admitted": bool(verdict["admitted"]), "reason": verdict["reason"],
                      "gates": verdict["gates"], "wall_seconds": round(wall, 1)}
            if verdict["admitted"]:
                commit(start_episode, frames, result, traces)
                _append(journal_path, record)
                admitted_slots.add(slot)
                if r > 0:
                    write_substitutions(journal_path)
            else:
                _append(journal_path, record)
                rejected += 1
            log(f"P1 slot={slot} attempt={attempt} r={r} k={k} admitted={record['admitted']} "
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

def make_execute():
    from experiments.props_p1_data.episode import run_attempt
    from experiments.props_p1_data.validate import admit

    def execute(episode, draw):
        frames, traces = [], []

        def sink(frame, trace):
            frames.append({key: np.copy(v) if isinstance(v, np.ndarray) else v for key, v in frame.items()})
            traces.append({key: np.copy(trace[key]) for key in TRACE_KEYS})

        result = run_attempt(episode, draw, sink)
        return result, frames, traces, admit(episode, result, frames, traces)

    return execute


def create_dataset(root):
    from experiments.paired_demonstrations.media import ENCODING, FEATURES, PairedDataset
    dataset = PairedDataset.create(
        repo_id=REPO_ID, root=Path(root) / "dataset", fps=30, features=copy.deepcopy(FEATURES),
        vcodec=ENCODING["vcodec"], encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)
    # One file per episode, so the dataset is readable after every finalize without a reopen.
    dataset.meta.info["data_files_size_in_mb"] = 0
    dataset.meta.info["video_files_size_in_mb"] = 0
    return dataset


def open_dataset(root):
    """Reopen a finalized dataset for `--resume`; used once per process, never per episode."""
    from experiments.paired_demonstrations.media import ENCODING, PairedDataset
    dataset = PairedDataset(repo_id=REPO_ID, root=Path(root) / "dataset", video_backend="pyav")
    dataset.vcodec = ENCODING["vcodec"]
    dataset._encoder_threads = ENCODING["encoder_threads"]
    dataset.batch_encoding_size = 1
    dataset._streaming_encoder = None
    dataset.meta.info["data_files_size_in_mb"] = 0
    dataset.meta.info["video_files_size_in_mb"] = 0
    return dataset


class Committer:
    """Writes an admitted episode and finalizes it immediately, without reconstructing the dataset
    object (verified safe: `outputs/props_p1_data/finalize_probe.py`)."""

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
                 blue_slot=np.asarray(episode["blue_slot"]), start=np.asarray(episode["start"]))


def make_check(root, provenance):
    root = Path(root)

    def check(admitted):
        free = shutil.disk_usage(root).free
        if free < contract.MIN_FREE_BYTES:
            raise RuntimeError(f"Only {free / 1024**3:.1f} GB free; need {contract.MIN_FREE_BYTES / 1024**3:.0f}")
        if contract.provenance() != provenance:
            raise RuntimeError("provenance changed during collection; aborting before the seal")

    return check


# --- modes -----------------------------------------------------------------------------------------

def _require_pinned_runtime():
    import mujoco
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f"MuJoCo {mujoco.__version__} is not the pinned {MUJOCO_VERSION}; run in the robosuite image")


def _freeze(root, plan, provenance):
    _write_json(root / "plan.json", plan)
    _write_json(root / "provenance.json", provenance)


def _finish(root, plan, provenance):
    from experiments.props_p1_data.validate import seal, validate_dataset
    journal = read_journal(root / "journal.jsonl")
    validation = validate_dataset(root, plan, journal)
    _write_json(root / "validation.json", validation)
    write_substitutions(root / "journal.jsonl")
    sealed = seal(root, plan, provenance)
    print(f"P1 SEALED episodes={validation['episodes']} fingerprint={sealed['fingerprint']}", flush=True)


def collect(out, resume=False):
    _require_pinned_runtime()
    root = Path(out)
    plan = p.make_plan()
    provenance = contract.provenance()
    if resume:
        if not root.is_dir():
            raise SystemExit(f"{root} does not exist")
        if json.loads((root / "plan.json").read_text()) != json.loads(json.dumps(plan)):
            raise SystemExit("plan.json differs from the plan of record; not resuming")
        if json.loads((root / "provenance.json").read_text()) != provenance:
            raise SystemExit("provenance.json differs from the current source; not resuming")
        dataset = open_dataset(root)
    else:
        if root.exists():
            raise SystemExit(f"No overwrite: {root} exists")
        if shutil.disk_usage(root.parent).free < contract.MIN_FREE_BYTES:
            raise SystemExit("Not enough free disk")
        root.mkdir(parents=True)
        _freeze(root, plan, provenance)
        dataset = create_dataset(root)
    committer = Committer(root, dataset)
    try:
        run_schedule(plan, make_execute(), root / "journal.jsonl", committer, resume=resume,
                     check=make_check(root, provenance), dataset_count=committer.count if resume else None)
    finally:
        committer.dataset.finalize()
    _finish(root, plan, provenance)


def pilot(out, slots, attempts):
    _require_pinned_runtime()
    root = Path(out)
    if root.exists():
        raise SystemExit(f"No overwrite: {root} exists")
    root.mkdir(parents=True)
    plan = p.make_plan()
    dataset = create_dataset(root)
    committer = Committer(root, dataset)
    try:
        summary = run_schedule(plan, make_execute(), root / "journal.jsonl", committer, slots=slots,
                               max_attempts=attempts, strict=False)
    finally:
        committer.dataset.finalize()
    _write_json(root / "pilot.json", {"slots": slots, "attempts": attempts, **summary,
                                      "journal": read_journal(root / "journal.jsonl")})
    print(f"P1 PILOT {summary}", flush=True)


def status(out):
    root = Path(out)
    journal = read_journal(root / "journal.jsonl") if (root / "journal.jsonl").exists() else []
    admitted = {j["slot"] for j in journal if j.get("admitted")}
    rejected = [j for j in journal if not j.get("admitted")]
    current = next((s for s in range(p.ADMIT_COUNT) if s not in admitted), None)
    print(json.dumps({"admitted": len(admitted), "rejected": len(rejected), "current_slot": current,
                      "rejections_by_reason": dict(Counter(j["reason"] for j in rejected)),
                      "sealed": (root / "seal.json").exists()}, indent=2))
    for line in journal[-5:]:
        print(json.dumps(line))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("pilot", "collect", "resume", "status"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--slots", type=int, nargs="+")
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args(argv)
    if args.pilot and not args.slots:
        parser.error("--pilot needs --slots")
    if not args.pilot and (args.slots or args.attempts != 1):
        parser.error("--slots and --attempts belong to --pilot")
    if args.pilot:
        pilot(args.out, args.slots, args.attempts)
    elif args.status:
        status(args.out)
    else:
        collect(args.out, resume=args.resume)


if __name__ == "__main__":
    main()
