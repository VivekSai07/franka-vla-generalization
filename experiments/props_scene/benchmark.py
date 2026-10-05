"""The P0 gate: 300 scripted attempts, run as parallel shards, summarised against fixed thresholds.

288 grid attempts (6 objects x 2 target bins x 2 bin arrangements x 12 seeds) plus 12 in the +y
near-base corner. Start pose, yaw, home jitter and the expert's seed all come from the attempt's
identity, so a plan is reproducible and a shard can be resumed. The success definition and the
thresholds are constants in `constants.py` and are not parameters here.

    python -m experiments.props_scene.benchmark --out /out/gate_X --shard 1/6 [--resume]
    python -m experiments.props_scene.benchmark --out /out/gate_X --recheck 12
    python -m experiments.props_scene.benchmark --out /out/gate_X --summary

Planning and summarising import no simulator; `run` imports it when it first executes an attempt.
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.props_scene import constants as c

SLOTS = ("left", "right")
RECHECK_MIN = 12


def _rng(identity):
    return np.random.RandomState(c.seed(identity))


def _attempt(index, stratum, obj, bin_colour, blue_slot, identity):
    pose_rng = _rng(identity + ":pose")
    x_range, y_range = (c.CORNER_X_M, c.CORNER_Y_M) if stratum == "corner" else (c.START_X_M, c.START_Y_M)
    pose = [float(pose_rng.uniform(*x_range)), float(pose_rng.uniform(*y_range)),
            float(pose_rng.uniform(*c.START_YAW_DEG))]
    jitter_rng = _rng(identity + ":home")
    jitter = [float(v) for v in jitter_rng.uniform(-c.HOME_JITTER_DEG_MAX, c.HOME_JITTER_DEG_MAX, 7)]
    other = "right" if blue_slot == "left" else "left"
    return {"index": index, "id": identity, "stratum": stratum, "object": obj, "bin": bin_colour,
            "blue_slot": blue_slot, "target_slot": blue_slot if bin_colour == "blue" else other,
            "pose": pose, "seed": c.seed(identity + ":expert"), "home_pose_jitter_deg": jitter}


def build_plan():
    plan = []
    for obj in c.OBJECTS:
        for bin_colour in c.BINS:
            for blue_slot in SLOTS:
                for k in range(c.GATE_SEEDS_PER_CELL):
                    plan.append(_attempt(len(plan), "grid", obj, bin_colour, blue_slot,
                                         f"grid:{obj}:{bin_colour}:{blue_slot}:{k}"))
    for i, obj in enumerate(c.OBJECTS):
        slot = SLOTS[i % 2]  # alternate the arrangement so the corner covers both target slots six times each
        for j, bin_colour in enumerate(c.BINS):
            plan.append(_attempt(len(plan), "corner", obj, bin_colour, slot, f"corner:{obj}:{j}"))
    if len(plan) != c.GATE_ATTEMPTS:
        raise AssertionError(f"plan has {len(plan)} attempts, expected {c.GATE_ATTEMPTS}")
    return plan


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shard_of(plan, k, n):
    if not 1 <= k <= n:
        raise ValueError(f"shard must be K/N with 1 <= K <= N, got {k}/{n}")
    return [a for a in plan if a["index"] % n == k - 1]


def _identity_fields(attempt):
    return {key: attempt[key] for key in ("id", "index", "stratum", "object", "bin", "blue_slot", "target_slot",
                                          "pose", "seed")}


def execute_attempt(attempt):
    from experiments.props_scene.validate import execute
    started = time.time()
    result, _, _, check, digest, start_error = execute(attempt, render=True)
    return {**_identity_fields(attempt), "success": bool(result["success"]), "reason": result["reason"],
            "phase_reached": int(result["phase_reached"]), "grasp_lost_phase": result["grasp_lost_phase"],
            "stable_ticks": int(result["stable_ticks"]), "frames": int(result["frames"]),
            "final_metrics": result["final_metrics"], "final_placement": result["final_placement"],
            "resting_in_bin": bool(result["final_placement"]["resting_in_bin"]),
            "replay": check, "replay_ok": bool(check["ok"]), "trace_sha256": digest,
            "start_error_m": start_error, "wall_seconds": round(time.time() - started, 1)}


def _failed(attempt, error):
    return {**_identity_fields(attempt), "success": False, "reason": f"exception:{type(error).__name__}",
            "error": str(error), "replay_ok": False, "resting_in_bin": False, "trace_sha256": None}


def _read(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _pin_plan(out, plan):
    path = out / "plan.json"
    digest = plan_digest(plan)
    if path.exists():
        if json.loads(path.read_text()).get("digest") != digest:
            raise ValueError(f"{path} was written for a different plan")
        return
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    temporary.write_text(json.dumps({"digest": digest, "attempts": plan}, indent=1) + "\n")
    temporary.replace(path)


def run(out, k, n, resume=False, executor=execute_attempt, limit=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    _pin_plan(out, plan)
    path = out / f"attempts_shard{k}of{n}.jsonl"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; pass --resume to continue it")
    done = {r["id"] for r in _read(path)}
    todo = [a for a in shard_of(plan, k, n) if a["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    with path.open("a") as stream:
        for attempt in todo:
            try:
                record = executor(attempt)
            except Exception as error:  # an infrastructure fault is a failed attempt, never a skipped one
                record = _failed(attempt, error)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"[{k}/{n}] {attempt['index']:3d} {attempt['id']} -> {record['reason']}", flush=True)
    return len(todo)


def _tally(records):
    n = len(records)
    wins = sum(1 for r in records if r["success"])
    return {"n": n, "successes": wins, "rate": wins / n if n else 0.0}


def _group(records, key, field="success"):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {name: _tally([{"success": bool(r[field])} for r in rows]) for name, rows in sorted(groups.items())}


def summarise(records, plan, recheck=None):
    by_id = {r["id"]: r for r in records}
    expected = {a["id"] for a in plan}
    missing = sorted(expected - set(by_id))
    rows = [by_id[a["id"]] for a in plan if a["id"] in by_id]
    overall = _tally(rows)
    per_object, per_slot = _group(rows, "object"), _group(rows, "target_slot")
    gates = {
        "overall": overall["rate"] >= c.GATE_OVERALL,
        "per_object": all(t["rate"] >= c.GATE_PER_OBJECT for t in per_object.values()) and len(per_object) == len(c.OBJECTS),
        "per_slot": all(t["rate"] >= c.GATE_PER_SLOT for t in per_slot.values()) and len(per_slot) == len(SLOTS),
        "replay": all(r["replay_ok"] for r in rows),
        "determinism": bool(recheck and recheck["count"] >= RECHECK_MIN and recheck["matched"] == recheck["count"]),
    }
    if missing or not recheck or recheck["count"] < RECHECK_MIN:
        verdict = "incomplete"
    else:
        verdict = "go" if all(gates.values()) else "no-go"
    return {
        "verdict": verdict, "gates": gates, "missing": missing, "overall": overall,
        "thresholds": {"overall": c.GATE_OVERALL, "per_object": c.GATE_PER_OBJECT, "per_slot": c.GATE_PER_SLOT},
        "by_object": per_object, "by_target_slot": per_slot, "by_blue_slot": _group(rows, "blue_slot"),
        "by_bin": _group(rows, "bin"), "by_stratum": _group(rows, "stratum"),
        "by_reason": dict(sorted(Counter(r["reason"] for r in rows).items())),
        "by_object_and_target_slot": {f"{o}/{s}": _tally([r for r in rows if r["object"] == o and r["target_slot"] == s])
                                      for o in c.OBJECTS for s in SLOTS},
        "replay_failures": [r["id"] for r in rows if not r["replay_ok"]],
        "recheck": recheck,
        # Not part of the verdict: where the prop physically rests, whatever the written definition says.
        "diagnostic_resting_in_bin": {"overall": _tally([{"success": bool(r["resting_in_bin"])} for r in rows]),
                                      "by_object": _group(rows, "object", "resting_in_bin")},
    }


def load_records(out):
    records = []
    for path in sorted(Path(out).glob("attempts_shard*of*.jsonl")):
        records += _read(path)
    return records


def recheck(out, count, executor=execute_attempt):
    """Re-run `count` attempts spread over the plan in this process; digests must equal the recorded ones."""
    plan = build_plan()
    by_id = {r["id"]: r for r in load_records(out)}
    recorded = [a for a in plan if a["id"] in by_id]
    if len(recorded) < count:
        raise ValueError(f"only {len(recorded)} attempts are recorded, cannot recheck {count}")
    picked = [recorded[i * len(recorded) // count] for i in range(count)]
    details = []
    for attempt in picked:
        again = executor(attempt)
        original = by_id[attempt["id"]]
        details.append({"id": attempt["id"], "original": original["trace_sha256"], "again": again["trace_sha256"],
                        "same": original["trace_sha256"] is not None and original["trace_sha256"] == again["trace_sha256"]})
        print(f"recheck {attempt['id']}: {'same' if details[-1]['same'] else 'DIFFERENT'}", flush=True)
    result = {"count": len(details), "matched": sum(d["same"] for d in details), "details": details}
    (Path(out) / "recheck.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def _print_summary(s):
    def line(name, t):
        return f"  {name:<24} {t['successes']:3d}/{t['n']:<3d} {100 * t['rate']:5.1f}%"
    print(f"verdict: {s['verdict']}   gates: {s['gates']}   missing attempts: {len(s['missing'])}")
    print("overall\n" + line("all", s["overall"]))
    for title, key in (("per object", "by_object"), ("per target slot", "by_target_slot"),
                       ("per bin arrangement (blue slot)", "by_blue_slot"), ("per target bin", "by_bin"),
                       ("per stratum", "by_stratum")):
        print(title)
        for name, t in s[key].items():
            print(line(name, t))
    print("per reason:", s["by_reason"])
    print("replay failures:", s["replay_failures"])
    print("determinism recheck:", None if not s["recheck"] else {k: s["recheck"][k] for k in ("count", "matched")})
    print("DIAGNOSTIC (not in the verdict): resting in the bin")
    print(line("all", s["diagnostic_resting_in_bin"]["overall"]))
    for name, t in s["diagnostic_resting_in_bin"]["by_object"].items():
        print(line(name, t))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="1/1", help="K/N, 1-based")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="run only the first LIMIT attempts of the shard (smoke tests)")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--recheck", type=int, metavar="N")
    args = parser.parse_args()
    if args.summary:
        path = args.out / "recheck.json"
        summary = summarise(load_records(args.out), build_plan(), json.loads(path.read_text()) if path.exists() else None)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        _print_summary(summary)
    elif args.recheck:
        recheck(args.out, args.recheck)
    else:
        k, n = (int(v) for v in args.shard.split("/"))
        run(args.out, k, n, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
