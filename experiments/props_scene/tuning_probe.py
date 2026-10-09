"""Tuning probe: re-run attempts with constants overridden, in parallel, without rendering.

Not part of the package. Overrides are applied by setting attributes on experiments.props_scene.constants
in each worker before any model is built, so nothing in the worktree changes.

    probe.py NAME 'JSON overrides' IDS_SOURCE [WORKERS]
      IDS_SOURCE: 'failed:<gate dir>' | 'all:<gate dir>' | 'near:<gate dir>' (grid attempts with x < 0.30) | 'scan:'
"""
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

NAME, OVERRIDES, SOURCE = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 24


def _apply(overrides):
    from experiments.props_scene import constants as c
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(getattr(c, key), dict):
            getattr(c, key).update({k: tuple(v) if isinstance(v, list) else v for k, v in value.items()})
        elif isinstance(value, list):
            setattr(c, key, tuple(value))
        else:
            setattr(c, key, value)


def _work(job):
    attempt, overrides = job
    _apply(overrides)
    from experiments.props_scene.validate import execute
    started = time.time()
    try:
        result, _, _, check, _, _ = execute(attempt, render=False)
        return {"id": attempt["id"], "object": attempt["object"], "pose": attempt["pose"], "success": result["success"],
                "reason": result["reason"], "resting": result["final_placement"]["resting_in_bin"],
                "replay_ok": check["ok"], "seconds": round(time.time() - started, 1)}
    except Exception as error:  # noqa: BLE001
        return {"id": attempt["id"], "object": attempt["object"], "pose": attempt["pose"], "success": False,
                "reason": f"exception:{type(error).__name__}:{error}", "resting": False, "replay_ok": False}


def _work_chunk(jobs):
    return [_work(job) for job in jobs]


def main():
    from experiments.props_scene import benchmark as b
    plan = b.build_plan()
    kind, _, rest = SOURCE.partition(":")
    if kind in ("failed", "all", "near"):
        records = b.load_records(rest)
        by_id = {r["id"]: r for r in records}
        if kind == "failed":
            ids = [a["id"] for a in plan if a["id"] in by_id and not by_id[a["id"]]["success"]]
        elif kind == "near":
            ids = [a["id"] for a in plan if a["id"] in by_id and a["pose"][0] < 0.30]
        else:
            ids = [a["id"] for a in plan if a["id"] in by_id]
        attempts = [a for a in plan if a["id"] in set(ids)]
    elif kind == "scan":
        attempts = []
        for obj in ("can", "bread", "lemon", "milk", "bottle", "cereal"):
            for xi in range(20, 35):
                for y in (-0.10, 0.0, 0.10):
                    attempts.append({"id": f"scan:{obj}:{xi}:{y}", "object": obj, "bin": "blue", "blue_slot": "left",
                                     "pose": [xi / 100, y, 30.0], "seed": 1, "home_pose_jitter_deg": [0.0] * 7})
    else:
        raise SystemExit("unknown source")
    only = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
    if only:
        attempts = [a for a in attempts if a["object"] in only]
    print(f"{NAME}: {len(attempts)} attempts, overrides {OVERRIDES}", flush=True)
    attempts.sort(key=lambda a: (a["object"], a["blue_slot"], a["index"] if "index" in a else 0))
    jobs = [(a, OVERRIDES) for a in attempts]
    size = max(1, -(-len(jobs) // WORKERS))
    chunks = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    with get_context("spawn").Pool(min(WORKERS, len(chunks)), maxtasksperchild=1) as pool:
        rows = [row for part in pool.map(_work_chunk, chunks, chunksize=1) for row in part]
    wins = sum(r["success"] for r in rows)
    out = Path("/out/probes")
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.json").write_text(json.dumps({"overrides": OVERRIDES, "source": SOURCE, "rows": rows}, indent=1))
    print(f"{NAME}: {wins}/{len(rows)} success")
    if kind == "scan":
        for obj in sorted({r['object'] for r in rows}):
            line = ""
            for xi in range(20, 35):
                sel = [r for r in rows if r['object'] == obj and r['id'].split(':')[2] == str(xi)]
                line += f" {xi}:{sum(r['success'] for r in sel)}/{len(sel)}"
            print(f" {obj:7s}{line}")
    from collections import Counter
    print(" by reason:", dict(Counter(r["reason"] for r in rows)))
    print(" by object:", {o: f"{sum(r['success'] for r in rows if r['object'] == o)}/{sum(1 for r in rows if r['object'] == o)}"
                          for o in sorted({r['object'] for r in rows})})


if __name__ == "__main__":
    main()
