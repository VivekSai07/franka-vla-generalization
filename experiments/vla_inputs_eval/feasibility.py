"""The scripted P2 expert on every M0 layout, before any policy number is read. Robosuite image, CPU, no rendering.

A layout the expert cannot solve is dropped for **all** arms and counted by cell, never replaced and never blamed on a
policy. Shardable: `--shard k --shards n` writes `feasibility.shard_k.jsonl` under `--out` (one row per layout, never
overwritten); `--merge` combines them into `feasibility.json` and requires every layout of the plan exactly once.
"""
import argparse
import json
import os
import time
from pathlib import Path

from experiments.props_p2_data.plan import digest

MUJOCO_VERSION = "3.4.0"


def load_plan(plan_dir):
    """plan.json, verified against its recorded digest."""
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if digest(plan) != (plan_dir / "plan.sha256").read_text().strip():
        raise ValueError(f"{plan_dir}: plan.json does not match plan.sha256")
    return plan


def shard_ids(ids, shard, shards):
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is not valid")
    return sorted(ids)[shard::shards]


def run_one(layout):
    """One expert attempt on a layout, built and run exactly as the P2 collector did."""
    from experiments.props_p2_data import episode
    draw = {"seed": layout["seed"], "home_pose_jitter_deg": list(layout["home_pose_jitter_deg"])}
    started = time.monotonic()
    outcome = episode.run_attempt(layout, draw, lambda frame, trace: None, render=False)
    return dict(feasible=bool(outcome["success"]), reason=outcome["reason"], frames=outcome["frames"],
                stable_ticks=outcome["stable_ticks"], seconds=round(time.monotonic() - started, 1))


def run_shard(out, shard, shards, plan_dir, only=None, runner=None):
    """Expert attempt on every layout of this shard; returns the rows. Refuses an existing shard file."""
    plan = load_plan(plan_dir)
    runner = runner or run_one
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"feasibility.shard_{shard}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    ids = shard_ids(list(plan["layouts"]) if only is None else only, shard, shards)
    partial, rows = out / f"feasibility.shard_{shard}.partial.jsonl", []
    with partial.open("w") as stream:
        for lid in ids:
            rows.append(dict(layout_id=lid, **runner(plan["layouts"][lid])))
            stream.write(json.dumps(rows[-1]) + "\n")
            stream.flush()
            print("FEAS", lid, rows[-1]["feasible"], rows[-1]["reason"], flush=True)
    os.link(partial, target)
    partial.unlink()
    return rows


def merge(out, plan_dir):
    """feasibility.json from every shard file; the shards must cover the plan's layouts exactly once."""
    plan, out = load_plan(plan_dir), Path(out)
    target = out / "feasibility.json"
    if target.exists():
        raise FileExistsError(target)
    rows = [json.loads(line) for path in sorted(out.glob("feasibility.shard_*.jsonl")) for line in path.read_text().splitlines()]
    ids = [r["layout_id"] for r in rows]
    if sorted(ids) != sorted(plan["layouts"]):
        raise ValueError(f"shards cover {len(set(ids))} of {len(plan['layouts'])} layouts exactly once: {len(ids)} rows")
    dropped = sorted(r["layout_id"] for r in rows if not r["feasible"])
    by_cell = {}
    for lid in dropped:
        key = "-".join(map(str, plan["layouts"][lid]["cell"]))
        by_cell[key] = by_cell.get(key, 0) + 1
    summary = dict(plan_sha256=digest(plan), feasible=sorted(r["layout_id"] for r in rows if r["feasible"]), dropped=dropped,
                   dropped_by_cell=dict(sorted(by_cell.items())), reasons={r["layout_id"]: r["reason"] for r in rows if not r["feasible"]})
    temporary = out / "feasibility.json.tmp"
    temporary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    os.link(temporary, target)
    temporary.unlink()
    print("FEAS_DONE feasible", len(summary["feasible"]), "dropped", len(dropped), by_cell, flush=True)
    return summary


def main(argv=None):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    if args.merge:
        merge(args.out, args.plan)
        return
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f"MuJoCo {mujoco.__version__} is not {MUJOCO_VERSION}; run in the robosuite image")
    run_shard(args.out, args.shard, args.shards, args.plan)
    print("FEAS_SHARD_DONE", args.shard, args.shards, flush=True)


if __name__ == "__main__":
    main()
