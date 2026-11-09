"""The scripted expert on every G0 layout and condition, before any policy number is read. Robosuite image, CPU, no rendering.

Forked from `experiments/vla_inputs_eval/feasibility.py` (frozen, unchanged). Rules (G0 spec D8):
  - a layout the expert cannot solve under NOMINAL conditions is dropped with all six of its trials (the paired comparison needs it);
  - `lighting`, `background` and `drop` change nothing the expert could trip on, so they share the nominal trial's feasibility;
  - `distractors` and `spatial` change geometry: the expert must solve the layout under that condition;
  - `move` is feasible if the nominal is and the expert solves a fresh episode from the post-move pose.
A trial dropped for an arm is dropped for every arm, counted by axis and level, and never replaced.
Shardable: `--shard k --shards n` writes `feasibility.shard_k.jsonl` under `--out` (one row per job, never overwritten); `--merge`
combines them into `feasibility.json` and requires every job of the plan exactly once.
"""
import argparse
import json
import os
import time
from pathlib import Path

from experiments.gen_eval import protocol
from experiments.gen_scene import levels
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.props_p2_data.plan import digest

MUJOCO_VERSION = "3.4.0"
JOBS = ("nominal", "distractors", "spatial", "move")
INHERIT = ("lighting", "background")     # share the nominal trial's feasibility; `drop` does too (below)


def load_plan(plan_dir):
    """plan.json, verified against its recorded digest."""
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if digest(plan) != (plan_dir / "plan.sha256").read_text().strip():
        raise ValueError(f"{plan_dir}: plan.json does not match plan.sha256")
    return plan


def jobs_of(layout):
    """The expert runs a layout needs: nominal, its distractor and spatial conditions, and the fresh episode of a `move`."""
    jobs = ["nominal", "distractors", "spatial"]
    if layout["axes"]["disturbance"]["level"] == "move":
        jobs.append("move")
    return jobs


def job_scene(layout, job):
    """(layout, condition) the expert runs for a job."""
    if job == "nominal":
        return layout, NOMINAL
    if job in ("distractors", "spatial"):
        return layout, Condition.from_dict(layout["axes"][job]["condition"])
    if job == "move":
        x, y = layout["axes"]["disturbance"]["condition"]["move_to"]
        return dict(layout, start=[x, y, layout["start"][2]]), NOMINAL
    raise ValueError(f"unknown job {job!r}")


def all_jobs(plan):
    """Every (layout id, job) of the plan, sorted: the unit of sharding."""
    return sorted((lid, job) for lid, layout in plan["layouts"].items() for job in jobs_of(layout))


def shard_jobs(jobs, shard, shards):
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is not valid")
    return sorted(jobs)[shard::shards]


def run_one(layout, job):
    """One expert attempt, built and run as the engine's expert would on the trial's scene."""
    from experiments.gen_scene import compose
    from experiments.gen_scene.expert import GenExpert
    layout, condition = job_scene(layout, job)
    started = time.monotonic()
    model, resolved = compose.model_for(layout, condition)
    expert = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    episode = {"object": layout["object"], "bin": layout["bin"], "seed": layout["seed"],
               "home_pose_jitter_deg": list(layout["home_pose_jitter_deg"])}
    outcome = expert.run(scene["initial_state"], episode, lambda frame, trace: None, render=False)
    return dict(feasible=bool(outcome["success"]), reason=outcome["reason"], frames=outcome["frames"],
                stable_ticks=outcome["stable_ticks"], seconds=round(time.monotonic() - started, 1))


def run_shard(out, shard, shards, plan_dir, only=None, runner=None):
    """Expert attempt on every job of this shard; returns the rows. Refuses an existing shard file."""
    plan = load_plan(plan_dir)
    runner = runner or run_one
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"feasibility.shard_{shard}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    jobs = shard_jobs(all_jobs(plan) if only is None else only, shard, shards)
    partial, rows = out / f"feasibility.shard_{shard}.partial.jsonl", []
    with partial.open("w") as stream:
        for lid, job in jobs:
            rows.append(dict(layout_id=lid, job=job, **runner(plan["layouts"][lid], job)))
            stream.write(json.dumps(rows[-1]) + "\n")
            stream.flush()
            print("FEAS", lid, job, rows[-1]["feasible"], rows[-1]["reason"], flush=True)
    os.link(partial, target)
    partial.unlink()
    return rows


def decide(layout, ok):
    """{trial key: None if feasible else the reason} for the six trials of a layout; `ok` maps job -> bool."""
    lid, out = layout["layout_id"], {}
    if not ok["nominal"]:
        return {protocol.trial_key(lid, axis): "nominal infeasible" for axis in protocol.AXES}
    for axis in protocol.AXES:
        key = protocol.trial_key(lid, axis)
        if axis in ("nominal", *INHERIT):
            out[key] = None
        elif axis in ("distractors", "spatial"):
            out[key] = None if ok[axis] else f"{axis} infeasible"
        else:   # disturbance
            level = layout["axes"]["disturbance"]["level"]
            out[key] = None if level == "drop" or ok["move"] else "post-move pose infeasible"
    return out


def merge(out, plan_dir):
    """feasibility.json from every shard file; the shards must cover the plan's jobs exactly once."""
    plan, out = load_plan(plan_dir), Path(out)
    target = out / "feasibility.json"
    if target.exists():
        raise FileExistsError(target)
    rows = [json.loads(line) for path in sorted(out.glob("feasibility.shard_*.jsonl")) for line in path.read_text().splitlines()]
    got = sorted((r["layout_id"], r["job"]) for r in rows)
    if got != all_jobs(plan):
        raise ValueError(f"shards cover {len(set(got))} of {len(all_jobs(plan))} jobs exactly once: {len(got)} rows")
    ok = {}
    for r in rows:
        ok.setdefault(r["layout_id"], {})[r["job"]] = bool(r["feasible"])
    feasible, dropped, counts = [], {}, {}
    for tier in protocol.TIERS:
        for lid in plan["tiers"][tier]:
            layout = plan["layouts"][lid]
            decided = decide(layout, ok[lid])
            for axis in protocol.AXES:
                key = protocol.trial_key(lid, axis)
                level = protocol.condition_of(layout, axis)[0]
                cell = counts.setdefault(tier, {}).setdefault(axis, {}).setdefault(level, {"planned": 0, "feasible": 0})
                cell["planned"] += 1
                if decided[key] is None:
                    feasible.append(key)
                    cell["feasible"] += 1
                else:
                    dropped[key] = decided[key]
    reasons = {r["layout_id"] + ":" + r["job"]: r["reason"] for r in rows if not r["feasible"]}
    summary = dict(plan_sha256=digest(plan), feasible_trials=sorted(feasible), dropped_trials=dict(sorted(dropped.items())),
                   counts=counts, expert_reasons=dict(sorted(reasons.items())))
    temporary = out / "feasibility.json.tmp"
    temporary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    os.link(temporary, target)
    temporary.unlink()
    print("FEAS_DONE feasible", len(feasible), "dropped", len(dropped), flush=True)
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
