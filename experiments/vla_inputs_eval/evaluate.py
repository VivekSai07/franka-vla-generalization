"""M0 stages: prepare, worker, collect, select, compare, final. Part of the frozen shared core.

Pure except `worker`, which imports the simulator and the arm's adapter lazily (robosuite image, GPU for a real policy).

    prepare   verify an arm's checkpoints and bind the shared core, the arm and the manifests (prepared.json)
    worker    run one interleaved share of one stage with exactly ONE checkpoint (no in-process checkpoint switch)
    collect   merge a stage's trial folders into result.json (complete only if every planned trial exists)
    select    pick an arm's checkpoint on the selection tier only (most successes; ties recorded, later checkpoint wins)
    compare   an arm against A0 on the comparison tier only: exact McNemar, stratified paired bootstrap, Holm family
    final     the family-wise table over the arms that have a comparison

Workers are independent processes writing disjoint trial folders, so a stage can be sharded and resumed: a worker keeps every
complete trial folder, moves a half-written one under `quarantine/`, and runs the rest. A trial is never re-run once complete.
A policy's nonfinite action is an outcome (failure), not a missing trial; infrastructure errors raise and the stage is resumed.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

from experiments.vla_inputs_eval import adapters, contract, feasibility, plan as plan_module, protocol, stats
from experiments.xvla_props_p2_eval import analysis

ROOT = contract.ROOT
HOT_C, COOL_C = 83, 75
ARM_NAME = re.compile(r"^A[0-9]$")
CONTROL = "A0"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_new_json(path, value):
    """Like `write_json`, but never replaces an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, value)


def read_json(path):
    return json.loads(Path(path).read_text())


# --- manifests -----------------------------------------------------------------------------------------------------------

def load_manifests(root):
    """(plan, feasibility) of a campaign root, each checked against its recorded digest."""
    root = Path(root)
    plan = feasibility.load_plan(root / "plan")
    feas = read_json(root / "feasibility/feasibility.json")
    if feas["plan_sha256"] != contract.digest(plan):
        raise ValueError("feasibility.json was computed for a different plan")
    return plan, feas


# --- prepare ---------------------------------------------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"^([a-z0-9_.]+):([A-Za-z_][A-Za-z0-9_]*)$")


def runtime():
    versions = {}
    for name in ("torch", "lerobot", "transformers", "tokenizers", "mujoco", "robosuite", "numpy", "safetensors", "av"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), versions=versions, image=os.environ.get("M0_IMAGE"))


def checkpoint_facts(checkpoint):
    """Full hash of the weights against their marker, plus a hash of every other file of the checkpoint directory."""
    checkpoint = Path(checkpoint)
    marker = read_json(checkpoint / "checkpoint_result.json")
    model = checkpoint / "model.safetensors"
    model_sha256 = contract.file_hash(model)
    if marker.get("status") != "complete" or model_sha256 != marker["model_sha256"]:
        raise ValueError(f"{checkpoint}: model.safetensors does not match its checkpoint_result.json")
    stat = model.stat()
    others = {p.name: contract.file_hash(p) for p in sorted(checkpoint.iterdir()) if p.is_file() and p.name != "model.safetensors"}
    return dict(path=str(checkpoint), model_sha256=model_sha256, model_bytes=stat.st_size, model_mtime_ns=stat.st_mtime_ns,
                files=others)


def arm_files(identifier, root=ROOT):
    """The arm's source closure: its package (non-test modules and README), the extra repository files it declares
    (`EXTRA_SOURCE`), and every other repository module its import loads that the core does not already bind."""
    match = _IDENTIFIER.match(identifier)
    if not match or not match.group(1).startswith(adapters.ARM_PREFIX):
        raise ValueError(f"adapter identifier {identifier!r} is not an arm module")
    module = importlib.import_module(match.group(1))
    root = Path(root)
    package = Path(module.__file__).resolve().parent
    files = {p.relative_to(root).as_posix() for p in package.glob("*.py")} | {(package / "README.md").relative_to(root).as_posix()}
    files |= set(getattr(module, "EXTRA_SOURCE", ()))
    core = set(contract.core_provenance(root))
    for loaded in list(sys.modules.values()):
        path = getattr(loaded, "__file__", None)
        if path and Path(path).is_absolute() and Path(path).resolve().is_relative_to(root.resolve()):
            rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
            if "/tests/" not in rel and not rel.endswith("__init__.py") and rel not in core:
                files.add(rel)
    return {rel: contract.file_hash(root / rel) for rel in sorted(files)}


def prepare(root, arm, identifier, checkpoints, repo=ROOT, phase_checkpoint=None):
    """Write `arms/<arm>/prepared.json`. `checkpoints`: {update: path}. For arms other than A0, A0 must be prepared with an
    identical shared core, and `phase_checkpoint` (the 4.0-epoch checkpoint, evaluated on the comparison tier whatever the
    selection says) must be one of the checkpoints."""
    root = Path(root)
    if not ARM_NAME.match(arm):
        raise ValueError(f"arm {arm!r} is not A0..A9")
    if arm != CONTROL and phase_checkpoint not in checkpoints:
        raise ValueError(f"arm {arm}: declare the matched-phase (4.0-epoch) checkpoint, one of {sorted(checkpoints)}")
    if arm == CONTROL and phase_checkpoint is not None:
        raise ValueError("A0 is a single fixed checkpoint; it has no matched-phase secondary")
    plan, feas = load_manifests(root)
    sidecar_fingerprint = contract.check_signal_source(repo)
    contract.s0.check_p2_frozen(repo)
    core = contract.core_provenance(repo)
    record = dict(
        arm=dict(name=arm, adapter=identifier, files=arm_files(identifier, repo), runtime=runtime()),
        core=dict(provenance=core, sha256=contract.digest(core)),
        plan_sha256=contract.file_hash(root / "plan/plan.json"), feasibility_sha256=contract.file_hash(root / "feasibility/feasibility.json"),
        dropped=feas["dropped"], dropped_by_cell=feas["dropped_by_cell"], s0_seal=sidecar_fingerprint,
        phase_checkpoint=phase_checkpoint,
        checkpoints={str(k): checkpoint_facts(v) for k, v in sorted(checkpoints.items())},
        prepared_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    record["arm"]["sha256"] = contract.digest([record["arm"]["adapter"], record["arm"]["files"]])
    if arm != CONTROL:
        control = read_json(root / "arms" / CONTROL / "prepared.json")
        if control["core"]["sha256"] != record["core"]["sha256"]:
            raise ValueError(f"arm {arm}: shared core differs from A0's; a core change is a new protocol with a new control")
    (root / "arms" / arm).mkdir(parents=True, exist_ok=True)
    write_new_json(root / "arms" / arm / "prepared.json", record)
    print("PREPARED", arm, json.dumps(dict(core=record["core"]["sha256"][:12], arm=record["arm"]["sha256"][:12],
                                           checkpoints=sorted(record["checkpoints"]), dropped=len(feas["dropped"]))), flush=True)
    return record


def load_prepared(root, arm):
    path = Path(root) / "arms" / arm / "prepared.json"
    if not path.is_file():
        raise RuntimeError(f"Run prepare first: {path} is missing")
    return read_json(path)


def verify_core(prepared, repo=ROOT):
    """Raises unless the shared core is byte for byte what it was when the arm was prepared."""
    if contract.core_sha256(repo) != prepared["core"]["sha256"]:
        changed = sorted(r for r, h in contract.core_provenance(repo).items() if prepared["core"]["provenance"].get(r) != h)
        raise ValueError(f"the shared core changed since prepare: {changed[:5]}")


def verify_arm(prepared, repo=ROOT):
    """Raises unless the arm's adapter closure is what it was when the arm was prepared."""
    changed = sorted(r for r, h in prepared["arm"]["files"].items() if contract.file_hash(Path(repo) / r) != h)
    if changed:
        raise ValueError(f"arm {prepared['arm']['name']} changed since prepare: {changed[:5]}")


def verify_manifests(root, prepared):
    root = Path(root)
    if (contract.file_hash(root / "plan/plan.json") != prepared["plan_sha256"]
            or contract.file_hash(root / "feasibility/feasibility.json") != prepared["feasibility_sha256"]):
        raise ValueError("the layout or feasibility manifest changed since prepare")


def verify_checkpoint(prepared, checkpoint):
    """Cheap per-worker re-check: marker hash, size and mtime of the weights."""
    facts = prepared["checkpoints"][str(checkpoint)]
    directory = Path(facts["path"])
    marker = read_json(directory / "checkpoint_result.json")
    stat = (directory / "model.safetensors").stat()
    if (marker["model_sha256"] != facts["model_sha256"] or stat.st_size != facts["model_bytes"]
            or stat.st_mtime_ns != facts["model_mtime_ns"]):
        raise ValueError(f"{directory} changed since prepare; run prepare again")
    return directory


# --- worker ----------------------------------------------------------------------------------------------------------------

def gpu_state():
    """Temperature (C), utilisation (%) and memory (MiB) of GPU 0, or None where nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=20)
        t, u, m = (float(x) for x in out.stdout.strip().split(","))
        return dict(temperature_c=t, utilization_pct=u, memory_mib=m)
    except Exception:  # telemetry is diagnostic; its absence must not fail a trial
        return None


def cool_down(sleep=time.sleep):
    """Pause while the GPU is at or above HOT_C until it falls to COOL_C. Returns seconds paused."""
    paused, state = 0.0, gpu_state()
    if state is None or state["temperature_c"] < HOT_C:
        return paused
    while state is not None and state["temperature_c"] > COOL_C:
        sleep(15)
        paused += 15
        state = gpu_state()
    return paused


def memory_telemetry():
    """This process's peak host and (if torch is loaded) GPU memory, MiB."""
    out = dict(rss_max_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        out.update(torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   torch_max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    return out


def quarantine(directory, path):
    target = Path(directory) / "quarantine"
    target.mkdir(exist_ok=True)
    n = 0
    while (target / f"{path.name}.{n}").exists():
        n += 1
    path.rename(target / f"{path.name}.{n}")


def default_adapter_factory(prepared, checkpoint):
    facts = prepared["checkpoints"][str(checkpoint)]
    return adapters.load_adapter(prepared["arm"]["adapter"], Path(facts["path"]))


def default_runner(adapter, state_builder, layout, trial, folder):
    import mujoco
    from experiments.vla_inputs_eval import rollout, scenes
    model, data, names = scenes.restore(layout)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        return rollout.run(adapter, state_builder, model, data, names, renderer, trial, folder)


def worker(root, arm, tier, checkpoint, shard, shards, repo=ROOT, factory=default_adapter_factory, runner=default_runner,
           cooler=cool_down):
    """One interleaved share of one stage, one checkpoint, one adapter built once. Returns the rows it ran."""
    if type(checkpoint) is not int:
        raise ValueError("a worker serves exactly one checkpoint")
    root = Path(root)
    prepared = load_prepared(root, arm)
    if str(checkpoint) not in prepared["checkpoints"]:
        raise ValueError(f"checkpoint {checkpoint} was not prepared for arm {arm}")
    verify_core(prepared, repo)
    verify_arm(prepared, repo)
    verify_manifests(root, prepared)
    verify_checkpoint(prepared, checkpoint)
    plan, feas = load_manifests(root)
    trials = protocol.shard_trials(protocol.stage_trials(plan, feas, tier), shard, shards)
    directory = protocol.stage_dir(root, arm, tier, checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    pending = [t for t in trials if not protocol.trial_is_complete(protocol.trial_folder(directory, t))]
    rows = []
    if not pending:
        print("WORKER_DONE", arm, tier, checkpoint, shard, shards, "nothing to run", flush=True)
        return rows
    adapter = factory(prepared, checkpoint)
    adapters.check_adapter(adapter)
    state_builder = getattr(adapter, "state_builder", adapters.state_p2)
    try:
        for trial in pending:
            folder = protocol.trial_folder(directory, trial)
            if folder.exists():
                quarantine(directory, folder)
            folder.mkdir()
            paused, before = cooler(), gpu_state()
            row = runner(adapter, state_builder, plan["layouts"][trial["layout_id"]], trial, folder)
            write_json(folder / "telemetry.json", dict(before=before, after=gpu_state(), cooled_seconds=paused, **memory_telemetry()))
            rows.append(row)
            print("EVAL", directory.name, json.dumps({k: row[k] for k in ("ordinal", "layout_id", "kind", "task_success", "cause")}),
                  f"{row['wall_seconds']:.0f}s", flush=True)
    except BaseException as exc:
        write_json(directory / f"failure_shard{shard}.json", dict(arm=arm, tier=tier, checkpoint=checkpoint, shard=shard, shards=shards,
                                                                  error_type=type(exc).__name__, error=str(exc)))
        raise
    print("WORKER_DONE", arm, tier, checkpoint, shard, shards, flush=True)
    return rows


# --- collect ---------------------------------------------------------------------------------------------------------------

def collect(root, arm, tier, checkpoint):
    root = Path(root)
    prepared = load_prepared(root, arm)
    verify_manifests(root, prepared)
    plan, feas = load_manifests(root)
    directory = protocol.stage_dir(root, arm, tier, checkpoint)
    trials = protocol.stage_trials(plan, feas, tier)
    rows, missing = [], []
    for trial in trials:
        folder = protocol.trial_folder(directory, trial)
        if not folder.is_dir() or not protocol.trial_is_complete(folder):
            missing.append(trial["ordinal"])
            continue
        row = read_json(folder / "episode.json")
        planned = json.loads(json.dumps(trial))
        if any(row.get(k) != v for k, v in planned.items()):
            raise ValueError(f"Trial {trial['ordinal']} does not match the planned trial")
        rows.append(row)
    report = dict(status="complete" if not missing else "incomplete", arm=arm, tier=tier, checkpoint=checkpoint,
                  model_sha256=prepared["checkpoints"][str(checkpoint)]["model_sha256"], core_sha256=prepared["core"]["sha256"],
                  arm_sha256=prepared["arm"]["sha256"], dropped_layouts=feas["dropped"], dropped_by_cell=feas["dropped_by_cell"],
                  planned_trials=len(trials),
                  missing_ordinals=missing, failure_files=sorted(p.name for p in directory.glob("failure_shard*.json")),
                  episodes=rows, readouts=analysis.readouts(rows, []) if rows else None)
    write_json(directory / "result.json", report)
    print("COLLECT", directory.name, report["status"], len(rows), "of", len(trials), "missing", missing, flush=True)
    return report


def load_stage(root, arm, tier, checkpoint):
    path = protocol.stage_dir(root, arm, tier, checkpoint) / "result.json"
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; run collect")
    report = read_json(path)
    if report["status"] != "complete":
        raise RuntimeError(f"{path.parent.name} is not complete (missing {report['missing_ordinals']})")
    return report


# --- select ----------------------------------------------------------------------------------------------------------------

def select(root, arm):
    """Choose a checkpoint on the selection tier only. A tie goes to the later checkpoint and is recorded."""
    root = Path(root)
    prepared = load_prepared(root, arm)
    counts = {}
    for key in sorted(prepared["checkpoints"], key=int):
        episodes = load_stage(root, arm, "selection", int(key))["episodes"]
        if any(e["kind"] != "trained" or e["tier"] != "selection" for e in episodes):
            raise ValueError("the selection tier holds only trained-pair selection layouts; held-out layouts never select")
        counts[int(key)] = sum(1 for e in episodes if e["task_success"])
    winner, phase = stats.select(counts), prepared["phase_checkpoint"]
    decision = dict(arm=arm, counts={str(k): v for k, v in counts.items()}, tied=stats.tied(counts), winner=winner,
                    rule="most successes on the selection tier; a tie goes to the later checkpoint (administrative, not evidence)",
                    phase_checkpoint=phase, comparison_checkpoints=sorted({winner, phase}),
                    note="the comparison tier is run at both checkpoints (one when they coincide); C never chooses between them")
    write_new_json(root / "arms" / arm / "selection.json", decision)
    print("SELECTED", json.dumps(decision), flush=True)
    return decision


# --- compare / final ---------------------------------------------------------------------------------------------------------

def compare(root, arm, checkpoint, repo=ROOT, seed=0):
    """`arm` at `checkpoint` against A0 on the comparison tier. Refuses unless the shared cores agree and match the repo."""
    root = Path(root)
    if arm == CONTROL:
        raise ValueError("A0 is the control; compare an arm A1..A4")
    control, mine = load_prepared(root, CONTROL), load_prepared(root, arm)
    decision = read_json(root / "arms" / arm / "selection.json")
    if checkpoint not in decision["comparison_checkpoints"]:
        raise ValueError(f"checkpoint {checkpoint} is neither the selection winner nor the matched-phase checkpoint "
                         f"{decision['comparison_checkpoints']}; the comparison tier never chooses a checkpoint")
    role = ("primary and matched-phase" if decision["winner"] == decision["phase_checkpoint"] == checkpoint
            else "primary" if checkpoint == decision["winner"] else "matched-phase secondary")
    for name, prepared in ((CONTROL, control), (arm, mine)):
        verify_core(prepared, repo)
        verify_manifests(root, prepared)
    verify_arm(mine, repo)
    verify_arm(control, repo)
    if control["core"]["sha256"] != mine["core"]["sha256"]:
        raise ValueError("the arm's shared core differs from A0's")
    control_checkpoint = int(next(iter(control["checkpoints"])))
    new = load_stage(root, arm, "comparison", checkpoint)
    old = load_stage(root, CONTROL, "comparison", control_checkpoint)
    new_by, old_by = {e["layout_id"]: e for e in new["episodes"]}, {e["layout_id"]: e for e in old["episodes"]}
    if set(new_by) != set(old_by) or len(new_by) != new["planned_trials"]:
        raise ValueError("incomplete pairs: both arms need a trial on every feasible comparison layout")
    ids = sorted(new_by)
    plan, _ = load_manifests(root)
    dropped = sorted(i for i in new["dropped_layouts"] if plan["layouts"][i]["tier"] == "comparison")
    dropped_by_cell = {}
    for lid in dropped:
        key = "-".join(map(str, plan["layouts"][lid]["cell"]))
        dropped_by_cell[key] = dropped_by_cell.get(key, 0) + 1
    rows = [dict(layout_id=i, object=new_by[i]["object"], new=new_by[i]["task_success"], control=old_by[i]["task_success"]) for i in ids]
    b = sum(1 for r in rows if r["new"] and not r["control"])
    c = sum(1 for r in rows if r["control"] and not r["new"])
    p = stats.mcnemar(b, c)
    result = dict(
        arm=arm, checkpoint=checkpoint, role=role, control_checkpoint=control_checkpoint, layouts=len(ids), new_successes=sum(r["new"] for r in rows),
        control_successes=sum(r["control"] for r in rows), discordant=dict(new_only=b, control_only=c), mcnemar_exact_p=p,
        effect=stats.paired_effect(rows, seed=seed), core_sha256=mine["core"]["sha256"], arm_sha256=mine["arm"]["sha256"],
        control_arm_sha256=control["arm"]["sha256"], dropped_layouts=dropped,
        dropped_by_cell=dict(sorted(dropped_by_cell.items())),
        population=f"{len(ids)} feasible comparison layouts of {len(plan['tiers']['comparison'])} planned; {len(dropped)} dropped for every arm "
                   f"(expert could not solve them), counted by cell and never replaced",
        discordant_layouts=dict(new_only=[r["layout_id"] for r in rows if r["new"] and not r["control"]],
                                control_only=[r["layout_id"] for r in rows if r["control"] and not r["new"]]),
        readouts=dict(new=analysis.readouts(new["episodes"], []), control=analysis.readouts(old["episodes"], [])),
        caveat="conditional on these checkpoints, recipes and the one training seed; layouts are the independent units")
    write_new_json(root / "arms" / arm / f"compare_step_{checkpoint:06d}.json", result)
    print("COMPARE", arm, checkpoint, f"{result['new_successes']}/{len(ids)} vs {result['control_successes']}/{len(ids)}",
          f"b={b} c={c} p={p:.4f}", flush=True)
    return result


def final(root):
    """Raw and Holm-adjusted p over the fixed family A1..A4; arms without a comparison enter as p = 1."""
    root = Path(root)
    raw, files = {}, {}
    for arm in stats.FAMILY:
        selection = root / "arms" / arm / "selection.json"
        if not selection.is_file():
            continue
        found = root / "arms" / arm / f"compare_step_{read_json(selection)['winner']:06d}.json"
        if found.is_file():
            files[arm] = found.name
            raw[arm] = read_json(found)["mcnemar_exact_p"]
    out = dict(raw=raw, holm=stats.family_holm(raw), files=files, family=list(stats.FAMILY),
               note="primary = the selection-tier winner of each arm; the matched-phase secondary is reported separately and is "
                    "not in the family; original power is unadjusted; unrun arms enter the family as p = 1")
    write_json(root / "final.json", out)
    print("FINAL", json.dumps(out), flush=True)
    return out


# --- command line --------------------------------------------------------------------------------------------------------------

def _checkpoints(text):
    parsed = {}
    for item in text.split(","):
        key, _, path = item.partition("=")
        parsed[int(key)] = Path(path)
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="campaign output directory (holds plan/, feasibility/, arms/)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("--arm", required=True)
    p.add_argument("--adapter", required=True, help="module:Class under experiments.vla_inputs_arm_*")
    p.add_argument("--checkpoints", required=True, type=_checkpoints, help="update=path[,update=path]")
    p.add_argument("--phase-checkpoint", type=int, help="the 4.0-epoch checkpoint (arms other than A0)")
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True)
    w.add_argument("--tier", required=True, choices=protocol.TIERS)
    w.add_argument("--checkpoint", required=True, type=int, help="exactly one checkpoint per worker")
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--shards", type=int, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--arm", required=True)
    c.add_argument("--tier", required=True, choices=protocol.TIERS)
    c.add_argument("--checkpoint", required=True, type=int)
    s = sub.add_parser("select")
    s.add_argument("--arm", required=True)
    k = sub.add_parser("compare")
    k.add_argument("--arm", required=True)
    k.add_argument("--checkpoint", required=True, type=int)
    sub.add_parser("final")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.command == "plan":
        print("PLAN", plan_module.write(Path(args.root) / "plan"), flush=True)
    elif args.command == "prepare":
        prepare(args.root, args.arm, args.adapter, args.checkpoints, phase_checkpoint=args.phase_checkpoint)
    elif args.command == "worker":
        import torch
        torch.use_deterministic_algorithms(True)
        worker(args.root, args.arm, args.tier, args.checkpoint, args.shard, args.shards)
    elif args.command == "collect":
        return 0 if collect(args.root, args.arm, args.tier, args.checkpoint)["status"] == "complete" else 3
    elif args.command == "select":
        select(args.root, args.arm)
    elif args.command == "compare":
        compare(args.root, args.arm, args.checkpoint)
    else:
        final(args.root)


if __name__ == "__main__":
    sys.exit(main())
