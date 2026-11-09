"""G0 stages: prepare, worker, collect, select, compare, final. Part of the frozen shared core.

Pure except `worker`, which imports the simulator and the arm's adapter lazily (robosuite image, GPU for a real policy).

    plan      the layouts and their conditions (admission image)
    prepare   verify an arm's checkpoints and bind the shared core, the textures, the arm and the manifests (prepared.json)
    worker    run one share of one stage with exactly ONE checkpoint (no in-process checkpoint switch)
    collect   merge a stage's trial folders into result.json (complete only if every planned trial exists)
    select    pick a later arm's checkpoint on the selection tier only (most successes; ties recorded, later checkpoint wins)
    compare   one arm's benchmark stage: every axis and level paired against nominal on the same layouts
    final     the per-arm table over the compared arms

Forked from `experiments/vla_inputs_eval/evaluate.py` (frozen, unchanged). Differences: a trial carries a condition; there is no
control arm (the control is the same layouts under nominal conditions, inside every arm); `prepare` also binds the texture
files and requires every prepared arm to share one core; `compare` is per arm. Workers are independent processes writing disjoint
trial folders, so a stage can be sharded and resumed: a worker keeps every complete trial folder, moves a half-written one under
`quarantine/`, and runs the rest. A trial is never re-run once complete. A policy's nonfinite action is an outcome (failure), not
a missing trial; infrastructure errors raise and the stage is resumed. Nothing is keyed by a process id (FAILURES #71): every
container worker is pid 1; workers are named by their shard index.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import secrets
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from experiments.gen_eval import contract, feasibility, plan as plan_module, protocol, stats
from experiments.gen_scene import levels
from experiments.vla_inputs_eval import adapters
from experiments.xvla_props_p2_eval import analysis

ROOT = contract.ROOT
HOT_C, COOL_C = 83, 75
LARGE_BYTES = 64 << 20  # input files this large get a size+mtime check in workers and a full hash in `compare`
ARM_NAME = re.compile(r"^A[0-9]$")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp" + secrets.token_hex(4))
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


def _arm_module(identifier):
    match = _IDENTIFIER.match(identifier)
    if not match or not match.group(1).startswith(adapters.ARM_PREFIX):
        raise ValueError(f"adapter identifier {identifier!r} is not an arm module")
    return importlib.import_module(match.group(1))


def _input_files(directory):
    """{relative path: Path} of every file under `directory`; symlinks are refused."""
    files = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in an arm input directory: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = path
    return files


def arm_inputs(identifier, repo=ROOT):
    """The arm's declared input directories (`INPUT_DIRS`, relative to the repository root): every file with its sha256,
    size and mtime. The X-VLA base bundle (tokenizer files) is one."""
    out = {}
    for rel in getattr(_arm_module(identifier), "INPUT_DIRS", ()):
        files = _input_files(Path(repo) / rel)
        if not files:
            raise ValueError(f"arm input directory {rel} is missing or empty")
        out[rel] = {name: dict(sha256=contract.file_hash(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                    for name, p in files.items()}
    return out


def arm_files(identifier, root=ROOT):
    """The arm's source closure: its package (non-test modules and README), the extra repository files it declares
    (`EXTRA_SOURCE`), and every other repository module its import loads that the core does not already bind."""
    module = _arm_module(identifier)
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


def prepared_arms(root):
    """{arm: prepared record} of every arm prepared in this campaign."""
    return {p.parent.name: read_json(p) for p in sorted((Path(root) / "arms").glob("*/prepared.json"))}


def prepare(root, arm, identifier, checkpoints, repo=ROOT):
    """Write `arms/<arm>/prepared.json`. `checkpoints`: {update: path}. Every arm of a campaign shares one core."""
    root = Path(root)
    if not ARM_NAME.match(arm):
        raise ValueError(f"arm {arm!r} is not A0..A9")
    plan, feas = load_manifests(root)
    sidecar_fingerprint = contract.check_signal_source(repo)
    contract.s0.check_p2_frozen(repo)
    core = contract.core_provenance(repo)
    record = dict(
        arm=dict(name=arm, adapter=identifier, files=arm_files(identifier, repo), inputs=arm_inputs(identifier, repo),
                 runtime=runtime()),
        core=dict(provenance=core, sha256=contract.digest(core)),
        textures=contract.texture_files(repo),
        plan_sha256=contract.file_hash(root / "plan/plan.json"), feasibility_sha256=contract.file_hash(root / "feasibility/feasibility.json"),
        dropped_trials=feas["dropped_trials"], feasibility_counts=feas["counts"], s0_seal=sidecar_fingerprint,
        checkpoints={str(k): checkpoint_facts(v) for k, v in sorted(checkpoints.items())},
        prepared_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    record["arm"]["sha256"] = contract.digest([record["arm"]["adapter"], record["arm"]["files"],
                                               {d: {n: f["sha256"] for n, f in fs.items()} for d, fs in record["arm"]["inputs"].items()}])
    for other, theirs in prepared_arms(root).items():
        if other != arm and theirs["core"]["sha256"] != record["core"]["sha256"]:
            raise ValueError(f"arm {arm}: shared core differs from {other}'s; a core change is a new protocol and re-measures every arm")
    (root / "arms" / arm).mkdir(parents=True, exist_ok=True)
    write_new_json(root / "arms" / arm / "prepared.json", record)
    print("PREPARED", arm, json.dumps(dict(core=record["core"]["sha256"][:12], arm=record["arm"]["sha256"][:12],
                                           checkpoints=sorted(record["checkpoints"]), dropped=len(feas["dropped_trials"]))), flush=True)
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


def verify_textures(prepared, repo=ROOT):
    """Raises unless the texture files are exactly the ones recorded (full hash: 120 files, about 32 MB)."""
    if contract.texture_files(repo) != prepared["textures"]:
        raise ValueError("the texture files differ from the ones recorded at prepare")


def verify_arm(prepared, repo=ROOT, deep=False):
    """Raises unless the arm's adapter closure and input directories are what they were when the arm was prepared.
    Input files of LARGE_BYTES or more are checked by size and mtime unless `deep` (workers: cheap; `compare`: full)."""
    changed = sorted(r for r, h in prepared["arm"]["files"].items() if contract.file_hash(Path(repo) / r) != h)
    if changed:
        raise ValueError(f"arm {prepared['arm']['name']} changed since prepare: {changed[:5]}")
    for rel, recorded in prepared["arm"].get("inputs", {}).items():
        now = _input_files(Path(repo) / rel)
        if set(now) != set(recorded):
            raise ValueError(f"arm {prepared['arm']['name']}: files of input directory {rel} were added or removed")
        for name, record in recorded.items():
            stat = now[name].stat()
            full = deep or record["bytes"] < LARGE_BYTES
            if stat.st_size != record["bytes"] or (full and contract.file_hash(now[name]) != record["sha256"]) \
                    or (not full and stat.st_mtime_ns != record["mtime_ns"]):
                raise ValueError(f"arm {prepared['arm']['name']}: input file {rel}/{name} changed since prepare")


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
    """Restore the trial's scene (the layout under the trial's condition) and run it."""
    import mujoco
    from experiments.gen_eval import rollout, scenes
    from experiments.gen_scene.conditions import Condition
    condition = Condition.from_dict(trial["condition"])
    model, data, resolved = scenes.restore(layout, condition)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        return rollout.run(adapter, state_builder, model, data, resolved, renderer, trial, folder, condition)


def worker(root, arm, tier, checkpoint, shard, shards, repo=ROOT, factory=default_adapter_factory, runner=default_runner,
           cooler=cool_down):
    """One share of one stage, one checkpoint, one adapter built once. Returns the rows it ran."""
    if type(checkpoint) is not int:
        raise ValueError("a worker serves exactly one checkpoint")
    root = Path(root)
    prepared = load_prepared(root, arm)
    if str(checkpoint) not in prepared["checkpoints"]:
        raise ValueError(f"checkpoint {checkpoint} was not prepared for arm {arm}")
    verify_core(prepared, repo)
    verify_arm(prepared, repo)
    verify_manifests(root, prepared)
    verify_textures(prepared, repo)
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
            print("EVAL", directory.name, json.dumps({k: row[k] for k in ("ordinal", "layout_id", "axis", "level", "task_success", "cause")}),
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
    by_axis = {axis: dict(trials=sum(1 for r in rows if r["axis"] == axis), successes=sum(1 for r in rows if r["axis"] == axis and r["task_success"]))
               for axis in protocol.AXES}
    report = dict(status="complete" if not missing else "incomplete", arm=arm, tier=tier, checkpoint=checkpoint,
                  model_sha256=prepared["checkpoints"][str(checkpoint)]["model_sha256"], core_sha256=prepared["core"]["sha256"],
                  arm_sha256=prepared["arm"]["sha256"], dropped_trials=feas["dropped_trials"], planned_trials=len(trials),
                  missing_ordinals=missing, failure_files=sorted(p.name for p in directory.glob("failure_shard*.json")),
                  by_axis=by_axis, episodes=rows)
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
    """Choose a checkpoint on the selection tier only: most successes over every trial of the tier. A tie goes to the later
    checkpoint and is recorded. Never used for G0's own claims; it exists so a later arm's checkpoint is chosen without the
    benchmark tier."""
    root = Path(root)
    prepared = load_prepared(root, arm)
    counts = {}
    for key in sorted(prepared["checkpoints"], key=int):
        episodes = load_stage(root, arm, "selection", int(key))["episodes"]
        if any(e["tier"] != "selection" for e in episodes):
            raise ValueError("the selection tier holds only selection layouts")
        counts[int(key)] = sum(1 for e in episodes if e["task_success"])
    winner = stats.select(counts)
    decision = dict(arm=arm, counts={str(k): v for k, v in counts.items()}, tied=stats.tied(counts), winner=winner,
                    rule="most successes over every trial of the selection tier; a tie goes to the later checkpoint "
                         "(administrative, not evidence)",
                    note="the benchmark tier never chooses a checkpoint")
    write_new_json(root / "arms" / arm / "selection.json", decision)
    print("SELECTED", json.dumps(decision), flush=True)
    return decision


# --- compare / final ---------------------------------------------------------------------------------------------------------

def _cell(pairs):
    """Paired read-out of (layout, nominal episode, perturbed episode) triples."""
    rows = [dict(layout_id=lid, object=n["object"], nominal=n["task_success"], perturbed=p["task_success"]) for lid, n, p in pairs]
    lift = [dict(layout_id=lid, object=n["object"], nominal=n["grasp_lift"], perturbed=p["grasp_lift"]) for lid, n, p in pairs]
    both, nominal_only, perturbed_only, neither = stats.pair_counts(rows)
    drop = stats.paired_drop(rows)
    out = dict(layouts=len(rows), nominal_successes=sum(r["nominal"] for r in rows), perturbed_successes=sum(r["perturbed"] for r in rows),
               discordant=dict(nominal_only=nominal_only, perturbed_only=perturbed_only), mcnemar_exact_p=stats.mcnemar(perturbed_only, nominal_only),
               drop=drop, held=stats.held(drop),
               grasp_lift=dict(nominal=sum(r["nominal"] for r in lift), perturbed=sum(r["perturbed"] for r in lift),
                               drop=stats.paired_drop(lift)),
               causes=dict(sorted(Counter(p["cause"] for _, _, p in pairs).items())),
               nominal_causes=dict(sorted(Counter(n["cause"] for _, n, _ in pairs).items())))
    fired = [p for _, _, p in pairs if p.get("disturbance_fired")]
    if any(p["axis"] == "disturbance" for _, _, p in pairs):
        out["disturbance"] = dict(fired=len(fired), recovered=sum(1 for p in fired if p["recovered"]),
                                  not_fired=len(pairs) - len(fired))
    return out


def compare(root, arm, checkpoint, repo=ROOT):
    """One arm at one checkpoint on the benchmark tier: every axis and level paired against nominal on the same layouts."""
    root = Path(root)
    prepared = load_prepared(root, arm)
    verify_core(prepared, repo)
    verify_manifests(root, prepared)
    verify_textures(prepared, repo)
    verify_arm(prepared, repo, deep=True)
    for other, theirs in prepared_arms(root).items():
        if theirs["core"]["sha256"] != prepared["core"]["sha256"]:
            raise ValueError(f"the shared core of {other} differs from {arm}'s")
    stage = load_stage(root, arm, "benchmark", checkpoint)
    plan, feas = load_manifests(root)
    by = {(e["layout_id"], e["axis"]): e for e in stage["episodes"]}
    if len(by) != len(stage["episodes"]) or len(by) != stage["planned_trials"]:
        raise ValueError("duplicate or missing trials in the benchmark stage")
    layouts = [lid for lid in plan["tiers"]["benchmark"] if (lid, "nominal") in by]
    nominal_rows = [by[(lid, "nominal")] for lid in layouts]
    axes, pvalues = {}, {}
    for axis in protocol.AXES[1:]:
        pairs = [(lid, by[(lid, "nominal")], by[(lid, axis)]) for lid in layouts if (lid, axis) in by]
        entry = dict(pooled=_cell(pairs), levels={})
        for level in levels.LEVELS[axis]:
            subset = [t for t in pairs if t[2]["level"] == level]
            if subset:
                entry["levels"][level] = _cell(subset)
        axes[axis] = entry
        pvalues[axis] = entry["pooled"]["mcnemar_exact_p"]
    planned = feas["counts"]["benchmark"]
    result = dict(
        arm=arm, checkpoint=checkpoint, core_sha256=prepared["core"]["sha256"], arm_sha256=prepared["arm"]["sha256"],
        layouts_planned=len(plan["tiers"]["benchmark"]), layouts_with_nominal=len(layouts),
        nominal=dict(trials=len(nominal_rows), successes=sum(1 for r in nominal_rows if r["task_success"]),
                     rate=sum(1 for r in nominal_rows if r["task_success"]) / len(nominal_rows) if nominal_rows else None,
                     interval=list(analysis.clustered_interval(nominal_rows, "task_success")), causes=dict(sorted(Counter(r["cause"] for r in nominal_rows).items())),
                     grasp_lift=sum(1 for r in nominal_rows if r["grasp_lift"])),
        axes=axes, holm_over_axes=stats.holm(pvalues), held={axis: axes[axis]["pooled"]["held"] for axis in axes},
        planned_and_feasible=planned,
        population="feasible benchmark layouts; a trial the expert could not solve is dropped for every arm, counted by axis and "
                   "level, never replaced; a layout infeasible under nominal conditions leaves with all six trials",
        caveat="conditional on this checkpoint, recipe and the one training seed; layouts are the independent units; held = the paired "
               "drop's 95% interval upper bound is at most 10 points (pre-registered); the Holm adjustment over five axes is "
               "reported, no family-wise claim is made")
    write_new_json(root / "arms" / arm / f"compare_step_{checkpoint:06d}.json", result)
    print("COMPARE", arm, checkpoint, f"nominal {result['nominal']['successes']}/{len(nominal_rows)}",
          {axis: f"drop {axes[axis]['pooled']['drop']['estimate']:+.3f}" for axis in axes}, flush=True)
    return result


def final(root):
    """The per-arm table over every compared arm: nominal, each axis's pooled paired drop with its interval, held."""
    root = Path(root)
    arms = {}
    for path in sorted((root / "arms").glob("*/compare_step_*.json")):
        result = read_json(path)
        arms[f"{result['arm']}@{result['checkpoint']}"] = dict(
            file=path.name, nominal=result["nominal"]["rate"],
            drops={axis: result["axes"][axis]["pooled"]["drop"] for axis in result["axes"]}, held=result["held"],
            holm=result["holm_over_axes"])
    out = dict(arms=arms, note="no cross-arm test is made here; the per-arm paired drops are the measurement")
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
        prepare(args.root, args.arm, args.adapter, args.checkpoints)
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
