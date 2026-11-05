"""M3b analysis (spec D9): how much of A3's gain over A2 survives a box predicted from pixels, and what box errors cost.

Outside M0's frozen core: M0's `compare` is hard-wired to A0, so this module loads the arms' comparison stages through M0's own
readers (read only) and applies M0's statistics (`stats.mcnemar`, `stats.paired_effect`, `stats.holm`, its stratified resampling
convention) to the pairs M3b asks about. Pure numpy and json: the admission image, no simulator.

Everything below the constants is fixed before any A5-A9 trial and does not change after a result:

* the quantity: **R = (S - F) / (C - F)**, S = A5's successes on the 573 comparison layouts at checkpoint 101,520, F = A2's (88,830,
  its selected checkpoint: 523) and C = A3's (101,520: 571); the secondary floor is A2 at 101,520 (517). R is read with a layout
  bootstrap stratified by object, numerator and denominator resampled together (10,000 resamples, seed 0);
* the decision rule (D10): R >= 0.75 closes M3b; 0.25 <= R < 0.75 is partial (M3b-2 considered); R < 0.25, or A5 not above A2
  (positive paired effect and exact McNemar p < 0.05), points to M4a. The point estimate picks the branch; a threshold the interval
  straddles is reported as such;
* A6-A9 are each tested against A3 (exact McNemar), Holm over those four, exploratory; arms not run enter as p = 1;
* A9's reading: >= 90% language dominates, 40 to 90% mixed, < 40% the box is followed;
* the prediction recorded on 2026-11-05 before any run, in `PREDICTION`.

It refuses unless every arm was prepared with core `c4f6c303f2ae` (and, by default, the core and every arm still verify against the
repository), the layouts and the dropped layouts are identical, and the stages it needs are complete: A2 at 88,830 and 101,520, A3
and A5 at 101,520. It writes one new file, `m3b_analysis.json`, never inside an arm directory and never over an existing file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.vla_inputs_eval import evaluate, protocol, stats
from experiments.xvla_props_p2_eval import analysis

CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"
CHECKPOINT = 101520            # the single comparison checkpoint of A3 and of A5-A9
FLOOR_CHECKPOINT = 88830       # A2's selected checkpoint: the primary floor
CEILING_CHECKPOINT = 101520    # A3's checkpoint
CONTROL, CEILING, CANDIDATE = "A2", "A3", "A5"
CORRUPTIONS = ("A6", "A7", "A8", "A9")
CLOSED_AT, PARTIAL_AT, ALPHA = 0.75, 0.25, 0.05
LANGUAGE_AT, FOLLOWED_BELOW = 0.90, 0.40
REPLAN_STRIDE = 10             # X-VLA runs the policy every tenth tick
IOU_BINS = 10
IMAGE = 512
BOX_LOG_DIRECTORY = "predicted_box_logs"
BRANCH_CLOSED = "M3b closes; M4a reconsidered at lower priority"
BRANCH_PARTIAL = "partial: M3b-2 considered (retrain on noisy or predicted boxes)"
BRANCH_M4A = "M4a next (the gain needs near-perfect information)"
CLASSES = ("never_lifted", "wrong_object", "timeout_carrying_or_unstable", "dropped_outside")

# Recorded 2026-11-05, before any M3b run (spec "Prediction"). Counts of A5's failures by class are ranges for the count, not gates.
PREDICTION = dict(
    A5=dict(R=(0.6, 0.95), successes=(552, 569),
            classes=dict(never_lifted=(3, 20), wrong_object=(0, 12), timeout_carrying_or_unstable=(0, 8), dropped_outside=(0, 3))),
    A6=dict(successes=(560, 571)), A7=dict(successes=(500, 565)), A8=dict(successes=(480, 571)),
    A9=dict(successes_below=458, main_cause="wrong_object"))


# --- stages --------------------------------------------------------------------------------------------------------------

def _by_layout(rows, what):
    out = {r["layout_id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{what}: a layout appears twice")
    return out


def _paired(*named):
    """The rows of several arms keyed by layout, checked to cover the same layouts and objects; returns sorted ids and the dicts."""
    dicts = [_by_layout(rows, name) for name, rows in named]
    ids = sorted(dicts[0])
    for (name, _), other in zip(named[1:], dicts[1:]):
        if sorted(other) != ids:
            raise ValueError(f"{name} and {named[0][0]} do not cover the same layouts ({len(other)} against {len(ids)})")
        if any(other[i]["object"] != dicts[0][i]["object"] for i in ids):
            raise ValueError(f"{name} and {named[0][0]} disagree on a layout's object")
    return ids, dicts


def recovery(a5_rows, a2_rows, a3_rows, seed=0, resamples=10_000):
    """R = (S - F) / (C - F) with a layout bootstrap stratified by object, the three arms resampled together."""
    ids, (d5, d2, d3) = _paired(("A5", a5_rows), ("A2", a2_rows), ("A3", a3_rows))
    s, f, c = (sum(bool(d[i]["task_success"]) for i in ids) for d in (d5, d2, d3))
    if c <= f:
        raise ValueError(f"recovery is undefined: the ceiling ({c}) does not beat the floor ({f})")
    objects = sorted({d3[i]["object"] for i in ids})
    groups = {o: [i for i in ids if d3[i]["object"] == o] for o in objects}
    rng = np.random.default_rng(seed)
    draws = stats._indices({o: np.zeros(len(g)) for o, g in groups.items()}, resamples, rng)
    totals = {}
    for name, d in (("s", d5), ("f", d2), ("c", d3)):
        totals[name] = sum(np.asarray([int(bool(d[i]["task_success"])) for i in g], dtype=float)[draws[o]].sum(axis=1)
                           for o, g in groups.items())
    gap = totals["c"] - totals["f"]
    usable = gap > 0
    ratio = (totals["s"] - totals["f"])[usable] / gap[usable]
    return dict(R=(s - f) / (c - f), lo=float(np.percentile(ratio, 2.5)), hi=float(np.percentile(ratio, 97.5)),
                successes=s, floor=f, ceiling=c, gap=c - f, layouts=len(ids), resamples=int(resamples), seed=seed,
                degenerate_resamples=int((~usable).sum()),
                method="layouts resampled stratified by object, the three arms together; percentile interval over resamples with C* > F*")


def compare_to(new_rows, control_rows, seed=0, resamples=10_000):
    """Exact McNemar and M0's paired effect (stratified bootstrap) of `new_rows` against `control_rows` over the same layouts."""
    ids, (new, control) = _paired(("new", new_rows), ("control", control_rows))
    pairs = [dict(layout_id=i, object=new[i]["object"], new=bool(new[i]["task_success"]), control=bool(control[i]["task_success"])) for i in ids]
    b = sum(1 for p in pairs if p["new"] and not p["control"])
    c = sum(1 for p in pairs if p["control"] and not p["new"])
    both = sum(1 for p in pairs if p["new"] and p["control"])
    return dict(layouts=len(ids), new_successes=b + both, control_successes=c + both, new_only=b, control_only=c, both=both,
                neither=len(ids) - b - c - both, mcnemar_exact_p=stats.mcnemar(b, c), effect=stats.paired_effect(pairs, n=resamples, seed=seed))


def holm_over_corruptions(pvalues):
    """Holm over A6-A9 against A3 (arms not run enter as p = 1); anything else is refused."""
    outsiders = sorted(set(pvalues) - set(CORRUPTIONS))
    if outsiders:
        raise ValueError(f"Holm runs over {CORRUPTIONS} only, not {outsiders}")
    raw = {arm: pvalues.get(arm, 1.0) for arm in CORRUPTIONS}
    return dict(raw=raw, adjusted=stats.holm(raw), family=list(CORRUPTIONS), missing=[a for a in CORRUPTIONS if a not in pvalues])


# --- decision rule -------------------------------------------------------------------------------------------------------

def _band(r):
    return BRANCH_CLOSED if r >= CLOSED_AT else BRANCH_PARTIAL if r >= PARTIAL_AT else BRANCH_M4A


def branch(r, a5_vs_a2, interval=None):
    """The D10 branch: the point estimate picks it; A5 not above A2 (not a positive effect with p < 0.05) sends it to M4a."""
    above = a5_vs_a2["effect"]["estimate"] > 0 and a5_vs_a2["mcnemar_exact_p"] < ALPHA
    chosen = _band(r) if above else BRANCH_M4A
    out = dict(branch=chosen, R=r, a5_above_a2=bool(above), thresholds=dict(closed_at=CLOSED_AT, partial_at=PARTIAL_AT, alpha=ALPHA))
    if interval is not None:
        out["interval"] = list(interval)
        out["straddles"] = _band(interval[0]) != _band(interval[1])
    return out


def a9_reading(rate):
    if rate >= LANGUAGE_AT:
        return "language dominates: the box is supplementary localisation"
    return "mixed" if rate >= FOLLOWED_BELOW else "the box is followed"


# --- failure classes -----------------------------------------------------------------------------------------------------

def failure_classes(rows):
    causes = {cause: sum(1 for r in rows if r["cause"] == cause) for cause in analysis.CAUSES}
    out = dict(causes)
    out["timeout_carrying_or_unstable"] = causes["timeout_carrying"] + causes["unstable_in_target"]
    out["failures"] = len(rows) - causes["success"]
    return out


def against_prediction(arm, counts):
    ranges = PREDICTION[arm]["classes"]
    return {name: dict(count=counts[name], range=ranges[name], within=ranges[name][0] <= counts[name] <= ranges[name][1]) for name in ranges}


# --- the detector's per-tick error ---------------------------------------------------------------------------------------

def box_iou(a, b):
    """IoU of two boxes `(u_min, v_min, u_max, v_max, in_frame)`; two absent boxes agree (1.0), one absent box gives 0.0."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a[4] == 0.0 or b[4] == 0.0:
        return 1.0 if a[4] == b[4] else 0.0
    inside = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inside
    return float(inside / union)


def _steps(boxes):
    """Mean-able |change| in pixels of the four coordinates between consecutive replan ticks, where both boxes are in frame."""
    ticks = np.arange(0, len(boxes), REPLAN_STRIDE)
    chosen = boxes[ticks]
    out = []
    for before, after in zip(chosen[:-1], chosen[1:]):
        if before[4] and after[4]:
            out.append(np.abs(after[:4] - before[:4]).mean() * IMAGE)
    return out


def _camera(predicted, oracle):
    """Per-tick arrays and replan steps for one camera of one trial; `predicted`, `oracle`: (ticks, 5)."""
    return dict(iou=np.array([box_iou(p, o) for p, o in zip(predicted, oracle)]),
                predicted_absent=predicted[:, 4] == 0, oracle_absent=oracle[:, 4] == 0,
                predicted_steps=_steps(predicted), oracle_steps=_steps(oracle))


def _summary(parts):
    """Pool per-trial camera parts (a list of `_camera` dicts) into the metrics the report quotes."""
    if not parts:
        return None
    iou = np.concatenate([p["iou"] for p in parts])
    pa = np.concatenate([p["predicted_absent"] for p in parts])
    oa = np.concatenate([p["oracle_absent"] for p in parts])
    mean = lambda values: float(np.mean(values)) if len(values) else None
    return dict(ticks=int(len(iou)), mean_iou=float(iou.mean()), share_below_half=float((iou < 0.5).mean()),
                share_predicted_absent=float(pa.mean()), share_oracle_absent=float(oa.mean()),
                share_missed=float((pa & ~oa).mean()), share_false_alarm=float((~pa & oa).mean()),
                iou_histogram=np.histogram(iou, bins=IOU_BINS, range=(0.0, 1.0))[0].tolist(),
                replan_jitter_px=mean(sum((p["predicted_steps"] for p in parts), [])),
                oracle_replan_jitter_px=mean(sum((p["oracle_steps"] for p in parts), [])))


def _read_logs(log_dir):
    """{(file, trial): (predicted (ticks, 2, 5), oracle (ticks, 2, 5))} from the JSON-lines logs; ticks must run 0..n-1."""
    groups = {}
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                groups.setdefault((path.name, record["trial"]), []).append(record)
    out = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r["tick"])
        if [r["tick"] for r in records] != list(range(len(records))):
            raise ValueError(f"log group {key}: ticks are not 0..{len(records) - 1}")
        out[key] = (np.array([r["predicted"] for r in records], dtype=np.float64), np.array([r["oracle"] for r in records], dtype=np.float64))
    return out


def detector_error(log_dir, stage_dir):
    """Per-tick error of A5's predicted box against the live oracle box, from the adapter's logs and M0's traces.

    A log group (one trial's acts) is matched to a trial by the oracle box sequence, which must equal the trace's `target_box`
    byte for byte; a trace without a log, and a log without a trace, are reported.
    """
    logs = _read_logs(log_dir)
    by_oracle = {}
    for key, (_, oracle) in logs.items():
        by_oracle.setdefault((oracle.shape, oracle.tobytes()), []).append(key)
    matched, without_log, used = [], [], set()
    for folder in sorted(Path(stage_dir).glob("trial_*")):
        episode = json.loads((folder / "episode.json").read_text())
        with np.load(folder / "trace.npz") as trace:
            oracle = np.asarray(trace["target_box"], dtype=np.float64)
        keys = [k for k in by_oracle.get((oracle.shape, oracle.tobytes()), []) if k not in used]
        if not keys:
            without_log.append(episode["layout_id"])
            continue
        used.add(keys[0])
        matched.append((episode, logs[keys[0]][0], oracle))
    per_trial, pooled, by_outcome = [], {"front": [], "gripper": []}, {"success": {"front": [], "gripper": [], "trials": 0},
                                                                      "failure": {"front": [], "gripper": [], "trials": 0}}
    for episode, predicted, oracle in matched:
        outcome = "success" if episode["task_success"] else "failure"
        by_outcome[outcome]["trials"] += 1
        row = dict(layout_id=episode["layout_id"], ordinal=episode["ordinal"], task_success=bool(episode["task_success"]), ticks=len(oracle))
        for index, camera in enumerate(("front", "gripper")):
            part = _camera(predicted[:, index], oracle[:, index])
            pooled[camera].append(part)
            by_outcome[outcome][camera].append(part)
            row[camera] = _summary([part])
        per_trial.append(row)
    return dict(trials_matched=len(matched), trials_without_log=without_log, log_groups_without_trace=len(logs) - len(used),
                overall={camera: _summary(parts) for camera, parts in pooled.items()},
                by_outcome={k: dict(trials=v["trials"], front=_summary(v["front"]), gripper=_summary(v["gripper"])) for k, v in by_outcome.items()},
                per_trial=per_trial, replan_stride=REPLAN_STRIDE)


# --- the analysis --------------------------------------------------------------------------------------------------------

def _load(root, arm, checkpoint, verify, repo):
    prepared = evaluate.load_prepared(root, arm)
    if prepared["core"]["sha256"] != CORE_SHA256:
        raise ValueError(f"{arm} was prepared with a shared core other than the pinned {CORE_SHA256[:12]}")
    stage = evaluate.load_stage(root, arm, "comparison", checkpoint)
    if stage["core_sha256"] != CORE_SHA256:
        raise ValueError(f"{arm} at {checkpoint}: the stage ran with a shared core other than the pinned {CORE_SHA256[:12]}")
    if verify:
        evaluate.verify_core(prepared, repo)
        evaluate.verify_manifests(root, prepared)
        evaluate.verify_arm(prepared, repo, deep=True)
    return prepared, stage


def analyse(root, verify=True, repo=evaluate.ROOT, resamples=10_000, seed=0, box_log_dir=None):
    """The M3b result: R and its interval, A5 against A2 and A3, A6-A9 against A3 with Holm, failure classes, the detector's error."""
    root = Path(root)
    stages = {}
    for arm, checkpoint in ((CONTROL, FLOOR_CHECKPOINT), (CONTROL, CEILING_CHECKPOINT), (CEILING, CEILING_CHECKPOINT), (CANDIDATE, CHECKPOINT)):
        stages[(arm, checkpoint)] = _load(root, arm, checkpoint, verify, repo)[1]
    not_run = []
    for arm in CORRUPTIONS:
        if (root / "arms" / arm / "prepared.json").is_file() and protocol.stage_dir(root, arm, "comparison", CHECKPOINT).joinpath("result.json").is_file():
            stages[(arm, CHECKPOINT)] = _load(root, arm, CHECKPOINT, verify, repo)[1]
        else:
            not_run.append(arm)
    reference = stages[(CEILING, CEILING_CHECKPOINT)]
    for (arm, checkpoint), stage in stages.items():
        if sorted(stage["dropped_layouts"]) != sorted(reference["dropped_layouts"]):
            raise ValueError(f"{arm} at {checkpoint}: other dropped layouts than A3's")
        if stage["planned_trials"] != reference["planned_trials"]:
            raise ValueError(f"{arm} at {checkpoint}: another number of planned trials than A3's")
    rows = {key: stage["episodes"] for key, stage in stages.items()}
    a2, a2_phase, a3, a5 = rows[(CONTROL, FLOOR_CHECKPOINT)], rows[(CONTROL, CEILING_CHECKPOINT)], rows[(CEILING, CEILING_CHECKPOINT)], rows[(CANDIDATE, CHECKPOINT)]
    _paired(("A3", a3), ("A5", a5), ("A2", a2), ("A2 matched phase", a2_phase))
    count = lambda r: sum(bool(x["task_success"]) for x in r)
    main = recovery(a5, a2, a3, seed=seed, resamples=resamples)
    versus_a2 = compare_to(a5, a2, seed=seed, resamples=resamples)
    corruptions, pvalues = {}, {}
    for arm in CORRUPTIONS:
        if (arm, CHECKPOINT) in rows:
            versus = compare_to(rows[(arm, CHECKPOINT)], a3, seed=seed, resamples=resamples)
            pvalues[arm] = versus["mcnemar_exact_p"]
            corruptions[arm] = dict(successes=count(rows[(arm, CHECKPOINT)]), rate=count(rows[(arm, CHECKPOINT)]) / len(a3), vs_a3=versus,
                                    failure_classes=failure_classes(rows[(arm, CHECKPOINT)]))
    classes = failure_classes(a5)
    log_dir = Path(box_log_dir) if box_log_dir else root / "arms" / CANDIDATE / BOX_LOG_DIRECTORY
    result = dict(
        core_sha256=CORE_SHA256, checkpoint=CHECKPOINT, floor_checkpoint=FLOOR_CHECKPOINT, layouts=len(a3), dropped_layouts=sorted(reference["dropped_layouts"]),
        floor=count(a2), ceiling=count(a3), a5=count(a5), recovery=main,
        recovery_matched_phase=recovery(a5, a2_phase, a3, seed=seed, resamples=resamples),
        a5_vs_a2=versus_a2, a5_vs_a3=compare_to(a5, a3, seed=seed, resamples=resamples),
        decision=branch(main["R"], versus_a2, interval=(main["lo"], main["hi"])),
        failure_classes=dict(A2=failure_classes(a2), A3=failure_classes(a3), A5=classes),
        a5_against_prediction=against_prediction(CANDIDATE, classes),
        prediction=dict(A5=dict(R=PREDICTION["A5"]["R"], successes=PREDICTION["A5"]["successes"]),
                        **{arm: PREDICTION[arm] for arm in ("A6", "A7", "A8", "A9")}),
        corruptions=corruptions, holm=holm_over_corruptions(pvalues), not_run=not_run,
        a9_reading=a9_reading(corruptions["A9"]["rate"]) if "A9" in corruptions else None,
        detector_error=(detector_error(log_dir, protocol.stage_dir(root, CANDIDATE, "comparison", CHECKPOINT)) if log_dir.is_dir() else None),
        caveat="conditional on these checkpoints and the one training seed; layouts are the independent units; the forces stay privileged")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", required=True, help="M0's campaign directory (read only)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("analyse", help="write m3b_analysis.json under --root (never over an existing file)")
    run.add_argument("--no-verify", action="store_true", help="skip re-verifying the core and the arms against the repository")
    run.add_argument("--resamples", type=int, default=10_000)
    run.add_argument("--box-log-dir", help=f"A5's predicted-box logs (default: arms/A5/{BOX_LOG_DIRECTORY})")
    args = parser.parse_args(argv)
    result = analyse(args.root, verify=not args.no_verify, resamples=args.resamples, box_log_dir=args.box_log_dir)
    evaluate.write_new_json(Path(args.root) / "m3b_analysis.json", result)
    print("M3B_ANALYSIS", json.dumps(dict(floor=result["floor"], ceiling=result["ceiling"], a5=result["a5"], R=round(result["recovery"]["R"], 4),
                                          branch=result["decision"]["branch"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
