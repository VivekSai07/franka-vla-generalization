"""The M3 test: arm A3 (force state + drawn target box) against arm A2 (force state), on M0's comparison tier.

Not bound by the training contract and outside M0's frozen core: M0's `compare` is hard-wired to A0 as the control, so this module
loads both arms' comparison stages through M0's own loaders (read only) and applies M0's own statistics (`stats.mcnemar`,
`stats.paired_effect`, `stats.holm`, `analysis.readouts`) to the A3-against-A2 pairs. It refuses unless both arms were prepared with the
core whose hash is pinned here, both closures and the core still verify, the layout sets and the dropped layouts are identical,
and the pair of checkpoints is one the selection allows (the winners against each other, or the matched-phase checkpoint
against A2's): the comparison tier never chooses a checkpoint.

Multiplicity (spec D11, amended before any M3 result): the fixed family is {A1 vs A0, A2 vs A0, A3 vs A2, A4 vs A0}, Holm-adjusted,
arms that were not run entering at p = 1. The pre-registered criterion: the box helps on top of force iff A3's paired effect against
A2 is positive and its Holm-adjusted p is below 0.05. M0's own A3-against-A0 comparison (`evaluate.compare`) is secondary and
is run separately.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments.vla_inputs_eval import evaluate, protocol, stats
from experiments.xvla_props_p2_eval import analysis

ARM, CONTROL = "A3", "A2"
CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"
FAMILY = ("A1_vs_A0", "A2_vs_A0", "A3_vs_A2", "A4_vs_A0")
ALPHA = 0.05


def box_left_frame(directory, episodes):
    """Of `episodes` (trials of one stage), how many had the gripper-camera box out of frame at some tick, from M0's traces.

    `missing` counts trials whose trace is not on disk. Exploratory: the box leaves the frame at rollout, which never happened in
    training (every training frame has the target in frame).
    """
    checked = left = missing = 0
    for episode in episodes:
        path = protocol.trial_folder(directory, episode) / "trace.npz"
        if not path.is_file():
            missing += 1
            continue
        with np.load(path) as trace:
            box = trace["target_box"]
        checked += 1
        left += int((box[:, 1, 4] == 0).any())
    return dict(checked=checked, left_frame=left, missing=missing)


def exploratory(new_by, old_by, ids, directory):
    """Unadjusted, paired read-outs named by the spec; never decisive."""
    never = lambda episode: episode["cause"] == "never_lifted"
    new_never, old_never = {i for i in ids if never(new_by[i])}, {i for i in ids if never(old_by[i])}
    milk = [i for i in ids if new_by[i]["object"] == "milk"]
    held = [i for i in ids if new_by[i]["kind"] == "held_out"]
    return dict(
        never_lifted=dict(new=len(new_never), control=len(old_never), new_only=len(new_never - old_never),
                          control_only=len(old_never - new_never), both=len(new_never & old_never)),
        wrong_object=dict(new=sum(1 for i in ids if new_by[i]["wrong_object"]), control=sum(1 for i in ids if old_by[i]["wrong_object"])),
        milk=dict(n=len(milk), new=sum(new_by[i]["task_success"] for i in milk), control=sum(old_by[i]["task_success"] for i in milk)),
        held_out=dict(n=len(held), new=sum(new_by[i]["task_success"] for i in held), control=sum(old_by[i]["task_success"] for i in held)),
        gripper_box_left_frame=box_left_frame(directory, [new_by[i] for i in ids if not new_by[i]["task_success"]]))


def compare(root, checkpoint, control_checkpoint, repo=evaluate.ROOT, seed=0):
    """A3 at `checkpoint` against A2 at `control_checkpoint` on the comparison tier; writes `arms/A3/compare_A2_step_<ckpt>_vs_<control ckpt>.json`."""
    root = Path(root)
    mine, control = evaluate.load_prepared(root, ARM), evaluate.load_prepared(root, CONTROL)
    for name, prepared in ((CONTROL, control), (ARM, mine)):
        if prepared["core"]["sha256"] != CORE_SHA256:
            raise ValueError(f"{name} was prepared with a shared core other than the pinned {CORE_SHA256[:12]}")
        evaluate.verify_core(prepared, repo)
        evaluate.verify_manifests(root, prepared)
        evaluate.verify_arm(prepared, repo, deep=True)
    decision = evaluate.read_json(root / "arms" / ARM / "selection.json")
    control_decision = evaluate.read_json(root / "arms" / CONTROL / "selection.json")
    if checkpoint not in decision["comparison_checkpoints"] or control_checkpoint not in control_decision["comparison_checkpoints"]:
        raise ValueError(f"{ARM} {checkpoint} against {CONTROL} {control_checkpoint}: not both comparison checkpoints "
                         f"({decision['comparison_checkpoints']}, {control_decision['comparison_checkpoints']}); the comparison tier never chooses")
    winners = checkpoint == decision["winner"] and control_checkpoint == control_decision["winner"]
    phase = checkpoint == decision["phase_checkpoint"] and control_checkpoint == control_decision["phase_checkpoint"]
    if not (winners or phase):
        raise ValueError(f"{ARM} {checkpoint} against {CONTROL} {control_checkpoint} is neither the winners nor the matched-phase pair")
    role = "primary and matched-phase" if winners and phase else "primary" if winners else "matched-phase secondary"
    new, old = (evaluate.load_stage(root, ARM, "comparison", checkpoint),
                evaluate.load_stage(root, CONTROL, "comparison", control_checkpoint))
    new_by, old_by = {e["layout_id"]: e for e in new["episodes"]}, {e["layout_id"]: e for e in old["episodes"]}
    if (set(new_by) != set(old_by) or len(new_by) != new["planned_trials"] or new["planned_trials"] != old["planned_trials"]
            or set(new["dropped_layouts"]) != set(old["dropped_layouts"])):
        raise ValueError("incomplete or different pairs: both arms need a trial on every feasible comparison layout, "
                         "with the same dropped layouts")
    ids = sorted(new_by)
    plan, _ = evaluate.load_manifests(root)
    dropped = sorted(i for i in new["dropped_layouts"] if plan["layouts"][i]["tier"] == "comparison")
    dropped_by_cell = {}
    for layout in dropped:
        key = "-".join(map(str, plan["layouts"][layout]["cell"]))
        dropped_by_cell[key] = dropped_by_cell.get(key, 0) + 1
    rows = [dict(layout_id=i, object=new_by[i]["object"], new=new_by[i]["task_success"], control=old_by[i]["task_success"]) for i in ids]
    b = sum(1 for r in rows if r["new"] and not r["control"])
    c = sum(1 for r in rows if r["control"] and not r["new"])
    p = stats.mcnemar(b, c)
    result = dict(
        arm=ARM, control_arm=CONTROL, checkpoint=checkpoint, control_checkpoint=control_checkpoint, role=role, layouts=len(ids),
        new_successes=sum(r["new"] for r in rows), control_successes=sum(r["control"] for r in rows),
        discordant=dict(new_only=b, control_only=c), mcnemar_exact_p=p, effect=stats.paired_effect(rows, seed=seed),
        core_sha256=mine["core"]["sha256"], arm_sha256=mine["arm"]["sha256"], control_arm_sha256=control["arm"]["sha256"],
        dropped_layouts=dropped, dropped_by_cell=dict(sorted(dropped_by_cell.items())),
        population=f"{len(ids)} feasible comparison layouts of {len(plan['tiers']['comparison'])} planned; {len(dropped)} dropped for "
                   f"every arm (expert could not solve them), counted by cell and never replaced",
        discordant_layouts=dict(new_only=[r["layout_id"] for r in rows if r["new"] and not r["control"]],
                                control_only=[r["layout_id"] for r in rows if r["control"] and not r["new"]]),
        readouts=dict(new=analysis.readouts(new["episodes"], []), control=analysis.readouts(old["episodes"], [])),
        exploratory=exploratory(new_by, old_by, ids, protocol.stage_dir(root, ARM, "comparison", checkpoint)),
        caveat="conditional on these checkpoints, recipes and the one training seed per arm; layouts are the independent units")
    evaluate.write_new_json(root / "arms" / ARM / f"compare_A2_step_{checkpoint:06d}_vs_{control_checkpoint:06d}.json", result)
    print("COMPARE", ARM, checkpoint, "vs", CONTROL, control_checkpoint, f"{result['new_successes']}/{len(ids)} vs "
          f"{result['control_successes']}/{len(ids)}", f"b={b} c={c} p={p:.4f}", flush=True)
    return result


def _selection_winner(root, arm):
    path = Path(root) / "arms" / arm / "selection.json"
    return evaluate.read_json(path)["winner"] if path.is_file() else None


def _existing(path):
    return path if path is not None and path.is_file() else None


def final(root):
    """Holm over the amended family, read at each arm's selection winner; the pre-registered criterion for A3.

    A1 and A4 (not built) and any arm without a comparison file enter as p = 1.
    """
    root = Path(root)
    raw, files = {name: 1.0 for name in FAMILY}, {}
    a2_winner, a3_winner = _selection_winner(root, CONTROL), _selection_winner(root, ARM)
    a2 = _existing(None if a2_winner is None else root / "arms" / CONTROL / f"compare_step_{a2_winner:06d}.json")
    a3 = _existing(None if None in (a2_winner, a3_winner)
                   else root / "arms" / ARM / f"compare_A2_step_{a3_winner:06d}_vs_{a2_winner:06d}.json")
    for name, path in (("A2_vs_A0", a2), ("A3_vs_A2", a3)):
        if path is not None:
            files[name] = path.relative_to(root).as_posix()
            raw[name] = evaluate.read_json(path)["mcnemar_exact_p"]
    holm = stats.holm(raw)
    effect = evaluate.read_json(a3)["effect"]["estimate"] if a3 is not None else None
    out = dict(raw=raw, holm=holm, files=files, family=list(FAMILY), effect_a3_vs_a2=effect,
               criterion_met=bool(effect is not None and effect > 0 and holm["A3_vs_A2"] < ALPHA),
               note="amended family (spec D11); A3 vs A2 at the selection winners is the primary test; the matched-phase pair and M0's "
                    "native A3-against-A0 comparison are secondary; unrun arms enter at p = 1; criterion: positive paired effect "
                    "and Holm-adjusted p < 0.05")
    evaluate.write_json(root / "final_amended.json", out)
    print("FINAL_AMENDED", out["criterion_met"], {k: round(v, 6) for k, v in holm.items()}, flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", required=True, help="M0's campaign directory")
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("compare", help="A3 at --checkpoint against A2 at --control-checkpoint")
    pair.add_argument("--checkpoint", type=int, required=True)
    pair.add_argument("--control-checkpoint", type=int, required=True)
    commands.add_parser("final", help="Holm over the amended family and the pre-registered criterion")
    args = parser.parse_args(argv)
    if args.command == "compare":
        compare(args.root, args.checkpoint, args.control_checkpoint)
    else:
        final(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
