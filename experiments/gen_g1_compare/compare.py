"""G1's primary test (spec D10): arm A1 (X-VLA trained on G1's randomized data) against A2 on G0's benchmark tier.

Outside G0's frozen core and M0's: `gen_eval.evaluate.compare` reads one arm's own drops, and M0's `compare` is hard-wired to A0
as the control. This module loads both arms' stages through the evaluators' own loaders (read only), applies G0's own statistics
(`gen_eval.stats`) and M0's (`vla_inputs_eval.stats`), and refuses unless both arms were prepared with the pinned core, the same
plan, feasibility and dropped trials, and every trial of both arms is present.

Per axis 1-5, on the perturbed trials paired by layout: exact McNemar of A1 against A2 (discordant pairs), Holm over the five axes,
alpha 0.05. Criterion per axis: A1 HOLDS the axis (G0 D13: A1's own paired drop, nominal minus perturbed, has a 95% interval upper
bound of at most 10 points) AND beats A2 (more successes on the axis's perturbed trials and Holm-adjusted p below 0.05). Nominal is
reported with intervals; A1 more than 5 points below A2 on nominal is flagged as a cost. Levels are descriptive, unadjusted.

A1's checkpoint is the selection winner or the matched-phase checkpoint 101,520; A2's is 88,830, G0's control. The benchmark tier
never chooses a checkpoint. `compare_m0` is the secondary read: nominal, M0's comparison tier, A1's winner against A2.
"""
import argparse
from pathlib import Path

from experiments.gen_eval import evaluate as g0, protocol as g0_protocol, stats
from experiments.vla_inputs_eval import evaluate as m0, stats as m0_stats

G0_CORE_SHA256 = "512806282b1fb2d7ebdf516ef02cd07d4e1ec1fd559f6a76a060eb8330121ce5"
M0_CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"
NEW_ARM, CONTROL = "A1", "A2"
CONTROL_CHECKPOINT, PHASE_CHECKPOINT = 88830, 101520
AXES = g0_protocol.AXES[1:]
ALPHA, NOMINAL_COST = 0.05, 0.05


def _pairs(new_by, old_by, axis):
    """Layouts with a nominal trial and an `axis` trial in both arms, in sorted order."""
    return sorted(lid for lid, a in new_by if a == axis and (lid, "nominal") in new_by and (lid, "nominal") in old_by
                  and (lid, axis) in old_by)


def _cell(new_by, old_by, layouts, axis):
    """One axis (or one level of it) over `layouts`: A1 against A2 on the perturbed trials, and A1's own drop."""
    new_p = [new_by[(lid, axis)] for lid in layouts]
    old_p = [old_by[(lid, axis)] for lid in layouts]
    b = sum(1 for n, o in zip(new_p, old_p) if n["task_success"] and not o["task_success"])
    c = sum(1 for n, o in zip(new_p, old_p) if o["task_success"] and not n["task_success"])
    own = [dict(object=new_by[(lid, "nominal")]["object"], nominal=new_by[(lid, "nominal")]["task_success"],
                perturbed=new_by[(lid, axis)]["task_success"]) for lid in layouts]
    # `paired_drop` is mean(nominal - perturbed): with A1 as "nominal" and A2 as "perturbed" it is A1's gain over A2
    gain = stats.paired_drop([dict(object=n["object"], nominal=n["task_success"], perturbed=o["task_success"])
                              for n, o in zip(new_p, old_p)])
    drop = stats.paired_drop(own)
    return dict(layouts=len(layouts), new_successes=sum(n["task_success"] for n in new_p),
                control_successes=sum(o["task_success"] for o in old_p), discordant=dict(new_only=b, control_only=c),
                mcnemar_exact_p=stats.mcnemar(b, c), gain_over_control=gain, new_drop=drop, new_held=stats.held(drop))


def analyse(new_episodes, old_episodes):
    """The D10 read-out over two complete benchmark stages (lists of episode records)."""
    new_by = {(e["layout_id"], e["axis"]): e for e in new_episodes}
    old_by = {(e["layout_id"], e["axis"]): e for e in old_episodes}
    if len(new_by) != len(new_episodes) or len(old_by) != len(old_episodes):
        raise ValueError("duplicate trials in a stage")
    if set(new_by) != set(old_by):
        raise ValueError("the two arms do not hold the same trials")
    axes, raw = {}, {}
    for axis in AXES:
        layouts = _pairs(new_by, old_by, axis)
        entry = dict(pooled=_cell(new_by, old_by, layouts, axis), levels={})
        for level in sorted({new_by[(lid, axis)]["level"] for lid in layouts}):
            subset = [lid for lid in layouts if new_by[(lid, axis)]["level"] == level]
            entry["levels"][level] = _cell(new_by, old_by, subset, axis)
        axes[axis], raw[axis] = entry, entry["pooled"]["mcnemar_exact_p"]
    holm = stats.holm(raw)
    for axis, entry in axes.items():
        pooled = entry["pooled"]
        entry["holm_p"] = holm[axis]
        entry["beats_control"] = bool(pooled["new_successes"] > pooled["control_successes"] and holm[axis] < ALPHA)
        entry["criterion_met"] = bool(pooled["new_held"] and entry["beats_control"])
    layouts = sorted(lid for lid, a in new_by if a == "nominal")
    nominal = _cell(new_by, old_by, layouts, "nominal")
    for key in ("new_drop", "new_held"):
        nominal.pop(key)
    nominal["cost"] = bool(nominal["gain_over_control"]["estimate"] < -NOMINAL_COST)
    return dict(axes=axes, nominal=nominal, holm=holm, criterion_met={axis: axes[axis]["criterion_met"] for axis in axes})


def check_pair(new, old, new_stage, old_stage, checkpoint, winner):
    """Raises unless the two arms are comparable and `checkpoint` is one the selection allows."""
    for name, prepared in ((NEW_ARM, new), (CONTROL, old)):
        if prepared["core"]["sha256"] != G0_CORE_SHA256:
            raise ValueError(f"{name} was prepared with a shared core other than the pinned {G0_CORE_SHA256[:12]}")
    for key in ("plan_sha256", "feasibility_sha256", "dropped_trials"):
        if new[key] != old[key]:
            raise ValueError(f"the two arms were prepared with different {key}")
    if checkpoint not in (winner, PHASE_CHECKPOINT):
        raise ValueError(f"checkpoint {checkpoint} is neither the selection winner {winner} nor {PHASE_CHECKPOINT}; "
                         "the benchmark tier never chooses a checkpoint")
    if new_stage["planned_trials"] != old_stage["planned_trials"] or new_stage["dropped_trials"] != old_stage["dropped_trials"]:
        raise ValueError("the stages do not plan the same trials")
    for name, stage in ((NEW_ARM, new_stage), (CONTROL, old_stage)):
        if stage["status"] != "complete" or len(stage["episodes"]) != stage["planned_trials"] or stage["missing_ordinals"]:
            raise ValueError(f"{name}'s benchmark stage is not complete")
    return "primary and matched-phase" if checkpoint == winner == PHASE_CHECKPOINT else \
        "primary" if checkpoint == winner else "matched-phase secondary"


def compare(root, checkpoint, repo=g0.ROOT):
    """A1 at `checkpoint` against A2 at 88,830 on the benchmark tier; writes `arms/A1/compare_A2_step_<checkpoint>.json`."""
    root = Path(root)
    new, old = g0.load_prepared(root, NEW_ARM), g0.load_prepared(root, CONTROL)
    for prepared in (old, new):
        g0.verify_core(prepared, repo)
        g0.verify_manifests(root, prepared)
        g0.verify_textures(prepared, repo)
        g0.verify_arm(prepared, repo, deep=True)
    winner = g0.read_json(root / "arms" / NEW_ARM / "selection.json")["winner"]
    new_stage = g0.load_stage(root, NEW_ARM, "benchmark", checkpoint)
    old_stage = g0.load_stage(root, CONTROL, "benchmark", CONTROL_CHECKPOINT)
    role = check_pair(new, old, new_stage, old_stage, checkpoint, winner)
    result = dict(arm=NEW_ARM, control_arm=CONTROL, checkpoint=checkpoint, control_checkpoint=CONTROL_CHECKPOINT, role=role,
                  core_sha256=new["core"]["sha256"], arm_sha256=new["arm"]["sha256"], control_arm_sha256=old["arm"]["sha256"],
                  model_sha256=new["checkpoints"][str(checkpoint)]["model_sha256"],
                  **analyse(new_stage["episodes"], old_stage["episodes"]),
                  caveat="conditional on these checkpoints and the one training seed per arm; layouts are the independent units; "
                         "A1 saw 0.89 epoch of its data, A2 5.0 epochs of P2's (same 126,900-update recipe, effective batch 8); "
                         "'held' is A1's own paired drop against its own nominal (upper bound at most 10 points)")
    g0.write_new_json(root / "arms" / NEW_ARM / f"compare_A2_step_{checkpoint:06d}.json", result)
    print("COMPARE_A1_VS_A2", checkpoint, role, "criterion", result["criterion_met"], "nominal",
          f"{result['nominal']['new_successes']}/{result['nominal']['layouts']} vs {result['nominal']['control_successes']}",
          "holm", {k: round(v, 6) for k, v in result["holm"].items()}, flush=True)
    return result


def analyse_m0(new_by, old_by):
    """The secondary read over two complete comparison stages ({layout: episode}): M0's own statistics."""
    if set(new_by) != set(old_by):
        raise ValueError("incomplete or different pairs: both arms need a trial on every feasible comparison layout")
    ids = sorted(new_by)
    rows = [dict(layout_id=i, object=new_by[i]["object"], new=new_by[i]["task_success"], control=old_by[i]["task_success"]) for i in ids]
    b = sum(1 for r in rows if r["new"] and not r["control"])
    c = sum(1 for r in rows if r["control"] and not r["new"])
    effect = m0_stats.paired_effect(rows, seed=0)
    return dict(layouts=len(ids), new_successes=sum(r["new"] for r in rows), control_successes=sum(r["control"] for r in rows),
                discordant=dict(new_only=b, control_only=c), mcnemar_exact_p=m0_stats.mcnemar(b, c), effect=effect,
                cost=bool(effect["estimate"] < -NOMINAL_COST))


def compare_m0(m0_root, g0_root, checkpoint, repo=m0.ROOT):
    """Secondary: A1 at G0's selection winner against A2 at 88,830 on M0's comparison tier (nominal, 573 layouts)."""
    m0_root = Path(m0_root)
    winner = g0.read_json(Path(g0_root) / "arms" / NEW_ARM / "selection.json")["winner"]
    if checkpoint != winner:
        raise ValueError(f"checkpoint {checkpoint} is not G0's selection winner {winner}; the comparison tier never chooses")
    new, old = m0.load_prepared(m0_root, NEW_ARM), m0.load_prepared(m0_root, CONTROL)
    for name, prepared in ((CONTROL, old), (NEW_ARM, new)):
        if prepared["core"]["sha256"] != M0_CORE_SHA256:
            raise ValueError(f"{name} was prepared with a shared core other than the pinned {M0_CORE_SHA256[:12]}")
        m0.verify_core(prepared, repo)
        m0.verify_manifests(m0_root, prepared)
        m0.verify_arm(prepared, repo, deep=True)
    new_stage = m0.load_stage(m0_root, NEW_ARM, "comparison", checkpoint)
    old_stage = m0.load_stage(m0_root, CONTROL, "comparison", CONTROL_CHECKPOINT)
    if (new_stage["planned_trials"] != old_stage["planned_trials"] or len(new_stage["episodes"]) != new_stage["planned_trials"]
            or set(new_stage["dropped_layouts"]) != set(old_stage["dropped_layouts"])):
        raise ValueError("incomplete or different pairs: same planned trials and dropped layouts required")
    result = dict(arm=NEW_ARM, control_arm=CONTROL, checkpoint=checkpoint, control_checkpoint=CONTROL_CHECKPOINT,
                  core_sha256=new["core"]["sha256"], arm_sha256=new["arm"]["sha256"], control_arm_sha256=old["arm"]["sha256"],
                  **analyse_m0({e["layout_id"]: e for e in new_stage["episodes"]}, {e["layout_id"]: e for e in old_stage["episodes"]}),
                  caveat="secondary; conditional on these checkpoints and the one training seed per arm; A1 saw 0.89 epoch of its data, "
                         "A2 5.0 epochs of P2's")
    m0.write_new_json(m0_root / "arms" / NEW_ARM / f"compare_A2_step_{checkpoint:06d}_vs_{CONTROL_CHECKPOINT:06d}.json", result)
    print("COMPARE_M0_A1_VS_A2", checkpoint, f"{result['new_successes']}/{result['layouts']} vs {result['control_successes']}",
          f"b={result['discordant']['new_only']} c={result['discordant']['control_only']} p={result['mcnemar_exact_p']:.4f}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    primary = commands.add_parser("g0", help="D10 on G0's benchmark tier")
    primary.add_argument("--root", required=True, help="G0's campaign directory")
    primary.add_argument("--checkpoint", type=int, required=True)
    secondary = commands.add_parser("m0", help="secondary: M0's comparison tier")
    secondary.add_argument("--root", required=True, help="M0's campaign directory")
    secondary.add_argument("--g0-root", required=True, help="G0's campaign directory (holds A1's selection.json)")
    secondary.add_argument("--checkpoint", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "g0":
        compare(args.root, args.checkpoint)
    else:
        compare_m0(args.root, args.g0_root, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
