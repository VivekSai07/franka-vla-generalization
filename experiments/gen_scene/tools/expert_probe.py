"""Expert reach evidence for G0's geometry levels (Task 3). Robosuite image, CPU, unrendered. Not part of any protocol.

For each spatial level and each distractor level, the scripted expert (`GenExpert`) attempts the same 48 fresh layouts (one per M0
comparison cell, identities `gen:probe:...`, a namespace used by no plan), plus the nominal condition and the fresh episode from a
`move` destination. This tells whether a level is solvable at all before any policy sees it; it is the evidence for the outer-start
region and the distractor region, which are pinned in `levels.py` and tested. Nothing is tuned here except by fixing a demonstrated
expert bug, which is then re-gated. Sharded: `--shard k --shards n --out probe_k.jsonl`; `--summary files...` prints the rates.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

JOBS = [("nominal", None)] + [("spatial", l) for l in ("table_up5", "table_down5", "table_up10", "outer_start")] + \
       [("distractors", l) for l in ("three", "four", "five")] + [("move", None)]


def layouts():
    """48 layouts, one per M0 comparison cell."""
    from experiments.props_p2_data import plan as p2
    from experiments.vla_inputs_eval import plan as m0
    seeds, out = {}, []
    for cell in m0.COMPARISON_CELLS:
        obj, bin_colour, slot, count = cell
        identity = f"gen:probe:{m0.cell_name(cell)}:0"
        deck = m0._deck(obj, count, 1, f"{identity}:deck", seeds)
        draw = p2._draw_layout(obj, deck[0], identity, seeds)
        out.append(dict(layout_id=m0.layout_id(identity), identity=identity, object=obj, bin=bin_colour, blue_slot=slot, cell=list(cell),
                        home_pose_jitter_deg=p2._jitter(identity, seeds), seed=p2.seed(f"{identity}:expert"), **draw))
    return out


def scene_for(layout, kind, level, j):
    from experiments.gen_scene import levels
    from experiments.gen_scene.conditions import NOMINAL
    if kind == "nominal":
        return layout, NOMINAL
    if kind == "spatial":
        return layout, levels.spatial(level, layout, layout["identity"], j)
    if kind == "distractors":
        return layout, levels.distractors(level, layout, layout["identity"])
    x, y = levels.disturbance("move", layout, layout["identity"]).move_to
    return dict(layout, start=[x, y, layout["start"][2]]), NOMINAL


def attempt(layout, condition):
    from experiments.gen_scene import compose
    from experiments.gen_scene.expert import GenExpert
    model, resolved = compose.model_for(layout, condition)
    expert = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    episode = dict(object=layout["object"], bin=layout["bin"], seed=layout["seed"], home_pose_jitter_deg=layout["home_pose_jitter_deg"])
    return expert.run(scene["initial_state"], episode, lambda frame, trace: None, render=False)


def shard(index, shards, out):
    work = [(k, (i, kind, level)) for k, (i, kind, level) in enumerate((i, kind, level) for kind, level in JOBS for i in range(48))]
    rows = []
    ls = layouts()
    with open(out, "w") as stream:
        for k, (i, kind, level) in work:
            if k % shards != index:
                continue
            layout, condition = scene_for(ls[i], kind, level, i)
            result = attempt(layout, condition)
            row = dict(kind=kind, level=level or kind, layout=ls[i]["layout_id"], object=ls[i]["object"], success=bool(result["success"]),
                       reason=result["reason"], start=list(layout["start"]), bin_shift=condition.bin_shift_m)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print("PROBE", row["kind"], row["level"], row["layout"], row["success"], row["reason"], flush=True)


def summary(paths):
    rows = [json.loads(l) for p in paths for l in Path(p).read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["level"])].append(r)
    for key, group in by.items():
        wins = sum(r["success"] for r in group)
        print(f"{key[0]:12s} {key[1]:12s} {wins:3d}/{len(group)}  {dict(Counter(r['reason'] for r in group if not r['success']))}")
    return by


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--summary", nargs="+")
    args = parser.parse_args(argv)
    if args.summary:
        summary(args.summary)
    else:
        shard(args.shard, args.shards, args.out)


if __name__ == "__main__":
    sys.exit(main())
