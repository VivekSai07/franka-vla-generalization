"""Contact sheet of every G0 level, for the human approval gate. Robosuite image, CPU, rendered.

One row per axis level, drawn from the first non-lemon benchmark layout the plan gives that level. Columns: both cameras at tick 0, and
both at mid-episode of the scripted expert's run under that condition. `drop` is the nominal scene (the object is released 5 cm
after lift at rollout); `move` shows the original start and the fresh episode from the destination. Not part of any protocol.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.gen_eval import feasibility
from experiments.gen_scene import compose
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.gen_scene.expert import GenExpert


def rows_of(plan):
    """(label, layout, job-scene) per level, first non-lemon benchmark layout that has it."""
    chosen = {}
    for layout in plan["tiers"]["benchmark"]:
        layout = plan["layouts"][layout] if isinstance(layout, str) else layout
        if layout["object"] == "lemon":
            continue
        for axis, entry in layout["axes"].items():
            chosen.setdefault((axis, entry["level"]), layout)
    chosen.setdefault(("nominal", "nominal"), next(l for l in plan["layouts"].values() if l["object"] != "lemon"))
    return sorted(chosen.items())


def frames(layout, condition):
    """Tick-0 and mid-episode images of the expert's run."""
    model, resolved = compose.model_for(layout, condition)
    expert = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    episode = dict(object=layout["object"], bin=layout["bin"], seed=layout["seed"],
                   home_pose_jitter_deg=list(layout["home_pose_jitter_deg"]))
    kept = []
    outcome = expert.run(scene["initial_state"], episode, lambda frame, trace: kept.append(
        {k: v for k, v in frame.items() if getattr(v, "ndim", 0) == 3}), render=True)
    return kept[0], kept[len(kept) // 2], outcome["success"]


def build(job):
    (axis, level), layout, out = job
    if (axis, level) == ("nominal", "nominal") or axis == "disturbance" and level == "drop":
        first, mid, ok = frames(layout, NOMINAL)
    elif axis == "disturbance":
        first, _, ok = frames(layout, NOMINAL)
        moved, condition = feasibility.job_scene(layout, "move")
        mid, _, ok_moved = frames(moved, condition)
        ok = ok and ok_moved
    else:
        first, mid, ok = frames(layout, Condition.from_dict(layout["axes"][axis]["condition"]))
    keys = sorted(first)
    row = np.concatenate([first[k] for k in keys] + [mid[k] for k in keys], axis=1)
    label = f"{axis}:{level}  {layout['object']}  {layout['layout_id'][-24:]}  expert={'ok' if ok else 'FAIL'}"
    image = Image.fromarray(row)
    ImageDraw.Draw(image).text((4, 4), label, fill=(255, 255, 0))
    path = Path(out) / f"{axis}_{level}.png"
    image.save(path)
    return axis, level, str(path), keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    plan = feasibility.load_plan(args.plan)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(key, layout, args.out) for key, layout in rows_of(plan)]
    with Pool(args.workers) as pool:
        results = pool.map(build, jobs)
    paths = [r[2] for r in results]
    sheet = np.concatenate([np.asarray(Image.open(p)) for p in paths], axis=0)
    Image.fromarray(sheet).save(Path(args.out) / "sheet.png")
    print("SHEET", len(paths), "rows", sheet.shape, "columns", results[0][3], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
