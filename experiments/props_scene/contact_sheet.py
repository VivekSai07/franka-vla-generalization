"""Contact sheets: both cameras at the start of every phase, for chosen gate attempts, to look at by eye.

    python -m experiments.props_scene.contact_sheet --out /out/contact_sheets \\
        --attempts grid:can:blue:left:3 grid:cereal:orange:right:2 ...
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.props_scene import constants as c
from experiments.props_scene.benchmark import build_plan
from experiments.props_scene.expert import PropsExpert
from experiments.props_scene.scene import reset

CELL = 192
KEYS = ("observation.images.image", "observation.images.image2")


def frames_for(attempt):
    expert = PropsExpert((attempt["object"],), attempt["blue_slot"])
    scene = reset(expert.model, {attempt["object"]: tuple(attempt["pose"])}, attempt["home_pose_jitter_deg"])
    picked, seen = {}, set()

    def sink(frame, trace):
        phase = trace["phase_index"]
        if phase not in seen:
            seen.add(phase)
            picked[phase] = {key: frame[key].copy() for key in KEYS}
        picked["last"] = {key: frame[key].copy() for key in KEYS}

    episode = {"object": attempt["object"], "bin": attempt["bin"], "seed": attempt["seed"],
               "home_pose_jitter_deg": attempt["home_pose_jitter_deg"]}
    result = expert.run(scene["initial_state"], episode, sink, render=True)
    return picked, result


def sheet(attempts, path):
    columns = list(range(len(c.PHASE_NAMES))) + ["last"]
    labels = list(c.PHASE_NAMES) + ["end"]
    rows = []
    for attempt in attempts:
        picked, result = frames_for(attempt)
        title = (f"{attempt['id']}  start ({attempt['pose'][0]:.2f}, {attempt['pose'][1]:.2f}, {attempt['pose'][2]:.0f} deg)"
                 f"  -> {result['reason']}")
        for key in KEYS:
            row = Image.new("RGB", (CELL * len(columns), CELL + 14), "white")
            draw = ImageDraw.Draw(row)
            for i, column in enumerate(columns):
                if column in picked:
                    tile = Image.fromarray(picked[column][key]).resize((CELL, CELL))
                    row.paste(tile, (i * CELL, 14))
                draw.text((i * CELL + 2, 1), labels[i] if key == KEYS[0] else "", fill="black")
            if key == KEYS[0]:
                draw.text((CELL * 3, 1), title, fill="red")
            rows.append(row)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    print("wrote", path, out.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="sheet")
    parser.add_argument("--attempts", nargs="+", required=True)
    args = parser.parse_args()
    plan = {a["id"]: a for a in build_plan()}
    sheet([plan[i] for i in args.attempts], args.out / f"{args.name}.png")


if __name__ == "__main__":
    main()
