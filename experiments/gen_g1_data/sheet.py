"""Contact sheets of rendered gate attempts, for the user's approval of the randomization (D4). Robosuite image, CPU.

    --render <dir> --worker k --workers n    render this worker's share of `sheet_indices()` and save both cameras at two ticks per attempt
    --assemble <dir>                         tile the saved frames into `sheet_scene.png` (tick 0) and `sheet_carry.png` (mid-carry)

A cell is one attempt: the front and the gripper camera side by side, labelled with its slot, object, distractor count, height stratum
and the gate's verdict. Nothing here feeds the dataset; it reads the plan and runs the same attempt the gate runs, rendered.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import plan as p

PER_OBJECT = 4
COLUMNS = 3
CELL = 256
LABEL_HEIGHT = 22
KEYS = ("observation.images.image", "observation.images.image2")


def sheet_indices(plan=None, per_object=PER_OBJECT):
    """The first `per_object` gate attempts of every object, in gate order: 24 attempts covering all six objects."""
    attempts = (plan or p.make_plan())["gate_attempts"]
    chosen = []
    for obj in p.OBJECTS:
        chosen += [a["episode_index"] for a in attempts if a["object"] == obj][:per_object]
    return sorted(chosen)


def carry_tick(phase_index):
    """The middle tick of the `move` phase, or the last tick when the attempt never got that far."""
    from experiments.props_scene import constants as c
    ticks = [t for t, phase in enumerate(phase_index) if c.PHASE_NAMES[int(phase)] == "move"]
    return ticks[len(ticks) // 2] if ticks else len(phase_index) - 1


def tile(cells, columns=COLUMNS):
    """Row-major grid of equally sized RGB cells; the last row is padded with black."""
    if not cells:
        raise ValueError("no cells")
    height, width, _ = cells[0].shape
    if any(c.shape != cells[0].shape for c in cells):
        raise ValueError("cells differ in size")
    rows = -(-len(cells) // columns)
    sheet = np.zeros((rows * height, columns * width, 3), np.uint8)
    for i, cell in enumerate(cells):
        r, k = divmod(i, columns)
        sheet[r * height:(r + 1) * height, k * width:(k + 1) * width] = cell
    return sheet


def label(attempt, verdict):
    return (f"{attempt['episode_index']:02d} {attempt['object']} n={attempt['distractor_count']} "
            f"h={attempt['condition']['table_offset_m'] * 100:+.0f}cm {verdict}")


def _resize(image, size):
    import cv2
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def make_cell(front, gripper, text):
    """Both cameras at CELL px with a label strip on top."""
    import cv2
    row = np.concatenate([_resize(front, CELL), _resize(gripper, CELL)], axis=1)
    strip = np.zeros((LABEL_HEIGHT, row.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (4, LABEL_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([strip, row], axis=0)


def render_worker(out, worker, workers):
    from experiments.gen_g1_data.record import _require_pinned_runtime, make_execute
    _require_pinned_runtime()
    from experiments.gen_g1_data.validate import CAMERA_KEYS
    plan = p.make_plan()
    attempts = {a["episode_index"]: a for a in plan["gate_attempts"]}
    execute = make_execute(render=True)
    out = Path(out)
    for index in sheet_indices(plan)[worker::workers]:
        attempt = attempts[index]
        result, frames, traces, verdict = execute(attempt, p.attempt_draw(attempt, 0))
        phases = [int(t["phase_index"]) for t in traces]
        saved = {}
        for name, tick in (("scene", 0), ("carry", carry_tick(phases))):
            for k, key in enumerate(CAMERA_KEYS):
                saved[f"{name}_{k}"] = frames[tick][key]
        np.savez(out / f"cell_{index:03d}.npz", **saved)
        (out / f"cell_{index:03d}.json").write_text(json.dumps(
            {"index": index, "label": label(attempt, "OK" if verdict["admitted"] else verdict["reason"]),
             "admitted": bool(verdict["admitted"]), "reason": verdict["reason"], "expert_reason": result["reason"]}))
        print(f"SHEET cell {index} {verdict['reason']}", flush=True)


def assemble(out):
    import cv2
    out = Path(out)
    indices = sheet_indices()
    missing = [i for i in indices if not (out / f"cell_{i:03d}.npz").exists()]
    if missing:
        raise SystemExit(f"cells missing: {missing}")
    for name in ("scene", "carry"):
        cells = []
        for index in indices:
            info = json.loads((out / f"cell_{index:03d}.json").read_text())
            with np.load(out / f"cell_{index:03d}.npz") as stored:
                cells.append(make_cell(stored[f"{name}_0"], stored[f"{name}_1"], info["label"]))
        cv2.imwrite(str(out / f"sheet_{name}.png"), tile(cells)[:, :, ::-1])
    print(f"SHEET assembled {len(indices)} cells", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", metavar="DIR")
    mode.add_argument("--assemble", metavar="DIR")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.render:
        render_worker(args.render, args.worker, args.workers)
    else:
        assemble(args.assemble)


if __name__ == "__main__":
    main()
