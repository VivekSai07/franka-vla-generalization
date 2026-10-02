"""Stage A (admission image, CPU): export B2's own control-arm report starts for the sim2sim run.

Each trial is one of B2's sealed report trials (control condition): the scene is re-derived and
proven against the sealed bank by the B2 eval package's own functions, which only reproduce
bit-exactly on the MuJoCo version the bank was sealed with. The rollout stage runs in the robosuite
image (a different MuJoCo), so it gets plain qpos/qvel/ctrl arrays instead of re-deriving.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

CHECKPOINT = 50000
LAYOUTS = 10


def trials(layouts=LAYOUTS):
    """B2 report trials, control condition, first `layouts` report layouts, B2's own seeds."""
    from experiments.xvla_b2_distractors_eval import evaluate as ev
    keep = set(ev.report_layout_ids()[:layouts])
    return [dict(trial, seed=20260928 * 10 + trial["ordinal"]) for trial in ev.evaluation_plan("report")
            if trial["condition"] == "control" and trial["layout_id"] in keep]


def export(out, layouts=LAYOUTS):
    from experiments.xvla_b2_distractors import contract as c
    from experiments.xvla_b2_distractors_eval import evaluate as ev
    from src.robots.franka_panda.config import SCENE_PATH
    from data_collection.benchmark import restore_timing

    out = Path(out)
    scenes = out / "scenes"
    scenes.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((c.PARENT / "configuration.json").read_text())
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    restore_timing(model, manifest)
    plan, bank, picks = ev._plan(), ev._load_bank_manifest(), ev.eval_picks()
    stored = {row["ordinal"]: row for row in json.loads(
        (c.OUTPUT / ev.evaluation_dirname("report", CHECKPOINT) / "result.json").read_text())["report"]["episodes"]}

    rows, arrays = [], {}
    for trial in trials(layouts):
        lid = trial["layout_id"]
        sha, frozen = ev._materialize_scene(scenes, lid, "control", plan, bank)
        entry = ev.distractor_entry(lid, "control", sha, frozen, picks)
        data = ev.restore_distractor_scene(model, scenes, entry, ev._base()[1])
        b2 = stored[trial["ordinal"]]
        if (b2["layout_id"], b2["condition"], b2["seed"], b2["pick"]) != (lid, "control", trial["seed"], entry["pick"]):
            raise ValueError("Stored B2 report row does not match the planned trial")
        key = f"t{trial['ordinal']:03d}"
        arrays[f"{key}_qpos"], arrays[f"{key}_qvel"], arrays[f"{key}_ctrl"] = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        rows.append(dict(ordinal=trial["ordinal"], seed=trial["seed"], layout_id=lid, pick=entry["pick"],
                         place=entry["place"], arrangement=entry["arrangement"], instruction=entry["instruction"],
                         b2_task_success=b2["task_success"], b2_ticks=b2["ticks"]))
    np.savez(out / "starts.npz", **arrays)
    (out / "trials.json").write_text(json.dumps(dict(
        checkpoint=CHECKPOINT, mujoco=mujoco.__version__, camera_names=manifest["camera_names"],
        trials=rows), indent=2) + "\n")
    print("STARTS", len(rows), "trials;", sum(r["b2_task_success"] for r in rows), "succeeded in B2's own report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layouts", type=int, default=LAYOUTS)
    args = parser.parse_args()
    export(args.out, args.layouts)
