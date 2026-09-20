"""Freeze independent tuning/final initial states without success filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.benchmark import capture_state, configure_timing, source_provenance, runtime_versions, synchronize_observation, write_json
from data_collection.pilot_plan import validate_plan, plan_digest, slot_seed
from data_collection.pilot_integrity import scene_digest, validate_scene_geometry
from data_collection.collect_libero_dataset import randomize_cubes
from src.robots.franka_panda.config import SCENE_PATH, Q_HOME


def generate(plan, output):
    validate_plan(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    timing = configure_timing(model, 30)
    manifest = {"status": "incomplete", "plan_digest": plan_digest(plan), "timing": timing,
                "source_hashes": source_provenance(ROOT), "versions": runtime_versions(), "scenes": []}
    write_json(output / "manifest.json", manifest)
    seen = set()
    for slot in plan["scene_slots"]:
        seed = slot_seed(slot, 0)
        random.seed(seed); np.random.seed(seed)
        data = mujoco.MjData(model)
        randomize_cubes(model, data, slot["colors"])
        data.qpos[:len(Q_HOME)] = Q_HOME
        data.qvel[:] = 0
        for _ in range(round(.2 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        synchronize_observation(model, data)
        validate_scene_geometry(model, data, slot["colors"])
        geometry_hash = scene_digest(data.qpos)
        if geometry_hash in seen:
            raise ValueError("Duplicate evaluation initialization")
        seen.add(geometry_hash)
        name = slot["slot_id"].replace("/", "_") + ".npz"
        path = output / name
        np.savez_compressed(path, initial_state=capture_state(model, data), qpos=data.qpos.copy())
        manifest["scenes"].append({**slot, "seed": seed, "file": name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "scene_hash": geometry_hash})
    if manifest["source_hashes"] != source_provenance(ROOT):
        raise ValueError("Source changed during scene generation")
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(f"Scene bank complete: {len(manifest['scenes'])} scenes; no policy/expert success filtering")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(json.loads(args.plan.read_text()), args.output)