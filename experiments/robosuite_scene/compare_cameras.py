"""Render both cameras of the current scene and the robosuite scene at Q_HOME for visual comparison."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from experiments.robosuite_scene.scene import build_sim, overwrite_reset
from src.robots.franka_panda.config import SCENE_PATH, Q_HOME

CAMERAS = ("front_cam", "gripper_cam")


def _render(model, data, size=512):
    # mujoco.Renderer only; finish one model before opening the next (OSMesa contexts conflict).
    images = {}
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        for name in CAMERAS:
            renderer.update_scene(data, camera=name)
            image = renderer.render().copy()
            if not image.std() > 0:
                raise RuntimeError(f"blank render for {name}")
            images[name] = image
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    old_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    old_data = mujoco.MjData(old_model)
    old_data.qpos[:9] = Q_HOME  # arm and open fingers, as overwrite_reset sets on the robosuite side
    mujoco.mj_forward(old_model, old_data)
    current = _render(old_model, old_data)
    del old_model, old_data

    _, model, data = build_sim()
    overwrite_reset(model, data)  # arm at Q_HOME, cubes at FIXED_CUBES
    robosuite = _render(model, data)

    for tag, images in (("current", current), ("robosuite", robosuite)):
        for name in CAMERAS:
            Image.fromarray(images[name]).save(out / f"{name}_{tag}.png")
    Image.fromarray(np.hstack([current["front_cam"], robosuite["front_cam"],
                               current["gripper_cam"], robosuite["gripper_cam"]])).save(out / "side_by_side.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
