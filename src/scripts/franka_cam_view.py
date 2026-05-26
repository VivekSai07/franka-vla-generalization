import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "src/robots/franka_panda/model/smolvla_scene.xml"))
data = mujoco.MjData(model)

data.qpos[:9] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
mujoco.mj_forward(model, data)

cam_list = ["gripper_cam", "front_cam", "top_cam"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
renderer = mujoco.Renderer(model, height=512, width=512)

for i, cam_name in enumerate(cam_list):
    renderer.update_scene(data, camera=cam_name)
    pixels = renderer.render()
    axes[i].imshow(pixels)
    axes[i].set_title(f"View: {cam_name}")
    axes[i].axis("off")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "cam_view_output.png")
plt.savefig(out_path)
print(f"Saved to {out_path}")
plt.show(block=True)
