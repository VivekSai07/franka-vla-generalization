import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Load model and data
model = mujoco.MjModel.from_xml_path('src/robots/franka_panda/model/smolvla_scene.xml')
data = mujoco.MjData(model)

# Set to Home Position from config.py
# Order: 7 arm joints + 2 finger joints
data.qpos[:9] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
mujoco.mj_forward(model, data)

# List of cameras to verify
cam_list = ["gripper_cam", "front_cam", "top_cam"]

# Setup visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
renderer = mujoco.Renderer(model, height=480, width=640)

for i, cam_name in enumerate(cam_list):
    renderer.update_scene(data, camera=cam_name)
    pixels = renderer.render()
    
    axes[i].imshow(pixels)
    axes[i].set_title(f"View: {cam_name}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(r'C:\Users\Vivek Sai\Downloads\Mujoco\Smolvla\src\scripts\cam_view_output.png')
plt.show(block=True)