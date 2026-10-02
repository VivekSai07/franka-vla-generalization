"""Frame extraction and replay checks. Must import without robosuite installed."""
import mujoco
import numpy as np

from data_collection.benchmark import synchronize_observation
from experiments.paired_demonstrations.media import CAMERAS, FEATURES


def frame_from_sim(model, data, action, task):
    """One LeRobot-schema frame from the simulator's current state."""
    action = np.asarray(action, dtype=np.float64)
    expected = FEATURES["action"]["shape"]
    if action.shape != expected or not np.isfinite(action).all():
        raise ValueError(f"action must be finite with shape {expected}, got {action.shape}")
    synchronize_observation(model, data)
    frame = {"observation.state": np.asarray(data.qpos[:9], dtype=np.float32).copy(),
             "action": action.astype(np.float32),
             "task": task}
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        for key, camera in CAMERAS.items():
            renderer.update_scene(data, camera=camera)
            frame[key] = renderer.render().copy()
    return frame
