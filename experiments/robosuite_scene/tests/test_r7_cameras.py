import unittest

import numpy as np


def _angle_deg(a, b):
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class R7CameraTests(unittest.TestCase):
    def test_camera_poses_match_current_scene(self):
        import mujoco
        from experiments.robosuite_scene.scene import build_sim
        from src.robots.franka_panda.config import SCENE_PATH, Q_HOME

        _, new_model, new_data = build_sim()
        old_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        old_data = mujoco.MjData(old_model)
        configs = {
            "home": np.asarray(Q_HOME[:7], dtype=float),
            "bent": np.array([0.4, -0.3, 0.2, -1.8, 0.3, 1.9, 0.2]),
        }
        for label, q in configs.items():
            for model, data in ((old_model, old_data), (new_model, new_data)):
                data.qpos[:7] = q
                mujoco.mj_forward(model, data)
            for name in ("front_cam", "gripper_cam"):
                old_cam, new_cam = old_model.camera(name), new_model.camera(name)
                self.assertAlmostEqual(float(old_model.cam_fovy[old_cam.id]),
                                       float(new_model.cam_fovy[new_cam.id]), places=9)
                dpos = float(np.linalg.norm(old_data.cam_xpos[old_cam.id] - new_data.cam_xpos[new_cam.id]))
                drot = _angle_deg(old_data.cam_xmat[old_cam.id].reshape(3, 3),
                                  new_data.cam_xmat[new_cam.id].reshape(3, 3))
                print(f"{label} {name}: dpos={dpos:.6f} m drot={drot:.4f} deg")
                self.assertLess(dpos, 1e-3, f"{label} {name} position")
                self.assertLess(drot, 0.1, f"{label} {name} rotation")


if __name__ == "__main__":
    unittest.main()
