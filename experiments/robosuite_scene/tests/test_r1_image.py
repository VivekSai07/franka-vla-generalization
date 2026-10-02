import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEALED = (
    "src",
    "data_collection",
    "validation",
    "experiments/fixed_layout",
    "experiments/paired_demonstrations",
    "experiments/pose_snapshot_v2",
    "experiments/b1_language_data",
    "experiments/b2_distractor_data",
    "experiments/broad_workspace_data",
)

class R1ImageTests(unittest.TestCase):
    def test_sealed_paths_have_no_diff_against_branch_base(self):
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", "10f2524", "--", *SEALED], cwd=ROOT, text=True)
        self.assertEqual(diff.strip(), "")

    def test_contract_records_a_new_digest(self):
        from experiments.robosuite_scene.contract import IMAGE, OLD_IMAGE, MUJOCO_VERSION, ROBOSUITE_VERSION
        self.assertEqual(OLD_IMAGE, "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc")
        self.assertTrue(IMAGE.startswith("sha256:") and IMAGE != OLD_IMAGE)
        self.assertEqual(ROBOSUITE_VERSION, "1.5.2")
        self.assertEqual(MUJOCO_VERSION, "3.4.0")
        self.assertNotEqual(IMAGE, "sha256:3ecd62c5a4f220dbf674494ae8057d9db439c8ff3911981edc29aeee87ccc2c3")
        from experiments.robosuite_scene.contract import FIXED_CUBES, GRASP_Z_OFFSET_M, EPISODE_COUNT
        self.assertEqual(FIXED_CUBES, {"target_cube_red": (0.45, 0.0, 0.02), "target_cube_green": (0.35, 0.2, 0.02)})
        self.assertEqual((GRASP_Z_OFFSET_M, EPISODE_COUNT), (0.105, 5))
