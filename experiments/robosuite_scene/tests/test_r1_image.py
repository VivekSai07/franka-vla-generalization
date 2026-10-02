import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEALED = (
    "src/robots/franka_panda/model",
    "experiments/fixed_layout/source.py",
    "experiments/fixed_layout/validate.py",
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
        self.assertEqual(MUJOCO_VERSION, "3.13.0")
