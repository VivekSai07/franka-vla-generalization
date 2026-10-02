"""The fork's run must equal pose_snapshot_v2's run except for the grasp-height name.

Parses source text only, so it runs in the admission image without robosuite.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "experiments/pose_snapshot_v2/expert.py"
FORK = ROOT / "experiments/robosuite_scene/expert.py"
ORIGINAL_GRASP_Z_OFFSET_M = 0.105


def run_function(path):
    found = [node for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(found) != 1:
        raise AssertionError(f"{path} must define exactly one run, found {len(found)}")
    return found[0]


class _Substitute(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Name(self, node):
        if node.id != "GRASP_Z_OFFSET_M":
            return node
        self.count += 1
        return ast.Constant(ORIGINAL_GRASP_Z_OFFSET_M)


class ExpertForkTests(unittest.TestCase):
    def test_run_differs_only_by_the_grasp_height_name(self):
        substitute = _Substitute()
        fork = substitute.visit(run_function(FORK))
        self.assertEqual(substitute.count, 1)
        self.assertEqual(ast.dump(fork), ast.dump(run_function(ORIGINAL)))


if __name__ == "__main__":
    unittest.main()
