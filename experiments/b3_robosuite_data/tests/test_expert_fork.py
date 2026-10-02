"""B3Expert.run must equal the B2 expert's run except for the grasp-height name.

Parses source text only, so it runs in the admission image without robosuite.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "experiments/b2_distractor_data/expert.py"
FORK = ROOT / "experiments/b3_robosuite_data/expert.py"
ORIGINAL_GRASP_Z_OFFSET_M = 0.105
SHARED_HELPERS = {"_apply_hover_offset", "_validate_home_jitter", "_validate_hover_offset"}


def run_function(path, class_name):
    classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise AssertionError(f"{path} must define exactly one class {class_name}")
    found = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(found) != 1:
        raise AssertionError(f"{class_name} must define exactly one run, found {len(found)}")
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
        fork = substitute.visit(run_function(FORK, "B3Expert"))
        self.assertEqual(substitute.count, 1)
        self.assertEqual(ast.dump(fork), ast.dump(run_function(ORIGINAL, "Expert")))

    def test_helpers_are_imported_not_redefined(self):
        tree = ast.parse(FORK.read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module == "experiments.b2_distractor_data.expert" for a in n.names}
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertEqual(imported, SHARED_HELPERS)
        self.assertFalse(defined & SHARED_HELPERS)

    def test_no_residue_of_the_source_package_beyond_the_helpers(self):
        text = FORK.read_text()
        self.assertEqual(text.count("b2_distractor_data"), 2)  # the import line and the docstring


if __name__ == "__main__":
    unittest.main()
