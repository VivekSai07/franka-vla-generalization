"""Fork hygiene of the G1 collector: dead names, and the fork must run its own code, not P2's."""
import ast
import builtins
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py")
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__", "__debug__"}


def code_lines(name):
    """The module's code as lines, without comments and docstrings (prose may name the package it was forked from)."""
    tree = ast.parse((PACKAGE / name).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).splitlines()


def undefined_names(source, name="x"):
    top = symtable.symtable(source, name, "exec")
    known = set(top.get_identifiers()) | DUNDERS
    offenders = []

    def walk(table):
        for child in table.get_children():
            if child.get_type() == "function":
                for sym in child.get_symbols():
                    ident = sym.get_name()
                    if sym.is_global() and not sym.is_assigned() and ident not in known and not hasattr(builtins, ident):
                        offenders.append(f"{name}:{child.get_name()}() reads undefined {ident!r}")
            walk(child)

    walk(top)
    return offenders


class UndefinedNames(unittest.TestCase):
    def test_no_function_reads_a_name_that_is_never_bound(self):
        offenders = [o for name in MODULES for o in undefined_names((PACKAGE / name).read_text(), name)]
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_the_scan_sees_a_dead_name(self):
        self.assertEqual(len(undefined_names("def f():\n    return missing_name\n")), 1)
        self.assertEqual(undefined_names("def f():\n    return len([])\n"), [])


class RunsItsOwnCode(unittest.TestCase):
    def test_only_the_plan_and_the_contract_import_the_sealed_p2_package(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            if name in ("plan.py", "contract.py"):
                continue
            self.assertNotIn("props_p2_data", text, name)

    def test_nothing_is_still_named_p2(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            for residue in ("props_p2_shard", "P2 shard", "P2 SHARD", "P2 MERGE", "P2 SEALED", "P2 GATE", "P2 DETERMINISM", "P2Expert",
                            "props_p2_data.episode", "props_p2_data.record", "props_p2_data.validate", "props_p2_data.merge",
                            "episode_{episode['episode_index']:03d}"):
                self.assertNotIn(residue, text, f"{name}: {residue}")

    def test_the_expert_and_the_scene_are_the_engines(self):
        text = "\n".join(code_lines("episode.py"))
        self.assertIn("from experiments.gen_scene.expert import GenExpert", text)
        self.assertIn("from experiments.gen_scene import compose", text)
        self.assertNotIn("props_scene.scene import", text)
        self.assertNotIn("P2Expert", text)

    def test_the_signals_come_from_s0_and_are_not_re_implemented(self):
        text = "\n".join(code_lines("episode.py"))
        self.assertIn("from experiments.vla_inputs_signals import signals", text)
        for call in ("signals.wrist_wrench(", "signals.finger_force(", "signals.props_box("):
            self.assertIn(call, text)
        for forbidden in ("def wrist_wrench", "def finger_force", "def props_box", "mj_rnePostConstraint", "mj_contactForce"):
            self.assertNotIn(forbidden, text)

    def test_the_collector_never_names_the_test_pool(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            for forbidden in ('pool="test"', "pool='test'", '["test"]', "['test']", 'PRIMITIVE_SHAPES["B"]', "PRIMITIVE_COLOURS[\"B\"]"):
                self.assertNotIn(forbidden, text, f"{name}: {forbidden}")

    def test_the_collector_imports_no_evaluator_or_arm_package(self):
        for name in MODULES:
            for line in code_lines(name):
                if line.startswith(("from ", "import ")):
                    self.assertNotIn("vla_inputs_arm_", line, name)
                    self.assertNotIn("xvla_", line, name)
                    if name != "plan.py":
                        self.assertNotIn("gen_eval", line, name)


if __name__ == "__main__":
    unittest.main()
