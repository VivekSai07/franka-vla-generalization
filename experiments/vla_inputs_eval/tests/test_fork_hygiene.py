"""Fork hygiene for the M0 shared core: dead names, reaching into the P2 evaluation package, residue."""
import builtins
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py")
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__", "__debug__"}
DELIBERATE = {"from experiments.xvla_props_p2_eval import analysis"}
FORKED_FROM = "xvla_props_p2_eval"
RESIDUE = ("report_p2", "xvla_props_p2.contract")  # the sealed root path and the plan's evaluation_holdout key are read on purpose


class UndefinedNames(unittest.TestCase):
    def test_no_function_reads_a_name_that_is_never_bound(self):
        offenders = []
        for name in MODULES:
            top = symtable.symtable((PACKAGE / name).read_text(), name, "exec")
            known = set(top.get_identifiers()) | DUNDERS

            def walk(table):
                for child in table.get_children():
                    if child.get_type() == "function":
                        for sym in child.get_symbols():
                            ident = sym.get_name()
                            if sym.is_global() and not sym.is_assigned() and ident not in known and not hasattr(builtins, ident):
                                offenders.append(f"{name}:{child.get_name()}() reads undefined {ident!r}")
                    walk(child)

            walk(top)
        self.assertEqual(offenders, [], "; ".join(offenders))


class Residue(unittest.TestCase):
    def test_the_core_reaches_into_the_p2_evaluation_package_only_for_its_classifier(self):
        for name in MODULES:
            for line in (PACKAGE / name).read_text().splitlines():
                stripped = line.strip()
                if FORKED_FROM in stripped and stripped.startswith(("from ", "import ")):
                    self.assertIn(stripped, DELIBERATE, f"{name}: {stripped}")

    def test_the_tests_do_not_import_the_p2_evaluation_loop(self):
        allowed = {"from experiments.xvla_props_p2_eval import scenes as p2_scenes"}
        for path in (PACKAGE / "tests").glob("test_*.py"):
            if path.name == "test_fork_hygiene.py":
                continue
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if FORKED_FROM in stripped and stripped.startswith(("from ", "import ")):
                    self.assertTrue(stripped in allowed, f"{path.name}: {stripped}")

    def test_no_p2_evaluation_residue_in_the_core(self):
        for name in MODULES:
            text = (PACKAGE / name).read_text()
            for residue in RESIDUE:
                self.assertNotIn(residue, text, f"{name} still mentions {residue}")

    def test_the_core_imports_no_arm_package(self):
        for name in MODULES:
            for line in (PACKAGE / name).read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("from ", "import ")):
                    self.assertNotIn("vla_inputs_arm_", stripped, f"{name}: {stripped}")


if __name__ == "__main__":
    unittest.main()
