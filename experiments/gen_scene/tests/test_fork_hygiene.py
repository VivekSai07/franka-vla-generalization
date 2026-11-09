"""Fork hygiene of the randomization engine: dead names, and the fork must run its own code, not the source's."""
import builtins
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py")
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__", "__debug__"}


def code_lines(name):
    return [l.strip() for l in (PACKAGE / name).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


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

    def test_the_scan_sees_a_dead_name(self):
        source = "def f():\n    return missing_name\n"
        top = symtable.symtable(source, "x", "exec")
        function = top.get_children()[0]
        undefined = [s.get_name() for s in function.get_symbols()
                     if s.is_global() and not s.is_assigned() and s.get_name() not in top.get_identifiers() and not hasattr(builtins, s.get_name())]
        self.assertEqual(undefined, ["missing_name"])


class RunsItsOwnCode(unittest.TestCase):
    def test_the_expert_is_a_fork_not_a_subclass_of_the_sealed_ones(self):
        text = "\n".join(code_lines("expert.py"))
        for forbidden in ("import PropsExpert", "import P2Expert", "(PropsExpert)", "(P2Expert)"):
            self.assertNotIn(forbidden, text)
        self.assertIn("from experiments.gen_scene.metrics import placement", text)
        self.assertNotIn("props_scene.metrics import in_bin, placement", text)

    def test_the_expert_reads_the_table_from_the_model(self):
        text = "\n".join(code_lines("expert.py"))
        self.assertIn("compose.table_top_z(model)", text)
        self.assertNotIn("props_scene.scene import build_model", text)

    def test_only_the_gates_and_compose_touch_props_scenes_model_builder(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            if name in ("compose.py", "gates.py"):
                continue
            self.assertNotIn("build_model", text.replace("compose.build_model", "").replace("def build_model", ""), name)

    def test_gen_scene_imports_no_eval_or_arm_package(self):
        for name in MODULES:
            for line in code_lines(name):
                if line.startswith(("from ", "import ")):
                    self.assertNotIn("vla_inputs_arm_", line, name)
                    self.assertNotIn("gen_eval", line, name)


if __name__ == "__main__":
    unittest.main()
