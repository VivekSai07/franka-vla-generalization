"""Fork hygiene for the G0 shared core: dead names, residue of the sources, process ids, what it imports."""
import builtins
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py")
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__", "__debug__"}
# M0 modules the core imports, read-only: the adapter interface and M0's plan samplers (bound by EXTRA_CORE).
ALLOWED_M0 = {"from experiments.vla_inputs_eval import adapters", "from experiments.vla_inputs_eval import plan as m0",
              "from experiments.vla_inputs_eval import adapters, protocol as m0_protocol", "from experiments.vla_inputs_eval import evaluate as m0_evaluate",
              "from experiments.vla_inputs_eval import protocol as m0_protocol", "from experiments.vla_inputs_pilot_plan import plan as pilot"}
RESIDUE = ("m0_template", "COMPARISON_PER_CELL", "phase_checkpoint", "CONTROL =")


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


class Residue(unittest.TestCase):
    def test_no_process_id_names_a_file_or_a_log(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            self.assertNotIn("getpid", text, f"{name} keys something by the process id (every container worker is pid 1; FAILURES #71)")
            self.assertNotIn("os.getppid", text)

    def test_the_core_imports_m0_only_through_the_declared_read_only_doors(self):
        for name in MODULES:
            for line in code_lines(name):
                if line.startswith(("from ", "import ")) and "vla_inputs_eval" in line:
                    self.assertIn(line, ALLOWED_M0, f"{name}: {line}")

    def test_the_core_imports_no_arm_package_and_no_p2_evaluation_loop(self):
        for name in MODULES:
            for line in code_lines(name):
                if line.startswith(("from ", "import ")):
                    self.assertNotIn("vla_inputs_arm_", line, name)
                    self.assertNotIn("xvla_props_p2_eval import rollout", line, name)
                    self.assertNotIn("xvla_props_p2_eval import scenes", line, name)

    def test_the_rollout_does_not_use_props_scenes_absolute_placement(self):
        text = "\n".join(code_lines("rollout.py"))
        self.assertIn("from experiments.gen_scene.metrics import placement", text)
        self.assertNotIn("import in_bin, placement", text)

    def test_the_rollout_runs_the_engines_scene_not_props_scenes_builder(self):
        for name in ("rollout.py", "scenes.py", "evaluate.py", "feasibility.py"):
            self.assertNotIn("props_scene.build_model", "\n".join(code_lines(name)).replace("compose.build_model", ""))
            self.assertNotIn("from experiments.props_scene.scene import build_model", "\n".join(code_lines(name)))

    def test_no_m0_only_vocabulary_survives_the_fork(self):
        for name in MODULES:
            text = "\n".join(code_lines(name))
            for residue in RESIDUE:
                self.assertNotIn(residue, text, f"{name} still mentions {residue}")


if __name__ == "__main__":
    unittest.main()
