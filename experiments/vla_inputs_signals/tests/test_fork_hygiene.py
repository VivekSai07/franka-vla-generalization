"""Static checks on the S0 package: dead names, reaching into frozen P2 code, source closure, README present."""
import builtins
import importlib
import shutil
import symtable
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.vla_inputs_signals import contract

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ("contract.py", "replay.py", "signals.py", "sidecar.py")
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__", "__debug__"}
FROZEN = ("experiments.props_p2_data", "experiments.props_p1_data", "experiments.b3_robosuite_data")


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


class FrozenCode(unittest.TestCase):
    def test_only_replay_reaches_into_frozen_packages(self):
        for name in ("contract.py", "signals.py", "sidecar.py"):
            source = (PACKAGE / name).read_text()
            for package in FROZEN:
                self.assertNotIn(package, source, f"{name} reaches into {package}")

    def test_every_repository_module_the_package_loads_is_hash_bound(self):
        for name in ("replay", "signals", "sidecar", "contract"):
            importlib.import_module(f"experiments.vla_inputs_signals.{name}")
        bound = set(contract.provenance())
        loaded = set()
        for module in list(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_absolute() and Path(path).resolve().is_relative_to(ROOT):
                loaded.add(Path(path).resolve().relative_to(ROOT).as_posix())
        unbound = sorted(p for p in loaded if p not in bound and "/tests/" not in p and not p.endswith("__init__.py"))
        self.assertEqual(unbound, [], f"loaded but not in provenance(): {unbound[:8]}")

    def test_p2_files_are_unchanged_since_its_seal(self):
        contract.check_p2_frozen()

    def test_p2_seal_is_the_one_this_binds_to(self):
        import json
        self.assertEqual(json.loads((contract.P2_ROOT / "seal.json").read_text())["fingerprint"], contract.P2_SEAL)


class ProvenanceSensitivity(unittest.TestCase):
    def test_one_changed_byte_in_any_kind_of_bound_file_changes_provenance(self):
        base = contract.provenance()
        with tempfile.TemporaryDirectory() as tmp:
            for rel in base:
                (Path(tmp) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, Path(tmp) / rel)
            self.assertEqual(contract.provenance(tmp), base)
            for rel in ("experiments/vla_inputs_signals/README.md", "experiments/vla_inputs_signals/signals.py",
                        "experiments/props_scene/scene.py", "data_collection/pilot_plan.py", contract.EXTRA_FILES[3]):
                path = Path(tmp) / rel
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.assertNotEqual(contract.provenance(tmp)[rel], base[rel], rel)
                path.write_bytes(original)

    def test_a_new_file_in_the_package_is_picked_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in contract.provenance():
                (Path(tmp) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, Path(tmp) / rel)
            (Path(tmp) / "experiments/vla_inputs_signals/extra.py").write_text("x = 1\n")
            self.assertIn("experiments/vla_inputs_signals/extra.py", contract.provenance(tmp))


class Readme(unittest.TestCase):
    def test_readme_exists_is_not_empty_and_is_bound(self):
        readme = PACKAGE / "README.md"
        self.assertGreater(readme.stat().st_size, 500)
        self.assertIn("experiments/vla_inputs_signals/README.md", contract.provenance())

    def test_provenance_covers_every_py_file_of_the_package(self):
        bound = set(contract.provenance())
        for path in PACKAGE.rglob("*.py"):
            if "__pycache__" not in path.parts:
                self.assertIn(path.relative_to(ROOT).as_posix(), bound)

    def test_source_files_are_the_package_files(self):
        self.assertEqual(set(contract.SOURCE_FILES), set(contract.source_paths()))


if __name__ == "__main__":
    unittest.main()
