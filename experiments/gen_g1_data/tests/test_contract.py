import json
import sys
import unittest
from pathlib import Path

from experiments.gen_g1_data import contract, plan as p

PACKAGE = Path(contract.__file__).resolve().parent
SOURCE_PACKAGE = "props_" + "p3_data"   # split so this file does not match its own scan
SIBLING = "props_" + "p2_data"
P2_PROVENANCE = Path(contract.ROOT) / "outputs/props_p2_data/props_p2_20261013/provenance.json"


class ContractTests(unittest.TestCase):
    def test_provenance_binds_the_package_the_scene_the_engine_and_the_signals(self):
        prov = contract.provenance()
        for required in ("experiments/gen_g1_data/plan.py", "experiments/gen_g1_data/README.md",
                         "experiments/gen_g1_data/tests/test_plan.py", "experiments/gen_g1_data/contract.py",
                         "experiments/props_scene/expert.py", "experiments/props_scene/scene.py",
                         "experiments/robosuite_scene/xml_contract.py", "experiments/gen_scene/compose.py",
                         "experiments/gen_scene/expert.py", "experiments/gen_scene/levels.py", "experiments/gen_scene/pools.py",
                         "experiments/gen_scene/conditions.py", "experiments/gen_scene/assets_manifest.json",
                         "experiments/vla_inputs_signals/signals.py", "experiments/props_p2_data/plan.py",
                         "experiments/props_p2_data/README.md", "experiments/paired_demonstrations/media.py",
                         "data_collection/collect_libero_dataset.py", "src/robots/franka_panda/config.py"):
            self.assertIn(required, prov)
        self.assertEqual(prov, contract.provenance())

    def test_provenance_does_not_bind_the_live_design_documents(self):
        for doc in (p.SPEC, p.PLAN_DOC):
            self.assertNotIn(doc, contract.provenance())
        self.assertFalse(any(path.startswith("docs/") for path in contract.provenance()))

    def test_provenance_covers_everything_p2s_seal_covered_but_its_docs(self):
        sealed = set(json.loads(P2_PROVENANCE.read_text()))   # needs the sealed P2 data mounted; fails, never skips, without
        missing = sealed - set(contract.provenance()) - {path for path in sealed if path.startswith("docs/")}
        self.assertEqual(missing, set())

    def test_a_changed_byte_changes_the_provenance(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for rel in contract.provenance():
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(contract.ROOT / rel, target)
            before = contract.provenance(tmp)
            self.assertEqual(before, contract.provenance())
            for rel in ("experiments/gen_scene/levels.py", "experiments/gen_g1_data/README.md", "experiments/vla_inputs_signals/signals.py",
                        "experiments/props_scene/expert.py"):
                path = Path(tmp) / rel
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                self.assertNotEqual(contract.provenance(tmp)[rel], before[rel], rel)
                path.write_bytes(original)
            self.assertEqual(contract.provenance(tmp), before)

    def test_a_symlinked_source_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.py"
            target.write_text("x = 1\n")
            link = Path(tmp) / "link.py"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                contract._file_hash(link)

    def test_no_residue_from_the_p3_package_and_no_sibling_import_but_the_planned_ones(self):
        for path in PACKAGE.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md") and "__pycache__" not in path.parts:
                text = path.read_text()
                self.assertNotIn(SOURCE_PACKAGE, text, path)
                if path.name != "test_contract.py":
                    for line in text.splitlines():
                        if SIBLING in line and "import" in line:
                            self.assertRegex(line, r"from experiments\.props_p2_data import plan as p2|"
                                             r"from experiments\.props_p2_data\.plan import", path)

    def test_images_and_shard_count(self):
        self.assertTrue(contract.IMAGE.startswith("sha256:54d861aa"))
        self.assertTrue(contract.ADMISSION_IMAGE.startswith("sha256:08de1e09"))
        self.assertEqual(contract.SHARDS, 16)
        self.assertEqual(sum(len(p.shard_slots(k, contract.SHARDS)) for k in range(contract.SHARDS)), p.ADMIT_COUNT)

    def test_check_plan_accepts_the_plan_of_record_only(self):
        contract.check_plan(p.make_plan())
        changed = p.make_plan()
        changed["episodes"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            contract.check_plan(changed)

    def test_plan_and_contract_import_no_simulator(self):
        for name in ("mujoco", "robosuite", "torch", "lerobot"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
