import unittest
from pathlib import Path

from experiments.props_p2_data import contract, plan as p

PACKAGE = Path(contract.__file__).resolve().parent
SOURCE_PACKAGE = "props_" + "p1_data"  # split so this file does not match its own scan


class ContractTests(unittest.TestCase):
    def test_provenance_binds_this_package_the_scene_the_spec_and_the_plan(self):
        prov = contract.provenance()
        for required in ("experiments/props_p2_data/plan.py", "experiments/props_p2_data/README.md",
                         "experiments/props_p2_data/tests/test_plan.py", "experiments/props_scene/expert.py",
                         "experiments/props_scene/scene.py", "experiments/robosuite_scene/xml_contract.py",
                         "experiments/paired_demonstrations/media.py", p.SPEC, p.PLAN):
            self.assertIn(required, prov)
        self.assertEqual(prov, contract.provenance())

    def test_no_residue_from_the_p1_package(self):
        # Review Focus 6: a fork that still imports P1 would test and record P1's code.
        for path in PACKAGE.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md"):
                self.assertNotIn(SOURCE_PACKAGE, path.read_text(), path)

    def test_images_and_shard_count(self):
        self.assertTrue(contract.IMAGE.startswith("sha256:54d861aa"))
        self.assertTrue(contract.ADMISSION_IMAGE.startswith("sha256:08de1e09"))
        p.shard_slots(0, contract.SHARDS)

    def test_plan_and_contract_import_no_simulator(self):
        import sys
        for name in ("mujoco", "robosuite", "torch", "lerobot"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
