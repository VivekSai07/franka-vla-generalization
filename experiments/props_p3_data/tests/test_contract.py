import tempfile
import unittest
from pathlib import Path

from experiments.props_p3_data import contract, plan as p

PACKAGE = Path(contract.__file__).resolve().parent
# Split so this file does not match its own scan.
SOURCE_PACKAGES = ("props_" + "p1_data", "props_" + "p2_data")


class ContractTests(unittest.TestCase):
    def test_provenance_binds_this_package_the_scene_the_spec_and_the_plan(self):
        prov = contract.provenance()
        for required in ("experiments/props_p3_data/plan.py", "experiments/props_p3_data/README.md",
                         "experiments/props_p3_data/tests/test_plan.py", "experiments/props_scene/expert.py",
                         "experiments/props_scene/scene.py", "experiments/robosuite_scene/xml_contract.py",
                         "experiments/paired_demonstrations/media.py", p.SPEC, p.PLAN):
            self.assertIn(required, prov)
        self.assertEqual(prov, contract.provenance())

    def test_provenance_does_not_reach_into_the_p1_or_p2_collections(self):
        for path in contract.provenance():
            for name in SOURCE_PACKAGES:
                self.assertNotIn(name, path)

    def test_no_residue_from_the_p1_or_p2_packages(self):
        # Review Focus 6: a fork that still imports P1 or P2 would test and record their code.
        for path in PACKAGE.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md"):
                for name in SOURCE_PACKAGES:
                    self.assertNotIn(name, path.read_text(), path)

    def test_the_readme_is_not_empty(self):
        # An absence-only check passes an empty file (FAILURES #56): a bound README must be present and say what this is.
        text = (PACKAGE / "README.md").read_text()
        self.assertGreater(len(text), 1000)
        for needle in ("props P3", p.PROMPT, "orange", "blue", "bottle", "cereal", "seal"):
            self.assertIn(needle, text)

    def test_a_symlinked_source_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            real, link = Path(tmp) / "real.py", Path(tmp) / "link.py"
            real.write_text("x = 1\n")
            link.symlink_to(real)
            self.assertEqual(len(contract._file_hash(real)), 64)
            with self.assertRaises(ValueError):
                contract._file_hash(link)

    def test_images_and_shard_count(self):
        self.assertTrue(contract.IMAGE.startswith("sha256:54d861aa"))
        self.assertTrue(contract.ADMISSION_IMAGE.startswith("sha256:08de1e09"))
        self.assertEqual(contract.OUTPUT, "outputs/props_p3_data")
        self.assertEqual(p.shard_slots(0, contract.SHARDS)[0], 0)
        self.assertEqual(len(p.shard_slots(contract.SHARDS - 1, contract.SHARDS)), p.ADMIT_COUNT // contract.SHARDS)

    def test_plan_and_contract_import_no_simulator(self):
        import sys
        for name in ("mujoco", "robosuite", "torch", "lerobot"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
