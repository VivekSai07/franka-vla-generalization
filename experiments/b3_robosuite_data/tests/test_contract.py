import re
import unittest
from pathlib import Path

from experiments.b3_robosuite_data import contract


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = contract.provenance()

    def test_every_value_is_a_sha256(self):
        self.assertTrue(self.provenance)
        for path, value in self.provenance.items():
            self.assertRegex(value, r"^[0-9a-f]{64}$", path)

    def test_lists_what_the_collection_depends_on(self):
        keys = set(self.provenance)
        package = Path(contract.ROOT / "experiments/b3_robosuite_data")
        for path in package.rglob("*.py"):
            self.assertIn(str(path.relative_to(contract.ROOT)), keys)
        self.assertIn("experiments/b3_robosuite_data/README.md", keys)
        for name in ("scene.py", "xml_contract.py", "validate.py", "expert.py", "record.py"):
            self.assertIn(f"experiments/robosuite_scene/{name}", keys)
        self.assertIn(contract.SPEC, keys)
        self.assertIn(contract.PLAN, keys)
        self.assertIn("experiments/b2_distractor_data/expert.py", keys)
        self.assertIn("data_collection/benchmark.py", keys)
        self.assertTrue(any(k.startswith("src/robots/franka_panda/model/") for k in keys))

    def test_is_stable_and_sorted(self):
        self.assertEqual(contract.provenance(), self.provenance)
        self.assertEqual(list(self.provenance), sorted(self.provenance))

    def test_images_are_the_pinned_digests(self):
        self.assertEqual(contract.IMAGE, "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e")
        self.assertEqual(contract.ADMISSION_IMAGE,
                         "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc")

    def test_changing_a_bound_file_changes_provenance(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in self.provenance:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(contract.ROOT / rel, target)
            before = contract.provenance(root)
            self.assertEqual(before, self.provenance)
            (root / "experiments/b3_robosuite_data/README.md").write_text("changed")
            self.assertNotEqual(contract.provenance(root), before)

    def test_symlinked_source_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in self.provenance:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x")
            link = root / "experiments/b3_robosuite_data/README.md"
            link.unlink()
            link.symlink_to(root / contract.SPEC)
            with self.assertRaises(ValueError):
                contract.provenance(root)


if __name__ == "__main__":
    unittest.main()
