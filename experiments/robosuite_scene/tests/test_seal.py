"""The seal fingerprint binds every file under the root except seal.json itself.

Uses only the standard library plus validate.py, so it runs in the admission image without robosuite.
"""
import os
import tempfile
import unittest
from pathlib import Path


class SealTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "root"
        (self.root / "dataset" / "data").mkdir(parents=True)
        (self.root / "episodes.json").write_text("[]")
        (self.root / "dataset" / "data" / "file.bin").write_bytes(b"\x00\x01")

    def tearDown(self):
        self._tmp.cleanup()

    def test_seal_file_is_excluded(self):
        from experiments.robosuite_scene.validate import fingerprint
        before = fingerprint(self.root)
        self.assertRegex(before, "^[0-9a-f]{64}$")
        (self.root / "seal.json").write_text('{"fingerprint": "a"}')
        self.assertEqual(fingerprint(self.root), before)
        (self.root / "seal.json").write_text('{"fingerprint": "b"}')
        self.assertEqual(fingerprint(self.root), before)

    def test_only_the_root_seal_is_excluded(self):
        from experiments.robosuite_scene.validate import fingerprint
        before = fingerprint(self.root)
        (self.root / "dataset" / "seal.json").write_text("{}")
        self.assertNotEqual(fingerprint(self.root), before)

    def test_any_other_change_changes_the_fingerprint(self):
        from experiments.robosuite_scene.validate import fingerprint
        before = fingerprint(self.root)
        (self.root / "dataset" / "data" / "file.bin").write_bytes(b"\x00\x02")
        changed = fingerprint(self.root)
        self.assertNotEqual(changed, before)
        (self.root / "extra.txt").write_text("")
        added = fingerprint(self.root)
        self.assertNotEqual(added, changed)
        (self.root / "extra.txt").rename(self.root / "renamed.txt")
        self.assertNotEqual(fingerprint(self.root), added)

    def test_symlink_is_rejected(self):
        from experiments.robosuite_scene.validate import fingerprint
        os.symlink(self.root / "episodes.json", self.root / "dataset" / "link.json")
        with self.assertRaises(ValueError):
            fingerprint(self.root)

    def test_symlink_root_is_rejected(self):
        from experiments.robosuite_scene.validate import fingerprint
        link = self.root.parent / "link"
        os.symlink(self.root, link)
        with self.assertRaises(ValueError):
            fingerprint(link)


if __name__ == "__main__":
    unittest.main()
