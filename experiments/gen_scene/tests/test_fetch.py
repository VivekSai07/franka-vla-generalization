import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from experiments.gen_scene import fetch


def jpeg_bytes(width, height):
    """A syntactically minimal JPEG header: SOI, an APP0 segment, SOF0 carrying the size."""
    sof = b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00" + sof


def rows_for(families, n):
    return [dict(id=f"{f}{i:03d}", download_url=f"https://x/get?file={f}{i:03d}_1K-JPG.zip") for f in families for i in range(n)]


class Selection(unittest.TestCase):
    def setUp(self):
        self.all_families = fetch.FAMILIES["table"] + fetch.FAMILIES["room"]
        self.rows = rows_for(self.all_families, 9)
        self.sizes = {fetch.member_name(r["id"]): (1024, 1024) for r in self.rows}

    def test_five_per_family_sorted_and_surfaces_disjoint(self):
        out = fetch.choose(self.rows, self.sizes)
        self.assertEqual((len(out["table"]), len(out["room"])), (60, 60))
        self.assertEqual(out["table"], sorted(out["table"]))
        for surface, families in fetch.FAMILIES.items():
            for family in families:
                self.assertEqual(sum(fetch.family_of(i) == family for i in out[surface]), 5)

    def test_smallest_hash_wins(self):
        out = fetch.choose(self.rows, self.sizes)
        wood = sorted((r["id"] for r in self.rows if fetch.family_of(r["id"]) == "Wood"), key=lambda i: fetch.sha256_hex(i.encode()))
        self.assertEqual({i for i in out["table"] if fetch.family_of(i) == "Wood"}, set(wood[:5]))

    def test_order_of_rows_does_not_matter(self):
        self.assertEqual(fetch.choose(self.rows, self.sizes), fetch.choose(list(reversed(self.rows)), self.sizes))

    def test_ineligible_materials_are_skipped(self):
        rows = [dict(r) for r in self.rows]
        for r in rows:
            if r["id"] == "Wood000":
                r["download_url"] = r["download_url"].replace("1K-JPG.zip", "4K-JPG.zip")
        sizes = dict(self.sizes)
        sizes[fetch.member_name("Wood001")] = (1024, 512)
        del sizes[fetch.member_name("Wood002")]
        out = fetch.choose(rows, sizes)["table"]
        self.assertFalse({"Wood000", "Wood001", "Wood002"} & set(out))
        self.assertEqual(sum(fetch.family_of(i) == "Wood" for i in out), 5)

    def test_a_short_family_is_an_error(self):
        with self.assertRaises(ValueError):
            fetch.choose([r for r in self.rows if r["id"] not in {"Wood%03d" % i for i in range(6)}], self.sizes)

    def test_family_of(self):
        self.assertEqual([fetch.family_of(i) for i in ("Wood084B", "WoodFloor037", "PaintedPlaster013", "Concrete042A")],
                         ["Wood", "WoodFloor", "PaintedPlaster", "Concrete"])
        with self.assertRaises(ValueError):
            fetch.family_of("Foo")


class JpegHeader(unittest.TestCase):
    def test_size_is_width_then_height(self):
        self.assertEqual(fetch.jpeg_size(jpeg_bytes(1024, 512)), (1024, 512))

    def test_non_jpeg_is_refused(self):
        with self.assertRaises(ValueError):
            fetch.jpeg_size(b"\x89PNG....")


class Extraction(unittest.TestCase):
    def test_build_end_to_end_on_a_synthetic_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            families = fetch.FAMILIES["table"] + fetch.FAMILIES["room"]
            rows = rows_for(families, 6)
            members = {fetch.member_name(r["id"]): jpeg_bytes(1024, 1024) for r in rows}
            tar_path = tmp / "t.tar"
            with tarfile.open(tar_path, "w") as tar:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            out, manifest_path = tmp / "out", tmp / "manifest.json"
            original = (fetch.verify_stage, fetch.read_metadata, fetch.SHARDS)
            fetch.verify_stage = lambda stage: None
            fetch.read_metadata = lambda stage: rows
            fetch.SHARDS = {"t.tar": "x"}
            try:
                manifest = fetch.build(tmp, out, manifest_path)
            finally:
                fetch.verify_stage, fetch.read_metadata, fetch.SHARDS = original
            self.assertEqual(len(manifest["textures"]), 120)
            self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(t["name"] for t in manifest["textures"]))
            self.assertEqual(json.loads(manifest_path.read_text()), json.loads(json.dumps(manifest)))
            for t in manifest["textures"]:
                self.assertEqual(fetch.sha256_hex((out / t["name"]).read_bytes()), t["sha256"])
            with self.assertRaises(FileExistsError):  # never overwrites an existing texture folder
                fetch.verify_stage = lambda stage: None
                try:
                    fetch.build(tmp, out, manifest_path)
                finally:
                    fetch.verify_stage = original[0]

    def test_a_corrupt_stage_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (fetch.METADATA, *fetch.SHARDS):
                (Path(tmp) / name).write_bytes(b"not the published bytes")
            with self.assertRaises(ValueError):
                fetch.verify_stage(tmp)


if __name__ == "__main__":
    unittest.main()
