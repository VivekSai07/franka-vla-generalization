import hashlib
import json
import unittest

from experiments.gen_scene import pools

MANIFEST_SHA256 = "f488ed6b0c1467294ee53fcc13e074d077e19cf8880ec9452fdd9b99ec8683d6"
SPLIT_SHA256 = "d0e3a86cffcd9c2cfde82e76053129ed56469a95b0cbc3a75c3a8577318b5825"
BIN_COLOURS = ((0.15, 0.35, 0.85), (0.95, 0.55, 0.10))


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class SplitRule(unittest.TestCase):
    NAMES = [f"T{i:03d}.jpg" for i in range(60)]

    def test_deterministic_disjoint_and_complete(self):
        a, b = pools.split(self.NAMES), pools.split(reversed(self.NAMES))
        self.assertEqual(a, b)
        self.assertFalse(set(a["train"]) & set(a["test"]))
        self.assertEqual(sorted(a["train"] + a["test"]), sorted(self.NAMES))

    def test_every_fifth_by_hash_goes_to_test(self):
        out = pools.split(self.NAMES)
        self.assertEqual((len(out["train"]), len(out["test"])), (48, 12))
        ordered = sorted(self.NAMES, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
        self.assertEqual(out["test"], sorted(ordered[4::5]))

    def test_duplicates_refused(self):
        with self.assertRaises(ValueError):
            pools.split(["a.jpg", "a.jpg"])


class Manifest(unittest.TestCase):
    def setUp(self):
        self.manifest = pools.load_manifest()

    def test_counts_and_licence(self):
        by_surface = {s: [t for t in self.manifest["textures"] if t["surface"] == s] for s in pools.SURFACES}
        self.assertEqual({s: len(v) for s, v in by_surface.items()}, {"table": 60, "room": 60})
        self.assertTrue(all(t["licence"] == "CC0-1.0" and t["width"] == t["height"] == 1024 for t in self.manifest["textures"]))
        self.assertEqual(len({t["name"] for t in self.manifest["textures"]}), 120)
        self.assertEqual(self.manifest["source"]["revision"], "50329ad52d825bfb740080546c30c04014983ec8")

    def test_manifest_is_the_pinned_one(self):
        self.assertEqual(pools.manifest_digest(self.manifest), MANIFEST_SHA256)

    def test_pool_split_is_the_pinned_one(self):
        digest = hashlib.sha256(json.dumps(pools.pools(self.manifest), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, SPLIT_SHA256)
        for surface, parts in pools.pools(self.manifest).items():
            self.assertEqual((len(parts["train"]), len(parts["test"])), (48, 12), surface)

    def test_table_and_room_share_no_texture(self):
        both = pools.pools(self.manifest)
        self.assertFalse(set(both["table"]["train"] + both["table"]["test"]) & set(both["room"]["train"] + both["room"]["test"]))

    def test_files_match_the_manifest(self):
        self.assertEqual(pools.verify_files(self.manifest), 120)

    def test_a_changed_file_is_detected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            for t in self.manifest["textures"]:
                (Path(tmp) / t["name"]).write_bytes(b"x" * t["bytes"])
            with self.assertRaises(ValueError):
                pools.verify_files(self.manifest, tmp)


class Primitives(unittest.TestCase):
    def test_pools_share_no_shape_family_and_no_shape_name(self):
        families = {pool: {kind for kind, _ in shapes.values()} for pool, shapes in pools.PRIMITIVE_SHAPES.items()}
        self.assertFalse(families["A"] & families["B"])
        self.assertFalse(set(pools.PRIMITIVE_SHAPES["A"]) & set(pools.PRIMITIVE_SHAPES["B"]))
        self.assertEqual(families, {"A": {"box", "cylinder"}, "B": {"sphere", "ellipsoid", "capsule"}})

    def test_pools_share_no_colour_and_none_is_near_the_other_pool_or_a_bin(self):
        a, b = pools.PRIMITIVE_COLOURS["A"], pools.PRIMITIVE_COLOURS["B"]
        self.assertFalse(set(a) & set(b))
        for ca in a.values():
            for cb in b.values():
                self.assertGreaterEqual(distance(ca, cb), 0.25)
        for colour in (*a.values(), *b.values()):
            for bin_colour in BIN_COLOURS:
                self.assertGreaterEqual(distance(colour, bin_colour), 0.25)

    def test_sizes_are_object_sized(self):
        for shapes in pools.PRIMITIVE_SHAPES.values():
            for kind, size in shapes.values():
                self.assertTrue(all(0.01 <= v <= 0.06 for v in size), (kind, size))


if __name__ == "__main__":
    unittest.main()
