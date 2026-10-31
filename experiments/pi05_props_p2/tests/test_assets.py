import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.pi05_props_p2 import assets

REAL = assets.ASSETS_ROOT.is_dir()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp/'assets'
        self.data = {'a/one.bin': b'first file', 'b/two.json': b'{"second": 2}'}
        for name, data in self.data.items():
            (self.root/name).parent.mkdir(parents=True, exist_ok=True)
            (self.root/name).write_bytes(data)
        self.files = {name: (len(data), digest(data)) for name, data in self.data.items()}

    def test_matching_tree_passes_and_reports_hashes(self):
        self.assertEqual(assets.require_assets(self.root, self.files),
                         {name: sha for name, (_, sha) in self.files.items()})

    def test_one_byte_change_fails(self):
        path = self.root/'a/one.bin'
        path.write_bytes(b'first fil' + b'E')
        with self.assertRaisesRegex(ValueError, 'hash'):
            assets.require_assets(self.root, self.files)

    def test_size_change_fails(self):
        (self.root/'a/one.bin').write_bytes(b'first file!')
        with self.assertRaisesRegex(ValueError, 'size'):
            assets.require_assets(self.root, self.files)

    def test_missing_file_fails(self):
        (self.root/'b/two.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            assets.require_assets(self.root, self.files)

    def test_unlisted_file_fails(self):
        (self.root/'b/extra.json').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            assets.require_assets(self.root, self.files)

    def test_download_cache_directory_is_not_an_asset(self):
        (self.root/'a/.cache/huggingface').mkdir(parents=True)
        (self.root/'a/.cache/huggingface/x.metadata').write_bytes(b'x')
        assets.require_assets(self.root, self.files)

    def test_symlink_fails(self):
        target = self.tmp/'elsewhere.bin'
        target.write_bytes(self.data['a/one.bin'])
        (self.root/'a/one.bin').unlink()
        os.symlink(target, self.root/'a/one.bin')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            assets.require_assets(self.root, self.files)

    def test_missing_root_fails(self):
        with self.assertRaises(ValueError):
            assets.require_assets(self.tmp/'nothing', self.files)


class TableTest(unittest.TestCase):
    def test_pins(self):
        self.assertEqual(assets.WEIGHTS_REVISION, '9e55186ad36e66b95cda57bc47818d9e6237ae30')
        self.assertEqual(assets.TOKENIZER_REVISION, '35e4f46485b4d07967e7e9935bc3786aad50687c')
        size, sha = assets.FILES['pi05_base/model.safetensors']
        self.assertEqual(size, 14_467_165_872)
        self.assertEqual(sha, '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f')

    def test_table_covers_both_repositories(self):
        self.assertEqual({name.split('/')[0] for name in assets.FILES}, {'pi05_base', 'paligemma-3b-pt-224'})
        self.assertIn('paligemma-3b-pt-224/tokenizer.json', assets.FILES)
        self.assertIn('paligemma-3b-pt-224/tokenizer.model', assets.FILES)
        self.assertTrue(all(len(sha) == 64 and size > 0 for size, sha in assets.FILES.values()))


@unittest.skipUnless(REAL, 'pi05 assets are not mounted at outputs/research/pi05_assets')
class RealAssetsTest(unittest.TestCase):
    def test_real_assets_match_the_pinned_table(self):
        found = assets.require_assets()
        self.assertEqual(set(found), set(assets.FILES))

    def test_real_assets_reject_a_one_byte_change_in_a_copied_tokenizer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = {name: value for name, value in assets.FILES.items() if name.startswith('paligemma')}
            for name in small:
                (Path(tmp)/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(assets.ASSETS_ROOT/name, Path(tmp)/name)
            assets.require_assets(Path(tmp), small)
            path = Path(tmp)/'paligemma-3b-pt-224/tokenizer_config.json'
            data = bytearray(path.read_bytes())
            data[10] ^= 1
            path.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, 'hash'):
                assets.require_assets(Path(tmp), small)


if __name__ == '__main__':
    unittest.main()
