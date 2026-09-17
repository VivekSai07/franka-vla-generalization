import hashlib
import tempfile
import unittest
from pathlib import Path


class AssetTests(unittest.TestCase):
    def test_fixed_public_assets_exclude_tokenizer_weights(self):
        from experiments.xvla_trial.assets import ASSETS
        self.assertEqual(set(ASSETS), {'model', 'tokenizer'})
        self.assertEqual(ASSETS['model']['revision'], 'cdb7964e4fe842935d671bfab5a5ebe00a96648c')
        self.assertEqual(ASSETS['tokenizer']['revision'], 'cb48c1365bd826bd521f650dc2e0940aee54720c')
        self.assertEqual(set(ASSETS['tokenizer']['files']),
                         {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'})

    def test_asset_hash_and_size_are_checked(self):
        from experiments.xvla_trial.assets import verify_file
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'asset'
            p.write_bytes(b'abc')
            digest = hashlib.sha256(b'abc').hexdigest()
            self.assertEqual(verify_file(p, 3, digest), {'bytes': 3, 'sha256': digest})
            for size, sha in [(4, digest), (3, '0'*64)]:
                with self.assertRaises(ValueError):
                    verify_file(p, size, sha)

    def test_fresh_path_rejects_existing_and_symlink(self):
        from experiments.xvla_trial.assets import claim_directory
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'new'
            claim_directory(path)
            with self.assertRaises(FileExistsError):
                claim_directory(path)
            alias = Path(d)/'alias'
            alias.symlink_to(path, target_is_directory=True)
            with self.assertRaises(ValueError):
                claim_directory(alias/'child')

    def test_exception_report_cannot_leak_urls_or_tokens(self):
        from experiments.xvla_trial.assets import safe_error
        exc = RuntimeError('https://signed.invalid?token=SECRET password=SECRET')
        self.assertEqual(safe_error(exc), {'error_type': 'RuntimeError', 'http_status': None})

    def test_download_failure_stops_and_keeps_safe_partial_manifest(self):
        from unittest.mock import patch
        from experiments.xvla_trial.assets import download
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cache'
            with patch('huggingface_hub.HfApi.model_info', side_effect=RuntimeError('SECRET')):
                self.assertFalse(download(path))
            text = (path/'manifest.json').read_text()
            self.assertIn('blocked', text)
            self.assertNotIn('SECRET', text)
            self.assertFalse((path/'model/model.safetensors').exists())