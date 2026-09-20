"""Metadata finalization must not invalidate numerical evidence or widen uploads."""
import json
from pathlib import Path
import tempfile
import unittest

from publication import package_baselines as package, publish


class CardFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.output = self.repo / package.RELEASE_RELATIVE
        self.root = self.output / 'packages/xvla'
        self.root.mkdir(parents=True)
        for name in ('config.json', 'runtime.json'):
            package.write_json(self.root / name, {})
        for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
            package.write_json(self.root / name, {'steps': []})
        package.write_json(self.root / 'release_manifest.json', {'model': 'xvla'})
        for name in ('README.md', 'load_baseline.py'):
            (self.root / name).write_text('original\n')
        assets = self.root / 'assets/tokenizer'
        assets.mkdir(parents=True)
        for name in package.XVLA_ASSETS:
            (assets / name).write_text('{}')
        header = b'{"__metadata__": {}}'
        (self.root / 'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header)
        self.refresh_inventory()
        package.write_json(self.output / 'packaging-verification.json', {'results': {'xvla': {
            'status': 'PASS', 'prediction_max_absolute_error': 0,
            'storage': {'all_stored_tensors_bit_exact': True},
            'weights_sha256': package.sha256(self.root / 'model.safetensors'),
            'release_files': 14, 'release_bytes': 1234,
        }}})
        self.proof_bytes = (self.output / 'packaging-verification.json').read_bytes()

    def refresh_inventory(self):
        (self.root / 'inventory.json').write_text(json.dumps(
            package.inventory(self.root, exclude=('inventory.json',)), sort_keys=True))

    def add_notices(self):
        (self.root / 'LICENSE').write_text('Apache License, Version 2.0\n')
        (self.root / 'NOTICE').write_text('Upstream attribution; modified personal archive.\n')
        self.refresh_inventory()

    def test_exact_license_and_notice_are_audited_and_publishable(self):
        before = package.audit_release(self.root)
        self.add_notices()
        after = package.audit_release(self.root)
        self.assertEqual(after['files'], before['files'] + 2)
        self.assertEqual(after['text_files_scanned'], before['text_files_scanned'] + 2)
        files = publish.reviewed_files(self.repo, 'xvla')
        self.assertIn('LICENSE', files)
        self.assertIn('NOTICE', files)

    def test_metadata_changes_use_fresh_inventory_not_old_report_size(self):
        old = package.inventory(self.root)
        (self.root / 'README.md').write_text('Final personal derivative archive with attribution.\n')
        package.write_json(self.root / 'LICENSE', 'Apache-2.0')
        (self.root / 'release_manifest.json').write_text(json.dumps({
            'model': 'xvla', 'provenance': {'license_status': 'upstream_card_declaration'}}))
        self.refresh_inventory()
        files = publish.reviewed_files(self.repo, 'xvla')
        self.assertNotEqual(files['README.md'], old['README.md'])
        self.assertEqual(files['model.safetensors'], old['model.safetensors'])
        self.assertEqual((self.output / 'packaging-verification.json').read_bytes(), self.proof_bytes)

    def test_license_and_notice_do_not_allow_neighboring_files(self):
        for name in ('LICENSE.extra', 'NOTICE.txt', 'extra.py'):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text('not allowed')
                self.refresh_inventory()
                with self.assertRaisesRegex(ValueError, 'allowlist'):
                    package.audit_release(self.root)
                path.unlink()

    def test_license_and_notice_are_not_exempt_from_text_scan(self):
        self.add_notices()
        for name in ('LICENSE', 'NOTICE'):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text()
                path.write_text('hf_' + 'a' * 32)
                self.refresh_inventory()
                with self.assertRaisesRegex(ValueError, 'credential-like'):
                    package.audit_release(self.root)
                path.write_text(original)

    def test_uninventoried_notice_changes_are_rejected(self):
        self.add_notices()
        (self.root / 'NOTICE').write_text('different attribution')
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            publish.reviewed_files(self.repo, 'xvla')

    def test_weight_changes_still_rejected_even_with_fresh_inventory(self):
        path = self.root / 'model.safetensors'
        path.write_bytes(path.read_bytes() + b'changed weight bytes')
        self.refresh_inventory()
        with self.assertRaisesRegex(ValueError, 'weight hash mismatch'):
            publish.reviewed_files(self.repo, 'xvla')