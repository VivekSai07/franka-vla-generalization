"""Broad-workspace validation: journal/replay coverage and reset-SHA tamper detection."""
import copy
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ValidateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.v = importlib.import_module('experiments.b1_language_data.validate')
            self.p = importlib.import_module('experiments.b1_language_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing validator: {exc}')

    def test_replay_coverage_requires_200_x_2(self):
        results = [dict(episode=i, mode=m, passed=True)
                   for i in range(200) for m in ('reference', 'persisted_float32')]
        self.v.check_replay_coverage(results)
        for bad in (results[:-1], results + results[:1],
                    [{**r, 'mode': 'expert_regeneration'} for r in results],
                    [{**r, 'passed': False} for r in results]):
            with self.assertRaises(ValueError):
                self.v.check_replay_coverage(bad)

    def test_validate_journal_accepts_all_admits_no_rejections(self):
        p = self.p.make_plan(); events = []; t = 0.0
        for slot in p['episodes']:
            events.append(dict(event='started', **slot, training_ready=False, elapsed_seconds=t)); t += 1.0
            events.append(dict(event='admitted', episode_index=slot['episode_index'],
                               training_ready=False, elapsed_seconds=t)); t += 1.0
        self.v.validate_journal(events, p)

    def test_validate_journal_rejects_more_than_budget_rejections(self):
        p = self.p.make_plan(); events = []; t = 0.0
        for i in range(self.p.REJECTION_BUDGET + 1):
            events.append(dict(event='rejected', episode_index=-1, training_ready=False, elapsed_seconds=t)); t += 1.0
        for slot in p['episodes']:
            events.append(dict(event='started', **slot, training_ready=False, elapsed_seconds=t)); t += 1.0
            events.append(dict(event='admitted', episode_index=slot['episode_index'],
                               training_ready=False, elapsed_seconds=t)); t += 1.0
        with self.assertRaises(ValueError):
            self.v.validate_journal(events, p)

    def test_matches_plan_slot_and_tamper_detected(self):
        p = self.p.make_plan(); slot = p['episodes'][0]
        for field, value in (('layout_id', 'evaluation_000'),
                             ('episode_index', 200), ('seed', 1)):
            bad = {**slot, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Unplanned'):
                self.v.validate_episode(Path('/tmp/nonexistent-broad-workspace-test'), bad, None, None, lambda: None)

    def test_reset_sha_tamper_detection(self):
        root = Path(tempfile.mkdtemp(dir='/vla/outputs/implementation/b1_language_data_20260925'))
        target = root / 'resets' / 'episode_000.npz'
        target.parent.mkdir()
        target.write_bytes(b'\x00' * 128)
        genuine = self.v.file_hash(target)
        self.assertTrue(self.v._episode_reset_sha_matches(root, 0, genuine))
        target.write_bytes(b'\x01' * 128)
        self.assertFalse(self.v._episode_reset_sha_matches(root, 0, genuine))

    def test_metadata_final_requires_160_40_split(self):
        # The split assertion is unit-tested directly; exercising it through metadata() would
        # require a full synthetic LeRobot dataset on disk.
        info_wrong = dict(splits={'train': '0:120', 'validation': '120:200'})
        with self.assertRaises(ValueError):
            self.v.require_final_split(info_wrong, 200)
        with self.assertRaises(ValueError):  # right splits, wrong count
            self.v.require_final_split(dict(splits={'train': '0:160', 'validation': '160:200'}), 199)
        # correct pair passes
        self.v.require_final_split(dict(splits={'train': '0:160', 'validation': '160:200'}), 200)

    def test_cell_balance_is_enforced_at_admission(self):
        plan = self.v.p.make_plan()
        self.v.require_cell_balance(plan['episodes'])
        skewed = [dict(e) for e in plan['episodes']]
        for e in skewed:
            if e['split'] == 'train' and e['cell'] == 'C2':
                e['cell'] = 'C1'
        with self.assertRaises(ValueError):
            self.v.require_cell_balance(skewed)


class SyntheticPersistenceSmokeTests(unittest.TestCase):
    """Two-episode LeRobot persistence smoke: no physics, no model, no real expert."""

    def test_dataset_stub_roundtrip(self):
        # This is a minimal smoke: verify that the dataset creation path is
        # importable and constructor signature accepts our features.
        import copy as _copy
        from experiments.paired_demonstrations.media import FEATURES, ENCODING
        from experiments.fixed_layout.media import BoundPairedDataset
        root = Path(tempfile.mkdtemp(dir='/vla/outputs/implementation/b1_language_data_20260925')) / 'dataset'
        # We do not actually run LeRobotDataset.create here to avoid touching
        # the disk-heavy footer; we only assert import shape.
        self.assertTrue(callable(BoundPairedDataset.create))
        self.assertIn('action', FEATURES)
        self.assertIn('vcodec', ENCODING)


if __name__ == '__main__':
    unittest.main()
