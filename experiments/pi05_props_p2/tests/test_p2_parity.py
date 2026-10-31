"""The copied P2 dataset binding agrees with the X-VLA contract's original on the real sealed dataset.

This is the only test file allowed to import `experiments.xvla_props_p2`; the contract itself imports nothing from it.
Needs the sealed P2 root mounted; skipped without it. The two full `load_data` passes read every one of the 244,080 rows.
"""
import copy
import unittest

import pandas as pd

from experiments.pi05_props_p2 import contract as new
from experiments.xvla_props_p2 import contract as old

HAVE_PARENT = (new.DATA/'meta/info.json').is_file()


class ConstantParityTests(unittest.TestCase):
    def test_the_dataset_constants_are_the_originals(self):
        for name in ('SEAL', 'SEAL_RECORD', 'VALIDATION_RECORD', 'FRAMES_PER_EPISODE', 'TRAIN', 'VAL', 'SEED',
                     'TRAINING_ROWS', 'VALIDATION_ROWS', 'TOTAL_FRAMES', 'CAMERAS', 'OBS', 'INSTRUCTIONS'):
            self.assertEqual(getattr(new, name), getattr(old, name), name)
        self.assertEqual(new.PARENT, old.PARENT)
        self.assertEqual(new.DATA, old.DATA)

    def test_only_the_chunk_dependent_numbers_differ(self):
        self.assertEqual((old.CHUNK_SIZE, new.CHUNK_SIZE), (30, 50))
        self.assertEqual((old.COMPLETE_STARTS, len(new.windows())), (203_040, 193_440))


@unittest.skipUnless(HAVE_PARENT, 'sealed P2 root not mounted')
class RealDatasetParityTests(unittest.TestCase):
    def test_instructions_and_task_table(self):
        self.assertEqual(new.planned_instructions(), old.planned_instructions())
        tasks = pd.read_parquet(new.DATA/'meta/tasks.parquet')
        ds = type('DS', (), {'meta': type('M', (), {'tasks': tasks})()})()
        self.assertEqual(new.task_index_map(ds), old.task_index_map(ds))

    def test_episode_metadata_checks_agree_on_the_real_and_on_changed_metadata(self):
        episodes = pd.read_parquet(next((new.DATA/'meta/episodes').rglob('*.parquet')))
        rows = [dict(r, tasks=list(r['tasks'])) for r in episodes.to_dict('records')]
        new_rows, _ = new.indices(rows)
        old_rows, _ = old.indices(rows)
        self.assertEqual(new_rows, old_rows)
        changed = copy.deepcopy(rows)
        changed[9]['tasks'] = [next(t for t in new.INSTRUCTIONS if t != changed[9]['tasks'][0])]
        for module in (new, old):
            with self.assertRaises(ValueError):
                module.indices(changed)

    def test_binding_fields_agree(self):
        from unittest.mock import patch
        with patch.object(new.assets, 'require_assets', return_value={}), \
                patch.object(old, 'require_assets', return_value={}), \
                patch.object(new, 'source_hashes', return_value={}), patch.object(old, 'source_hashes', return_value={}):
            a, b = new.current_binding(), old.current_binding()
        for key in ('dataset_fingerprint', 'seal_sha256', 'plan_sha256', 'provenance_sha256', 'validation_sha256',
                    'props_p2_provenance'):
            self.assertEqual(a[key], b[key], key)

    def test_load_data_statistics_agree_with_the_original(self):
        new_ds, new_starts, new_stats = new.load_data()
        old_ds, old_starts, old_stats = old.load_data()
        self.assertEqual(len(new_ds), len(old_ds))
        self.assertEqual(len(new_starts), 193_440)
        self.assertEqual(len(old_starts), 203_040)
        for key in ('observation.state', 'action'):
            for name in ('mean', 'std', 'min', 'max'):
                self.assertTrue((new_stats[key][name] == old_stats[key][name]).all(), (key, name))
            self.assertTrue((new_stats[key]['q01'] <= new_stats[key]['q99']).all())
            self.assertTrue((new_stats[key]['min'] <= new_stats[key]['q01']).all())
            self.assertTrue((new_stats[key]['q99'] <= new_stats[key]['max']).all())


if __name__ == '__main__':
    unittest.main()
