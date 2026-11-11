"""The signals join in the training contract: sealed state rows plus the clipped, standardised G1 signals, by explicit keys."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_contract import FRAMES, metadata, use_planned
from test_forces import fresh_loaded, mounted, real_derived, synthetic_table, use_synthetic_table


def contract():
    return importlib.import_module('experiments.xvla_g1_rand.contract')


def forces():
    return importlib.import_module('experiments.xvla_g1_rand.forces')


def sealed_state(c, episode, frame):
    """The sealed dataset's `observation.state` (float32, 9 values) at (episode, frame), read from the parquet."""
    import pyarrow.parquet as pq
    root = c.DATA/'data'
    guess = list(root.rglob(f'file-{episode:03d}.parquet'))
    for path in guess+[p for p in sorted(root.rglob('*.parquet')) if p not in guess]:
        table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state']).to_pandas()
        hit = table[(table['episode_index'] == episode) & (table['frame_index'] == frame)]
        if len(hit):
            return np.asarray(hit['observation.state'].iloc[0], dtype=np.float32)
    raise AssertionError('row not in the sealed dataset')


class FakeBase:
    """A dataset whose rows carry only what the join reads, at the sealed positions."""

    def __init__(self, shift=0, episode_shift=0, width=9):
        self.shift, self.episode_shift, self.width = shift, episode_shift, width
        self.meta = 'meta'

    def __len__(self):
        return 3000*FRAMES

    def __getitem__(self, i):
        return {'episode_index': torch.tensor(i//FRAMES+self.episode_shift), 'frame_index': torch.tensor(i % FRAMES+self.shift),
                'observation.state': torch.full((self.width,), float(i % 7)), 'task': 'x'}


class JoinTests(unittest.TestCase):
    """The join on a synthetic table: its logic needs no real signals."""

    def setUp(self):
        self.c, self.f = contract(), forces()
        use_planned(self, self.c)
        use_synthetic_table(self)

    def test_the_policy_state_is_17_values_the_nine_joints_then_the_eight_extras(self):
        c = self.c
        self.assertEqual(c.OBS['observation.state'], (17,))
        self.assertEqual(c.DATASET_OBS['observation.state'], (9,))
        self.assertEqual({k: v for k, v in c.OBS.items() if k != 'observation.state'},
                         {k: v for k, v in c.DATASET_OBS.items() if k != 'observation.state'})

    def test_joined_state_is_the_state_then_the_standardised_signals_of_the_same_frame(self):
        c = self.c
        state = torch.arange(9, dtype=torch.float32)
        for episode, frame in ((5, 17), (2699, 451), (2700, 0)):
            joined = c.joined_state(state, episode, frame)
            self.assertEqual((joined.dtype, tuple(joined.shape)), (torch.float32, (17,)))
            self.assertTrue(torch.equal(joined[:9], state))
            self.assertEqual(joined[9:].numpy().tobytes(), synthetic_table()[episode, frame].tobytes())
        self.assertNotEqual(c.joined_state(state, 5, 17)[9:].tolist(), c.joined_state(state, 5, 18)[9:].tolist())
        self.assertNotEqual(c.joined_state(state, 5, 17)[9:].tolist(), c.joined_state(state, 6, 17)[9:].tolist())

    def test_joined_state_refuses_a_state_that_is_not_nine_finite_float32_values(self):
        c = self.c
        for bad in (torch.zeros(17), torch.zeros(8), torch.zeros(9, dtype=torch.float64),
                    torch.full((9,), float('nan')), torch.zeros(1, 9)):
            with self.assertRaises(ValueError):
                c.joined_state(bad, 0, 0)

    def test_the_dataset_wrapper_joins_by_episode_and_frame(self):
        c = self.c
        ds = c.JoinedDataset(FakeBase())
        self.assertEqual(len(ds), c.TOTAL_FRAMES)
        self.assertEqual(ds.meta, 'meta')  # everything else is delegated
        for index in (0, 451, 452, c.TRAINING_ROWS-1, c.TRAINING_ROWS, c.TOTAL_FRAMES-1):
            row = ds[index]
            self.assertEqual(tuple(row['observation.state'].shape), (17,))
            self.assertEqual(row['observation.state'][9:].numpy().tobytes(),
                             synthetic_table()[index//FRAMES, index % FRAMES].tobytes())
            self.assertEqual(row['observation.state'][:9].tolist(), [float(index % 7)]*9)

    def test_the_dataset_wrapper_refuses_shifted_or_foreign_rows(self):
        c = self.c
        for kwargs in ({'shift': 1}, {'shift': -1}, {'episode_shift': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase(**kwargs))[600]
        with self.assertRaises(ValueError):  # a 17-value base state is already joined or from another dataset
            c.JoinedDataset(FakeBase(width=17))[0]

    def test_the_dataset_wrapper_verifies_the_signals_before_it_serves_a_row(self):
        c = self.c
        with patch.object(c.forces, 'table', side_effect=ValueError('seal does not verify')):
            with self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase())

    def test_every_training_row_has_a_signals_row(self):
        c, f = self.c, self.f
        rows, starts = c.indices(metadata())
        self.assertEqual(len(rows), c.TRAINING_ROWS)
        sample = rows[::1009]+[rows[-1]]
        extras = f.expected_extras(sample)
        self.assertEqual(extras.shape, (len(sample), 8))
        self.assertTrue(np.isfinite(extras).all())
        self.assertEqual(len(starts), c.COMPLETE_STARTS)


class SealGateTests(unittest.TestCase):
    def test_the_dataset_wrapper_refuses_signals_whose_seal_is_not_the_pinned_one(self):
        c, f = contract(), forces()
        fresh_loaded(self)  # a cached real table would otherwise answer without checking the seal
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'signals').mkdir()
            (Path(tmp)/'seal.json').write_text(json.dumps({**f.SEAL_RECORD, 'fingerprint': '0'*64}))
            with patch.object(f, 'SEALED', Path(tmp)):
                with self.assertRaisesRegex(ValueError, 'not the pinned sealed one'):
                    c.JoinedDataset(FakeBase())

    def test_the_dataset_wrapper_refuses_constants_that_differ_from_the_pin(self):
        c, f = contract(), forces()
        if not mounted():
            self.skipTest('sealed G1 dataset not mounted')
        fresh_loaded(self)
        with patch.object(f, 'CONSTANTS_SHA256', '0'*64):
            with self.assertRaisesRegex(ValueError, 'differ from the pinned'):
                c.JoinedDataset(FakeBase())


class RealJoinTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('sealed G1 dataset not mounted')
        self.c, self.f = contract(), forces()
        use_planned(self, self.c)

    def test_joined_rows_equal_the_sealed_state_plus_the_signals_for_real_frames(self):
        c, f = self.c, self.f
        if not (c.DATA/'data').exists():
            self.skipTest('sealed G1 dataset not mounted')
        raw = real_derived()[0]['values']
        k = f.constants()
        for episode, frame in ((0, 0), (0, 451), (1350, 200), (2699, 17), (2700, 0), (2999, 451)):
            sealed = sealed_state(c, episode, frame)
            joined = c.joined_state(torch.from_numpy(sealed), episode, frame).numpy()
            extras = ((np.clip(raw[episode, frame], k['low'], k['high'])-k['mean'])/k['std']).astype(np.float32)
            self.assertEqual(joined.tobytes(), np.concatenate([sealed, extras]).tobytes(), (episode, frame))

    def test_every_training_row_has_a_signals_row(self):
        c, f = self.c, self.f
        rows, starts = c.indices(metadata())
        extras = f.expected_extras(rows[::1009]+[rows[-1]])
        self.assertTrue(np.isfinite(extras).all())
        self.assertEqual(len(rows), c.TRAINING_ROWS)
        self.assertEqual(len(starts), c.COMPLETE_STARTS)

    def test_the_signals_split_is_the_plans_split(self):
        c = self.c
        self.assertEqual(real_derived()[0]['train'].tolist(), [i in c.TRAIN for i in range(c.EPISODES)])


if __name__ == '__main__':
    unittest.main()
