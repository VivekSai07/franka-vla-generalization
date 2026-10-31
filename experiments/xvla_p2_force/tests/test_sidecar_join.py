"""The sidecar join in the training contract: sealed state rows plus clipped, standardised S0 values, by explicit keys."""
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_contract import metadata
from test_forces import mounted

FRAMES = 452


def contract():
    return importlib.import_module('experiments.xvla_p2_force.contract')


def forces():
    return importlib.import_module('experiments.xvla_p2_force.forces')


def sealed_state(c, episode, frame):
    """The sealed dataset's `observation.state` (float32, 9 values) at (episode, frame), read from the parquet."""
    import pyarrow.parquet as pq
    root = c.DATA/'data'
    for path in sorted(root.rglob('*.parquet')):
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
        return 540*FRAMES

    def __getitem__(self, i):
        return {'episode_index': torch.tensor(i//FRAMES+self.episode_shift), 'frame_index': torch.tensor(i % FRAMES+self.shift),
                'observation.state': torch.full((self.width,), float(i % 7)), 'task': 'x'}


class JoinTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c, self.f = contract(), forces()

    def test_the_policy_state_is_17_values_the_nine_joints_then_the_eight_extras(self):
        c = self.c
        self.assertEqual(c.OBS['observation.state'], (17,))
        self.assertEqual(c.DATASET_OBS['observation.state'], (9,))
        self.assertEqual({k: v for k, v in c.OBS.items() if k != 'observation.state'},
                         {k: v for k, v in c.DATASET_OBS.items() if k != 'observation.state'})

    def test_joined_state_is_the_state_then_the_clipped_sidecar_values(self):
        c, f = self.c, self.f
        state = torch.arange(9, dtype=torch.float32)
        joined = c.joined_state(state, 5, 17)
        self.assertEqual((joined.dtype, tuple(joined.shape)), (torch.float32, (17,)))
        self.assertTrue(torch.equal(joined[:9], state))
        self.assertEqual(joined[9:].numpy().tobytes(), f.extra_state(5, 17).tobytes())

    def test_joined_rows_equal_the_sealed_state_plus_the_sidecar_for_real_frames(self):
        c, f = self.c, self.f
        if not (c.DATA/'data').exists():
            self.skipTest('Props P2 parent not mounted')
        raw = f.load_raw(verify=False)['values']
        k = f.constants()
        for episode, frame in ((0, 0), (0, 451), (271, 200)):
            sealed = sealed_state(c, episode, frame)
            joined = c.joined_state(torch.from_numpy(sealed), episode, frame).numpy()
            extras = ((np.clip(raw[episode, frame], k['low'], k['high'])-k['mean'])/k['std']).astype(np.float32)
            self.assertEqual(joined.tobytes(), np.concatenate([sealed, extras]).tobytes(), (episode, frame))

    def test_joined_state_refuses_a_state_that_is_not_nine_finite_float32_values(self):
        c = self.c
        for bad in (torch.zeros(17), torch.zeros(8), torch.zeros(9, dtype=torch.float64),
                    torch.full((9,), float('nan')), torch.zeros(1, 9)):
            with self.assertRaises(ValueError):
                c.joined_state(bad, 0, 0)

    def test_the_dataset_wrapper_joins_by_episode_and_frame(self):
        c, f = self.c, self.f
        ds = c.JoinedDataset(FakeBase())
        self.assertEqual(len(ds), 540*FRAMES)
        self.assertEqual(ds.meta, 'meta')  # everything else is delegated
        for index in (0, 451, 452, 203_039, 244_079):
            row = ds[index]
            self.assertEqual(tuple(row['observation.state'].shape), (17,))
            self.assertEqual(row['observation.state'][9:].numpy().tobytes(),
                             f.extra_state(index//FRAMES, index % FRAMES).tobytes())
            self.assertEqual(row['observation.state'][:9].tolist(), [float(index % 7)]*9)

    def test_the_dataset_wrapper_refuses_shifted_or_foreign_rows(self):
        c = self.c
        for kwargs in ({'shift': 1}, {'shift': -1}, {'episode_shift': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase(**kwargs))[600]
        with self.assertRaises(ValueError):  # a 17-value base state is already joined or from another dataset
            c.JoinedDataset(FakeBase(width=17))[0]

    def test_the_dataset_wrapper_refuses_a_sidecar_that_does_not_verify(self):
        c, f = self.c, self.f
        f._loaded.cache_clear()
        self.addCleanup(f._loaded.cache_clear)
        with patch.object(f, 'verify_sidecar', side_effect=ValueError('seal does not verify')):
            with self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase())

    def test_every_training_row_has_a_sidecar_row(self):
        c, f = self.c, self.f
        if not (c.PARENT/'plan.json').exists():
            self.skipTest('Props P2 parent not mounted')
        rows, starts = c.indices(metadata())
        self.assertEqual(len(rows), 480*FRAMES)
        extras = f.expected_extras(rows)
        self.assertEqual(extras.shape, (480*FRAMES, 8))
        self.assertTrue(np.isfinite(extras).all())
        self.assertEqual(len(starts), c.COMPLETE_STARTS)

    def test_the_sidecar_split_is_the_plans_split(self):
        c, f = self.c, self.f
        self.assertEqual(f.load_raw(verify=False)['train'].tolist(),
                         [i in c.TRAIN for i in range(540)])


if __name__ == '__main__':
    unittest.main()
