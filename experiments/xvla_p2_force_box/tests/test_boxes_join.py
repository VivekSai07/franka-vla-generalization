"""The sealed boxes and the loader: the sidecar's `target_box` joined and drawn by explicit (episode, frame) keys.

Also home of `mounted()`, the skip condition the other test modules import.
"""
import importlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

FRAMES = 452


def forces():
    return importlib.import_module('experiments.xvla_p2_force.forces')


def contract():
    return importlib.import_module('experiments.xvla_p2_force_box.contract')


def boxes():
    return importlib.import_module('experiments.xvla_p2_force_box.boxes')


def overlay():
    return importlib.import_module('experiments.xvla_p2_force_box.overlay')


def mounted():
    return (forces().SIDECAR/'seal.json').is_file()


def copy_sidecar(tmp):
    root = Path(tmp)/'sidecar'
    shutil.copytree(forces().SIDECAR, root)
    return root


def stored_boxes(episode):
    """The sidecar file's own `target_box`, read straight from the npz, independent of the package."""
    with np.load(forces().SIDECAR/'episodes'/f'episode_{episode:03d}.npz', allow_pickle=False) as stored:
        return stored['target_box'].copy()


class FakeBase:
    """A dataset whose rows carry only what the join reads, at the sealed positions, with plain grey frames."""

    def __init__(self, shift=0, episode_shift=0, width=9, grey=0.5):
        self.shift, self.episode_shift, self.width, self.grey = shift, episode_shift, width, grey
        self.meta = 'meta'

    def __len__(self):
        return 540*FRAMES

    def __getitem__(self, i):
        return {'episode_index': torch.tensor(i//FRAMES+self.episode_shift), 'frame_index': torch.tensor(i % FRAMES+self.shift),
                'observation.state': torch.full((self.width,), float(i % 7)), 'task': 'x',
                'observation.images.image': torch.full((3, 512, 512), self.grey),
                'observation.images.image2': torch.full((3, 512, 512), self.grey)}


class BoxTableTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.b, self.f = boxes(), forces()

    def test_the_table_is_every_sealed_frame_float64_two_cameras_of_five(self):
        table = self.b.table()
        self.assertEqual((table.dtype, table.shape), (np.float64, (540, 452, 2, 5)))
        self.assertFalse(table.flags.writeable)
        self.assertTrue(np.isfinite(table).all())
        self.assertTrue(((table >= 0) & (table <= 1)).all())

    def test_the_box_of_a_frame_is_the_sidecar_files_row_byte_for_byte(self):
        for episode, frame in ((0, 0), (0, 451), (18, 120), (271, 300), (539, 451)):
            self.assertEqual(self.b.target_box(episode, frame).tobytes(), stored_boxes(episode)[frame].tobytes(), (episode, frame))

    def test_target_box_returns_a_copy(self):
        box = self.b.target_box(0, 0)
        box[:] = -1
        self.assertGreaterEqual(float(self.b.target_box(0, 0).min()), 0)

    def test_expected_boxes_follow_dataset_row_indices(self):
        rows = [0, 421, 452*18+120, 540*452-1]
        expected = self.b.expected_boxes(rows)
        self.assertEqual(expected.shape, (4, 2, 5))
        for k, row in enumerate(rows):
            self.assertEqual(expected[k].tobytes(), self.b.target_box(row//FRAMES, row % FRAMES).tobytes())

    def test_indices_outside_the_sealed_grid_are_refused(self):
        for episode, frame in ((-1, 0), (540, 0), (0, -1), (0, 452), (True, 0), (0.0, 0), ('0', 0)):
            with self.subTest(index=(episode, frame)), self.assertRaises(ValueError):
                self.b.target_box(episode, frame)
        with self.assertRaises(ValueError):
            self.b.expected_boxes([540*452])

    def test_in_frame_is_one_on_every_training_row_in_both_cameras(self):
        train = self.b.table()[:480]
        self.assertEqual(float(train[..., 4].min()), 1.0)
        self.assertEqual(float(train[..., 4].max()), 1.0)

    def test_a_training_row_that_is_not_in_frame_is_refused_when_the_table_is_built(self):
        raw = self.b.load_raw(verify=False)
        raw[3, 10, 1, 4] = 0.0
        with self.assertRaises(ValueError):
            self.b.check_training_in_frame(raw)

    def test_the_table_is_built_through_the_in_frame_check(self):
        self.b._loaded.cache_clear()
        self.addCleanup(self.b._loaded.cache_clear)
        with patch.object(self.b, 'check_training_in_frame', side_effect=ValueError('not in frame')), \
                self.assertRaisesRegex(ValueError, 'not in frame'):
            self.b.table()

    def test_the_pinned_digest_matches_the_table(self):
        self.assertEqual(self.b.table_digest(self.b.table()), self.b.TABLE_SHA256)

    def test_a_table_that_differs_from_the_pinned_digest_is_refused(self):
        self.b._loaded.cache_clear()
        self.addCleanup(self.b._loaded.cache_clear)
        with patch.object(self.b, 'TABLE_SHA256', '0'*64), self.assertRaises(ValueError):
            self.b.table()

    def test_a_tampered_sidecar_is_refused_before_any_row_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            path = root/'episodes'/'episode_007.npz'
            with np.load(path, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays['target_box'] = arrays['target_box'].copy()
            arrays['target_box'][5, 0, 0] += 1e-6
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                self.b.load_raw(root)

    def test_the_seal_is_verified_before_the_episode_files_are_opened(self):
        with patch.object(self.f, 'verify_sidecar', side_effect=ValueError('seal does not verify')), \
                patch.object(self.b.np, 'load', side_effect=AssertionError('an episode was read before the seal')):
            with self.assertRaises(ValueError):
                self.b.load_raw()

    def test_missing_or_extra_episode_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            (root/'episodes'/'episode_100.npz').unlink()
            with patch.object(self.f, 'verify_sidecar', return_value=self.f.SIDECAR_RECORD), self.assertRaises(ValueError):
                self.b.load_raw(root)

    def test_the_binding_records_the_table_digest(self):
        self.assertEqual(self.b.binding(), {'target_box_sha256': self.b.TABLE_SHA256})

    def test_labels_are_the_sidecars_object_and_segment_of_every_episode(self):
        objects, segments = self.b.labels()
        self.assertEqual((len(objects), segments.shape, segments.dtype), (540, (540, 452), np.int8))
        self.assertEqual(sorted(set(objects)), ['bottle', 'bread', 'can', 'cereal', 'lemon', 'milk'])
        with np.load(forces().SIDECAR/'episodes'/'episode_005.npz', allow_pickle=False) as stored:
            self.assertEqual(objects[5], str(stored['object']))
            self.assertEqual(segments[5].tolist(), stored['segment'].tolist())
        self.assertTrue(((segments >= 0) & (segments <= 3)).all())
        self.assertTrue((np.diff(segments, axis=1) >= 0).all())

    def test_the_first_frame_of_a_segment_is_found_or_refused(self):
        b = self.b
        segments = np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int8)
        self.assertEqual([b.first_frame(segments, s) for s in (0, 1, 2, 3)], [0, 2, 4, 6])
        with self.assertRaises(ValueError):
            b.first_frame(np.array([0, 0, 2], dtype=np.int8), 1)


class JoinTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c, self.f, self.b, self.o = contract(), forces(), boxes(), overlay()

    def test_the_policy_state_is_still_17_values_the_nine_joints_then_the_eight_extras(self):
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

    def test_joined_state_refuses_a_state_that_is_not_nine_finite_float32_values(self):
        c = self.c
        for bad in (torch.zeros(17), torch.zeros(8), torch.zeros(9, dtype=torch.float64),
                    torch.full((9,), float('nan')), torch.zeros(1, 9)):
            with self.assertRaises(ValueError):
                c.joined_state(bad, 0, 0)

    def test_the_wrapper_joins_the_state_and_draws_the_frames_by_episode_and_frame(self):
        c, f, b, o = self.c, self.f, self.b, self.o
        ds = c.JoinedDataset(FakeBase())
        self.assertEqual(len(ds), 540*FRAMES)
        self.assertEqual(ds.meta, 'meta')  # everything else is delegated
        for index in (0, 451, 452, 452*18+120, 203_039, 244_079):
            row = ds[index]
            episode, frame = divmod(index, FRAMES)
            self.assertEqual(tuple(row['observation.state'].shape), (17,))
            self.assertEqual(row['observation.state'][9:].numpy().tobytes(), f.extra_state(episode, frame).tobytes())
            self.assertEqual(row['observation.state'][:9].tolist(), [float(index % 7)]*9)
            undrawn = FakeBase()[index]
            expected = o.draw_frame(undrawn, b.target_box(episode, frame))
            for key in c.CAMERAS:
                self.assertTrue(torch.equal(row[key], expected[key]), (index, key))
            self.assertFalse(torch.equal(row[c.CAMERAS[0]], undrawn[c.CAMERAS[0]]))

    def test_the_two_cameras_carry_their_own_box(self):
        c, b = self.c, self.b
        index = 452*18+120
        row = c.JoinedDataset(FakeBase())[index]
        front, gripper = b.target_box(18, 120)
        self.assertTrue(torch.equal(row[c.CAMERAS[0]], self.o.draw(FakeBase()[index][c.CAMERAS[0]], front)))
        self.assertTrue(torch.equal(row[c.CAMERAS[1]], self.o.draw(FakeBase()[index][c.CAMERAS[1]], gripper)))

    def test_the_overlay_goes_through_one_swappable_hook(self):
        c = self.c
        calls = []

        def identity(row, box):
            calls.append((row['episode_index'], box.shape))
            return row

        with patch.object(c, 'OVERLAY', identity):
            row = c.JoinedDataset(FakeBase())[600]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], (2, 5))
        self.assertTrue(torch.equal(row[c.CAMERAS[0]], FakeBase()[600][c.CAMERAS[0]]))
        self.assertIs(c.OVERLAY, self.o.draw_frame)  # restored, and the default is the real overlay

    def test_the_wrapper_refuses_shifted_or_foreign_rows(self):
        c = self.c
        for kwargs in ({'shift': 1}, {'shift': -1}, {'episode_shift': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase(**kwargs))[600]
        with self.assertRaises(ValueError):  # a 17-value base state is already joined or from another dataset
            c.JoinedDataset(FakeBase(width=17))[0]

    def test_the_wrapper_refuses_a_sidecar_that_does_not_verify(self):
        c, f, b = self.c, self.f, self.b
        for module in (f, b):
            module._loaded.cache_clear()
            self.addCleanup(module._loaded.cache_clear)
        with patch.object(f, 'verify_sidecar', side_effect=ValueError('seal does not verify')):
            with self.assertRaises(ValueError):
                c.JoinedDataset(FakeBase())

    def test_the_wrapper_verifies_the_box_table_too(self):
        c, b = self.c, self.b
        b._loaded.cache_clear()
        self.addCleanup(b._loaded.cache_clear)
        with patch.object(b, 'TABLE_SHA256', '0'*64), self.assertRaises(ValueError):
            c.JoinedDataset(FakeBase())

    def test_every_training_row_has_a_sidecar_box(self):
        c, b = self.c, self.b
        from test_contract import metadata
        if not (c.PARENT/'plan.json').exists():
            self.skipTest('Props P2 parent not mounted')
        rows, starts = c.indices(metadata())
        self.assertEqual(len(rows), 480*FRAMES)
        expected = b.expected_boxes(rows)
        self.assertEqual(expected.shape, (480*FRAMES, 2, 5))
        self.assertEqual(len(starts), c.COMPLETE_STARTS)

    def test_the_sidecar_split_is_the_plans_split(self):
        c, f = self.c, self.f
        self.assertEqual(f.load_raw(verify=False)['train'].tolist(), [i in c.TRAIN for i in range(540)])


if __name__ == '__main__':
    unittest.main()
