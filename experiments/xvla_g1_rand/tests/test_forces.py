"""The 8 extra state values: the sealed G1 signals, the clip and the standardisation. Pure numpy; admission image.

Structure, split, seal and constants are tested on a small fake sealed tree built in a temp directory (the module's size
constants, `SEALED` and `SEAL_RECORD` patched), so those tests never need the real data. The real 3,000-episode signals are
tested separately and skipped when the sealed dataset is not mounted.
"""
import hashlib
import functools
import importlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SMALL_EPISODES, SMALL_TRAIN, SMALL_FRAMES = 12, 10, 24
REAL_EPISODES, REAL_TRAIN, REAL_FRAMES = 3000, 2700, 452
_REAL = {}


def forces():
    return importlib.import_module('experiments.xvla_g1_rand.forces')


def mounted():
    f = forces()
    return (f.SEALED/'seal.json').is_file() and (f.SEALED/'signals').is_dir()


def real_derived():
    """(raw, constants) derived once per process from the real sealed signals, seal verified."""
    if 'derived' not in _REAL:
        _REAL['derived'] = forces().derived()
    return _REAL['derived']


def synthetic_table():
    """A (3000, 452, 8) float32 stand-in for the standardised extras; the validation episodes are huge so a leak shows."""
    if 'table' not in _REAL:
        table = np.random.default_rng(1).normal(size=(REAL_EPISODES, REAL_FRAMES, 8)).astype(np.float32)
        table[REAL_TRAIN:] *= 1e3
        table.flags.writeable = False
        _REAL['table'] = table
    return _REAL['table']


def use_synthetic_table(case):
    """Make `forces.table()` (hence extra_state, expected_extras and JoinedDataset) independent of the real signals."""
    patcher = patch.object(forces(), 'table', return_value=synthetic_table())
    patcher.start()
    case.addCleanup(patcher.stop)


def fresh_loaded(case):
    """A private `forces._loaded` cache: the test neither reads nor wipes the one that holds the real signals."""
    f = forces()
    patcher = patch.object(f, '_loaded', functools.lru_cache(maxsize=1)(f._loaded.__wrapped__))
    patcher.start()
    case.addCleanup(patcher.stop)


def fake_arrays(index, split, frames=SMALL_FRAMES):
    rng = np.random.default_rng(1000+index)
    big = split == 'validation'  # extremes that must never reach the constants
    return {'episode_index': np.array(index), 'frame_index': np.arange(frames),
            'wrist_wrench': np.full((frames, 6), 1e6) if big else rng.normal(index, 5., (frames, 6)),
            'finger_force': np.full((frames, 2), 1e6) if big else rng.normal(-index, 3., (frames, 2)),
            'target_box': np.zeros((frames, 2, 5)), 'props_box': np.zeros((frames, 1, 2, 5)),
            'prop_names': np.array(['lemon']), 'segment': np.zeros(frames, dtype=np.int8),
            'phase_index': np.zeros(frames, dtype=np.int64), 'object': np.array('lemon'), 'split': np.array(split)}


def build_fake(root):
    (root/'signals').mkdir(parents=True)
    for index in range(SMALL_EPISODES):
        np.savez(root/'signals'/f'episode_{index:04d}.npz',
                 **fake_arrays(index, 'train' if index < SMALL_TRAIN else 'validation'))
    seal = {'fingerprint': 'f'*64, 'plan_sha256': 'a'*64, 'episodes': SMALL_EPISODES}
    (root/'seal.json').write_text(json.dumps(seal))
    return seal


def rewrite_episode(root, index, **changes):
    path = root/'signals'/f'episode_{index:04d}.npz'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays.update(changes)
    path.unlink()
    np.savez(path, **arrays)


class FakeSealedCase(unittest.TestCase):
    """`forces` pointed at a small fake sealed signals tree for the duration of one test."""

    def setUp(self):
        self.f = f = forces()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'sealed'
        self.seal = build_fake(self.root)
        fresh_loaded(self)
        patcher = patch.multiple(f, EPISODES=SMALL_EPISODES, TRAIN_EPISODES=SMALL_TRAIN, FRAMES=SMALL_FRAMES,
                                 SEALED=self.root, SEAL_RECORD=self.seal)
        patcher.start()
        self.addCleanup(patcher.stop)

    def pin(self):
        """Pin the constants derived from the fake tree, as `CONSTANTS_SHA256` pins the real ones."""
        digest = self.f.constants_digest(self.f.derived(self.root)[1])
        patcher = patch.object(self.f, 'CONSTANTS_SHA256', digest)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.f._loaded.cache_clear()
        return digest

    def train_rows(self):
        return np.concatenate([np.concatenate([a['wrist_wrench'], a['finger_force']], axis=1)
                               for a in (fake_arrays(i, 'train') for i in range(SMALL_TRAIN))])


class FakeJoinTests(FakeSealedCase):
    """Rows are joined by explicit (episode, frame) keys; a missing, duplicated, unexpected or shifted row is an error."""

    def test_every_episode_and_frame_is_present_once(self):
        raw = self.f.load_raw(self.root)
        self.assertEqual(raw['values'].shape, (SMALL_EPISODES, SMALL_FRAMES, 8))
        self.assertEqual(raw['values'].dtype, np.float64)
        self.assertTrue(np.isfinite(raw['values']).all())

    def test_the_split_is_the_first_ten_train_and_the_last_two_validation(self):
        self.assertEqual(self.f.load_raw(self.root)['train'].tolist(), [True]*SMALL_TRAIN+[False]*2)

    def test_columns_are_wrench_then_fingers_in_the_sealed_order(self):
        f = self.f
        raw = f.load_raw(self.root)
        self.assertEqual(f.EXTRA_NAMES, ('fx', 'fy', 'fz', 'tx', 'ty', 'tz', 'finger_left', 'finger_right'))
        for index in (0, 7):
            arrays = fake_arrays(index, 'train')
            np.testing.assert_array_equal(raw['values'][index, :, :6], arrays['wrist_wrench'])
            np.testing.assert_array_equal(raw['values'][index, :, 6:], arrays['finger_force'])
        self.assertFalse(np.array_equal(raw['values'][0, :, :2], raw['values'][0, :, 6:]))

    def test_a_missing_episode_is_refused(self):
        (self.root/'signals/episode_0005.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'missing, duplicated or unexpected'):
            self.f.load_raw(self.root, verify=False)

    def test_an_unexpected_extra_episode_is_refused(self):
        shutil.copy(self.root/'signals/episode_0011.npz', self.root/f'signals/episode_{SMALL_EPISODES:04d}.npz')
        with self.assertRaisesRegex(ValueError, 'missing, duplicated or unexpected'):
            self.f.load_raw(self.root, verify=False)

    def test_a_duplicated_episode_is_refused(self):
        shutil.copy(self.root/'signals/episode_0004.npz', self.root/'signals/episode_0005.npz')  # 0005 now says episode 4
        with self.assertRaisesRegex(ValueError, 'file says episode 4'):
            self.f.load_raw(self.root, verify=False)

    def test_a_shifted_frame_index_is_refused(self):
        rewrite_episode(self.root, 9, frame_index=np.arange(1, SMALL_FRAMES+1))
        with self.assertRaisesRegex(ValueError, 'frame_index'):
            self.f.load_raw(self.root, verify=False)

    def test_a_short_episode_is_refused(self):
        with np.load(self.root/'signals/episode_0009.npz', allow_pickle=False) as stored:
            short = {key: stored[key][:SMALL_FRAMES-1] for key in ('wrist_wrench', 'finger_force', 'frame_index')}
        rewrite_episode(self.root, 9, **short)
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.f.load_raw(self.root, verify=False)

    def test_a_nonfinite_value_is_refused(self):
        wrench = fake_arrays(3, 'train')['wrist_wrench']
        wrench[4, 2] = np.nan
        rewrite_episode(self.root, 3, wrist_wrench=wrench)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.f.load_raw(self.root, verify=False)

    def test_a_wrong_split_is_refused_in_both_directions(self):
        for index, split in ((3, 'validation'), (SMALL_TRAIN, 'train'), (0, 'test')):
            with self.subTest(index=index, split=split):
                original = fake_arrays(index, 'train' if index < SMALL_TRAIN else 'validation')
                rewrite_episode(self.root, index, split=np.array(split))
                with self.assertRaisesRegex(ValueError, 'split'):
                    self.f.load_raw(self.root, verify=False)
                rewrite_episode(self.root, index, split=original['split'])
        self.f.load_raw(self.root, verify=False)  # restored


class FakeSealTests(FakeSealedCase):
    def test_an_untouched_tree_verifies(self):
        self.assertEqual(self.f.verify_seal(self.root), self.seal)
        self.f.load_raw(self.root)

    def test_a_seal_that_is_not_the_pinned_record_is_refused_field_by_field(self):
        f = self.f
        for key in self.seal:
            changed = {**self.seal, key: ('0'*64 if isinstance(self.seal[key], str) else self.seal[key]+1)}
            (self.root/'seal.json').write_text(json.dumps(changed))
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'not the pinned sealed one'):
                    f.verify_seal(self.root)
                with self.assertRaises(ValueError):
                    f.load_raw(self.root)
        (self.root/'seal.json').write_text(json.dumps({k: v for k, v in self.seal.items() if k != 'episodes'}))
        with self.assertRaises(ValueError):
            f.verify_seal(self.root)
        (self.root/'seal.json').write_text(json.dumps({**self.seal, 'extra': 1}))
        with self.assertRaises(ValueError):
            f.verify_seal(self.root)

    def test_a_pinned_record_that_differs_from_the_tree_is_refused_by_every_entry_point(self):
        f = self.f
        self.pin()
        f.table()  # an unpatched record verifies
        f._loaded.cache_clear()
        with patch.object(f, 'SEAL_RECORD', {**self.seal, 'fingerprint': '0'*64}):
            for name, call in (('table', f.table), ('constants', f.constants), ('binding', f.binding),
                               ('derived', lambda: f.derived(self.root))):
                with self.subTest(call=name), self.assertRaises(ValueError):
                    call()

    def test_the_seal_is_checked_before_a_single_signal_file_is_read(self):
        (self.root/'seal.json').write_text(json.dumps({**self.seal, 'fingerprint': '0'*64}))
        with patch.object(np, 'load', side_effect=AssertionError('signals read before the seal was verified')):
            with self.assertRaisesRegex(ValueError, 'not the pinned sealed one'):
                self.f.load_raw(self.root)

    def test_binding_records_the_seal_its_file_digest_and_the_constants_digest(self):
        digest = self.pin()
        binding = self.f.binding()
        self.assertEqual(binding['seal'], self.seal)
        self.assertEqual(binding['seal_sha256'], hashlib.sha256((self.root/'seal.json').read_bytes()).hexdigest())
        self.assertEqual(binding['constants_sha256'], digest)


class FakeConstantsTests(FakeSealedCase):
    def test_clip_bounds_are_the_percentiles_of_the_training_rows_only(self):
        f = self.f
        raw, constants = f.derived(self.root)
        rows = self.train_rows()
        self.assertEqual(rows.shape, (SMALL_TRAIN*SMALL_FRAMES, 8))
        np.testing.assert_array_equal(constants['low'], np.percentile(rows, 0.1, axis=0))
        np.testing.assert_array_equal(constants['high'], np.percentile(rows, 99.9, axis=0))
        self.assertLess(constants['high'].max(), 1e3)  # the 1e6 validation rows are not in the bounds
        everything = raw['values'].reshape(-1, 8)
        self.assertGreater(np.percentile(everything, 99.9, axis=0).min(), 1e5)  # premise: including them would move them

    def test_mean_and_std_are_of_the_clipped_training_rows_only(self):
        constants = self.f.derived(self.root)[1]
        rows = np.clip(self.train_rows(), constants['low'], constants['high'])
        np.testing.assert_allclose(constants['mean'], rows.mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(constants['std'], rows.std(0), rtol=0, atol=1e-12)

    def test_the_validation_episodes_do_not_move_the_constants(self):
        f = self.f
        before = f.constants_digest(f.derived(self.root)[1])
        for index in (SMALL_TRAIN, SMALL_TRAIN+1):
            rewrite_episode(self.root, index, wrist_wrench=np.full((SMALL_FRAMES, 6), -7e5),
                            finger_force=np.full((SMALL_FRAMES, 2), 3.))
        self.assertEqual(before, f.constants_digest(f.derived(self.root)[1]))
        rewrite_episode(self.root, 2, wrist_wrench=np.full((SMALL_FRAMES, 6), 5e4)+np.arange(SMALL_FRAMES)[:, None])
        self.assertNotEqual(before, f.constants_digest(f.derived(self.root)[1]))  # a training episode does move them

    def test_the_constants_derive_deterministically(self):
        f = self.f
        self.assertEqual(f.constants_digest(f.derived(self.root)[1]), f.constants_digest(f.derived(self.root)[1]))

    def test_a_digest_mismatch_is_refused_and_the_right_digest_passes(self):
        f = self.f
        with patch.object(f, 'CONSTANTS_SHA256', '0'*64):
            f._loaded.cache_clear()
            for name, call in (('constants', f.constants), ('table', f.table), ('extra_state', lambda: f.extra_state(0, 0)),
                               ('binding', f.binding)):
                with self.subTest(call=name), self.assertRaisesRegex(ValueError, 'differ from the pinned'):
                    call()
        digest = self.pin()
        self.assertEqual(f.constants_digest(f.constants()), digest)

    def test_the_training_rows_standardise_to_zero_mean_and_unit_spread(self):
        self.pin()
        f = self.f
        table = f.table()
        self.assertEqual((table.shape, table.dtype), ((SMALL_EPISODES, SMALL_FRAMES, 8), np.float32))
        rows = table[:SMALL_TRAIN].reshape(-1, 8).astype(np.float64)
        np.testing.assert_allclose(rows.mean(0), 0., atol=1e-5)
        np.testing.assert_allclose(rows.std(0), 1., atol=1e-5)
        k = f.constants()  # validation rows are clipped to the training range, never beyond it
        validation = table[SMALL_TRAIN:].reshape(-1, 8)
        np.testing.assert_array_equal(validation, np.broadcast_to(((k['high']-k['mean'])/k['std']).astype(np.float32),
                                                                  validation.shape))
        self.assertFalse(table.flags.writeable)

    def test_extra_state_is_the_clipped_standardised_signal_row(self):
        f = self.f
        self.pin()
        raw = f.load_raw(self.root)['values']
        k = f.constants()
        for episode, frame in ((0, 0), (0, SMALL_FRAMES-1), (6, 7), (SMALL_TRAIN-1, 17), (SMALL_TRAIN, 0),
                               (SMALL_EPISODES-1, SMALL_FRAMES-1)):
            expected = ((np.clip(raw[episode, frame], k['low'], k['high'])-k['mean'])/k['std']).astype(np.float32)
            got = f.extra_state(episode, frame)
            self.assertEqual((got.dtype, got.shape), (np.float32, (8,)))
            self.assertEqual(got.tobytes(), expected.tobytes(), (episode, frame))
        got[0] = 99.  # a copy: the shared table cannot be written through it
        self.assertNotEqual(f.extra_state(SMALL_EPISODES-1, SMALL_FRAMES-1)[0], 99.)

    def test_extra_state_refuses_indices_outside_the_signals(self):
        f = self.f
        self.pin()
        for episode, frame in ((SMALL_EPISODES, 0), (-1, 0), (0, SMALL_FRAMES), (0, -1)):
            with self.subTest(episode=episode, frame=frame), self.assertRaises(ValueError):
                f.extra_state(episode, frame)
        for bad in (1.0, True, '0', None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    f.extra_state(bad, 0)
                with self.assertRaises(ValueError):
                    f.extra_state(0, bad)

    def test_expected_extras_maps_dataset_rows_to_episode_and_frame_and_refuses_the_rest(self):
        f = self.f
        self.pin()
        last = SMALL_EPISODES*SMALL_FRAMES-1
        rows = [0, SMALL_FRAMES-1, SMALL_FRAMES, 5*SMALL_FRAMES+3, last]
        got = f.expected_extras(rows)
        self.assertEqual(got.shape, (5, 8))
        for k, row in enumerate(rows):
            np.testing.assert_array_equal(got[k], f.extra_state(*divmod(row, SMALL_FRAMES)))
        for bad in (last+1, -1):
            with self.assertRaises(ValueError):
                f.expected_extras([0, bad])

    def test_clipping_rates_are_reported_per_split_and_never_move_the_bounds(self):
        f = self.f
        self.pin()
        bounds = {key: value.copy() for key, value in f.constants().items()}
        rates = f.clipping_rates()
        self.assertEqual(set(rates), {'train', 'validation'})
        self.assertEqual(rates['train']['rows'], SMALL_TRAIN*SMALL_FRAMES)
        self.assertEqual(rates['validation']['rows'], 2*SMALL_FRAMES)
        self.assertEqual(rates['validation']['fraction_clipped'], [1.0]*8)  # every validation row is beyond the bound
        self.assertTrue(all(0. <= x <= 0.02 for x in rates['train']['fraction_clipped']))
        for key, value in bounds.items():
            np.testing.assert_array_equal(f.constants()[key], value)


class ConstantsTests(unittest.TestCase):
    def setUp(self):
        self.f = forces()

    def synthetic(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(0., 5., (6, 20, 8))
        raw[4:] = 1e6  # the validation episodes hold extremes that must not reach the constants
        train = np.array([True]*4+[False]*2)
        low, high = np.full(8, -3.), np.full(8, 4.)
        return raw, train, low, high

    def test_mean_and_std_are_of_the_clipped_training_rows_only(self):
        raw, train, low, high = self.synthetic()
        constants = self.f.derive_constants(raw, train, low, high)
        rows = np.clip(raw[:4].reshape(-1, 8), low, high)
        self.assertEqual(len(rows), 4*20)  # the rows counted: training episodes only
        np.testing.assert_allclose(constants['mean'], rows.mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(constants['std'], rows.std(0), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(constants['low'], low)
        np.testing.assert_array_equal(constants['high'], high)

    def test_clip_bounds_use_the_training_mask_and_nothing_else(self):
        raw, train, _, _ = self.synthetic()
        low, high = self.f.clip_bounds(raw, train)
        np.testing.assert_array_equal(low, np.percentile(raw[:4].reshape(-1, 8), 0.1, axis=0))
        np.testing.assert_array_equal(high, np.percentile(raw[:4].reshape(-1, 8), 99.9, axis=0))
        self.assertLess(high.max(), 100.)
        flipped = self.f.clip_bounds(raw, ~train)  # the mask, not the position, selects the rows
        self.assertTrue((flipped[0] == 1e6).all())
        with self.assertRaises(ValueError):
            self.f.clip_bounds(raw, np.zeros(6, dtype=bool))

    def test_mean_and_std_are_computed_by_exactly_rounded_sums(self):
        """Admission and the robosuite image must derive the same bytes, whatever numpy's reduction order."""
        raw, train, low, high = self.synthetic()
        constants = self.f.derive_constants(raw, train, low, high)
        rows = np.clip(raw[:4].reshape(-1, 8), low, high)
        for j in range(8):
            mean = math.fsum(rows[:, j].tolist())/len(rows)
            std = math.sqrt(math.fsum(((rows[:, j]-mean)**2).tolist())/len(rows))
            self.assertEqual(float(constants['mean'][j]), mean)
            self.assertEqual(float(constants['std'][j]), std)

    def test_the_constants_do_not_depend_on_the_order_of_the_rows(self):
        raw, train, low, high = self.synthetic()
        order = np.random.default_rng(3).permutation(4*20)
        shuffled = raw.copy()
        shuffled[:4] = raw[:4].reshape(-1, 8)[order].reshape(4, 20, 8)
        a = self.f.derive_constants(raw, train, low, high)
        b = self.f.derive_constants(shuffled, train, low, high)
        self.assertEqual(self.f.constants_digest(a), self.f.constants_digest(b))

    def test_the_digest_changes_with_any_constant_by_one_ulp(self):
        raw, train, low, high = self.synthetic()
        constants = self.f.derive_constants(raw, train, low, high)
        base = self.f.constants_digest(constants)
        for key in ('low', 'high', 'mean', 'std'):
            changed = {k: v.copy() for k, v in constants.items()}
            changed[key][3] = np.nextafter(changed[key][3], np.inf)
            self.assertNotEqual(self.f.constants_digest(changed), base, key)

    def test_a_constant_column_is_refused(self):
        raw, train, low, high = self.synthetic()
        raw[:4, :, 3] = 1.
        with self.assertRaises(ValueError):
            self.f.derive_constants(raw, train, low, high)

    def test_inverted_or_nonfinite_bounds_are_refused(self):
        raw, train, low, high = self.synthetic()
        for bad_low, bad_high in ((high, low), (low, np.where(np.arange(8) == 2, np.nan, high))):
            with self.assertRaises(ValueError):
                self.f.derive_constants(raw, train, bad_low, bad_high)

    def test_a_mask_or_array_of_the_wrong_shape_is_refused(self):
        raw, train, low, high = self.synthetic()
        for bad_raw, bad_train in ((raw, train[:5]), (raw[..., :7], train), (raw, np.zeros(6, dtype=bool))):
            with self.assertRaises(ValueError):
                self.f.derive_constants(bad_raw, bad_train, low, high)

    def test_standardise_clips_first_then_standardises_and_returns_float32(self):
        raw, train, low, high = self.synthetic()
        constants = self.f.derive_constants(raw, train, low, high)
        values = np.array([[100.]*8, [-100.]*8, [0.5]*8])
        out = self.f.standardise(values, constants)
        self.assertEqual(out.dtype, np.float32)
        expected = ((np.clip(values, low, high)-constants['mean'])/constants['std']).astype(np.float32)
        np.testing.assert_array_equal(out, expected)
        # Unclipped values would be far outside the clipped range's z-range.
        z_high = (high-constants['mean'])/constants['std']
        self.assertTrue((out[0] <= z_high.astype(np.float32)+1e-6).all())

    def test_standardise_refuses_nonfinite_or_wrong_width(self):
        raw, train, low, high = self.synthetic()
        constants = self.f.derive_constants(raw, train, low, high)
        for bad in (np.full((2, 8), np.nan), np.full((2, 8), np.inf), np.zeros((2, 7)), np.zeros((2, 9))):
            with self.assertRaises(ValueError):
                self.f.standardise(bad, constants)


class RealSealTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('sealed G1 dataset not mounted')
        self.f = forces()

    def test_the_pinned_record_is_the_sealed_one(self):
        f = self.f
        self.assertEqual(json.loads((f.SEALED/'seal.json').read_text()), f.SEAL_RECORD)
        self.assertEqual(f.SEAL_RECORD['fingerprint'], '845ac94063fd367cc315b8d287fc141df76ea171f9227f5794eef0bd8e4e161e')
        self.assertEqual(f.SEAL_RECORD['episodes'], REAL_EPISODES)
        self.assertEqual((f.EPISODES, f.TRAIN_EPISODES, f.FRAMES, f.N_EXTRA), (REAL_EPISODES, REAL_TRAIN, REAL_FRAMES, 8))

    def test_the_real_seal_verifies_and_a_pinned_record_that_differs_does_not(self):
        f = self.f
        self.assertEqual(f.verify_seal(), f.SEAL_RECORD)
        for key in ('fingerprint', 'provenance_sha256', 'plan_sha256', 'substitutions_sha256', 'image'):
            with patch.object(f, 'SEAL_RECORD', {**f.SEAL_RECORD, key: '0'*64}):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    f.verify_seal()


class RealSignalsTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('sealed G1 dataset not mounted')
        self.f = forces()

    def test_every_episode_and_frame_is_present_once_and_the_split_is_2700_to_300(self):
        raw = real_derived()[0]
        self.assertEqual(raw['values'].shape, (REAL_EPISODES, REAL_FRAMES, 8))
        self.assertEqual(raw['values'].dtype, np.float64)
        self.assertTrue(np.isfinite(raw['values']).all())
        self.assertEqual(int(raw['train'].sum())*REAL_FRAMES, 1_220_400)
        self.assertEqual(raw['train'].tolist(), [True]*REAL_TRAIN+[False]*(REAL_EPISODES-REAL_TRAIN))

    def test_columns_are_wrench_then_fingers_in_the_sealed_order(self):
        f = self.f
        raw = real_derived()[0]
        for index in (0, 1500, 2999):
            with np.load(f.SEALED/'signals'/f'episode_{index:04d}.npz', allow_pickle=False) as stored:
                wrench, finger = stored['wrist_wrench'], stored['finger_force']
            np.testing.assert_array_equal(raw['values'][index, :, :6], wrench)
            np.testing.assert_array_equal(raw['values'][index, :, 6:], finger)

    def test_the_derived_constants_digest_equals_the_pinned_one(self):
        f = self.f
        self.assertEqual(f.constants_digest(real_derived()[1]), f.CONSTANTS_SHA256)
        self.assertEqual(f.constants_digest(f.constants()), f.CONSTANTS_SHA256)
        changed = {key: value.copy() for key, value in f.constants().items()}
        changed['mean'][0] += 1e-9
        self.assertNotEqual(f.constants_digest(changed), f.CONSTANTS_SHA256)

    def test_constants_come_from_the_2700_training_episodes(self):
        f = self.f
        raw, constants = real_derived()
        training = raw['values'][:REAL_TRAIN].reshape(-1, 8)
        self.assertEqual(len(training), 1_220_400)
        np.testing.assert_array_equal(constants['low'], np.percentile(training, 0.1, axis=0))
        np.testing.assert_array_equal(constants['high'], np.percentile(training, 99.9, axis=0))
        rows = np.clip(training, constants['low'], constants['high'])
        np.testing.assert_allclose(constants['mean'], rows.mean(0), rtol=0, atol=1e-9)
        np.testing.assert_allclose(constants['std'], rows.std(0), rtol=0, atol=1e-9)
        everything = np.clip(raw['values'].reshape(-1, 8), constants['low'], constants['high'])
        self.assertFalse(np.allclose(constants['mean'], everything.mean(0), rtol=0, atol=1e-12))
        self.assertEqual(len(constants['mean']), len(f.EXTRA_NAMES))

    def test_constants_that_differ_from_the_pin_are_refused(self):
        f = self.f
        fresh_loaded(self)  # a cached real table would otherwise answer without checking the pin
        with patch.object(f, 'CONSTANTS_SHA256', '0'*64):
            with self.assertRaises(ValueError):
                f.constants()

    def test_the_training_rows_standardise_to_zero_mean_and_unit_spread(self):
        table = self.f.table()
        self.assertEqual((table.shape, table.dtype), ((REAL_EPISODES, REAL_FRAMES, 8), np.float32))
        rows = table[:REAL_TRAIN].reshape(-1, 8).astype(np.float64)
        np.testing.assert_allclose(rows.mean(0), 0., atol=1e-5)
        np.testing.assert_allclose(rows.std(0), 1., atol=1e-5)

    def test_extra_state_is_the_clipped_standardised_signal_row(self):
        f = self.f
        raw = real_derived()[0]['values']
        c = f.constants()
        for episode, frame in ((0, 0), (0, 451), (1350, 200), (2699, 17), (2700, 0), (2999, 451)):
            expected = ((np.clip(raw[episode, frame], c['low'], c['high'])-c['mean'])/c['std']).astype(np.float32)
            got = f.extra_state(episode, frame)
            self.assertEqual((got.dtype, got.shape), (np.float32, (8,)))
            self.assertEqual(got.tobytes(), expected.tobytes(), (episode, frame))

    def test_extra_state_refuses_indices_outside_the_signals(self):
        for episode, frame in ((REAL_EPISODES, 0), (-1, 0), (0, REAL_FRAMES), (0, -1)):
            with self.assertRaises(ValueError):
                self.f.extra_state(episode, frame)
        for bad in (1.0, True, '0'):
            with self.assertRaises(ValueError):
                self.f.extra_state(bad, 0)

    def test_expected_extras_maps_dataset_rows_to_episode_and_frame(self):
        f = self.f
        rows = [0, 451, 452, 1_220_399, 1_355_999]
        got = f.expected_extras(rows)
        self.assertEqual(got.shape, (5, 8))
        for k, row in enumerate(rows):
            np.testing.assert_array_equal(got[k], f.extra_state(row//REAL_FRAMES, row % REAL_FRAMES))
        with self.assertRaises(ValueError):
            f.expected_extras([1_356_000])

    def test_clipping_rates_are_about_two_in_a_thousand_on_training_rows(self):
        f = self.f
        raw, k = real_derived()[0], f.constants()
        rates = f.clipping_rates()
        for split, rows in (('train', raw['values'][raw['train']]), ('validation', raw['values'][~raw['train']])):
            rows = rows.reshape(-1, 8)
            self.assertEqual(rates[split]['fraction_clipped'], ((rows < k['low']).mean(0)+(rows > k['high']).mean(0)).tolist())
            self.assertEqual(rates[split]['rows'], len(rows))
        self.assertEqual(rates['train']['rows'], 1_220_400)
        self.assertEqual(rates['validation']['rows'], 135_600)
        self.assertTrue(all(0. <= x <= 0.0025 for x in rates['train']['fraction_clipped']))


if __name__ == '__main__':
    unittest.main()
