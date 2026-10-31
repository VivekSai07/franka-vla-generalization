"""The 8 extra state values: the S0 sidecar join, the clip and the standardisation. Pure numpy; admission image."""
import importlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def forces():
    return importlib.import_module('experiments.xvla_p2_force.forces')


def mounted():
    return (forces().SIDECAR/'seal.json').is_file()


def copy_sidecar(tmp):
    root = Path(tmp)/'sidecar'
    shutil.copytree(forces().SIDECAR, root)
    return root


def rewrite_episode(root, index, **changes):
    path = root/'episodes'/f'episode_{index:03d}.npz'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays.update(changes)
    path.unlink()
    np.savez(path, **arrays)


class SealTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.f = forces()

    def test_the_pinned_record_is_the_sealed_one(self):
        f = self.f
        self.assertEqual(json.loads((f.SIDECAR/'seal.json').read_text()), f.SIDECAR_RECORD)
        self.assertEqual(f.SIDECAR_RECORD['fingerprint'],
                         '7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c')
        self.assertEqual((f.SIDECAR_RECORD['episodes'], f.SIDECAR_RECORD['frames']), (540, 244_080))

    def test_the_real_sidecar_verifies(self):
        self.assertEqual(self.f.verify_sidecar()['fingerprint'], self.f.SIDECAR_RECORD['fingerprint'])

    def test_the_loader_refuses_a_sidecar_whose_seal_does_not_verify(self):
        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            f.load_raw(root)  # an untouched copy verifies and loads
            path = root/'episodes/episode_007.npz'
            data = bytearray(path.read_bytes())
            data[len(data)//2] ^= 1
            path.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                f.load_raw(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            seal = json.loads((root/'seal.json').read_text())
            (root/'seal.json').write_text(json.dumps({**seal, 'fingerprint': '0'*64}))
            with self.assertRaises(ValueError):
                f.load_raw(root)
        with tempfile.TemporaryDirectory() as tmp:  # report.json is inside the seal too
            root = copy_sidecar(tmp)
            report = json.loads((root/'report.json').read_text())
            report['clip_percentiles']['p99.9'][0] += 1.
            (root/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                f.load_raw(root)

    def test_a_sidecar_that_verifies_but_is_not_the_pinned_one_is_refused(self):
        """A genuine seal with another fingerprint is not the S0 data this run is bound to."""
        f = self.f
        for key in ('fingerprint', 'provenance_sha256', 'runtime_sha256', 'p2_seal'):
            with patch.object(f, 'SIDECAR_RECORD', {**f.SIDECAR_RECORD, key: '0'*64}):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    f.verify_sidecar()


class JoinTests(unittest.TestCase):
    """Rows are joined by explicit (episode, frame) keys; a missing, duplicated or shifted row is an error."""

    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.f = forces()

    def test_every_episode_and_frame_is_present_once(self):
        raw = self.f.load_raw()
        self.assertEqual(raw['values'].shape, (540, 452, 8))
        self.assertEqual(raw['values'].dtype, np.float64)
        self.assertTrue(np.isfinite(raw['values']).all())
        self.assertEqual(int(raw['train'].sum())*452, 480*452)  # 216,960 training rows
        self.assertEqual(raw['train'].tolist(), [True]*480+[False]*60)

    def test_columns_are_wrench_then_fingers_in_the_sealed_order(self):
        f = self.f
        raw = f.load_raw()
        with np.load(f.SIDECAR/'episodes/episode_000.npz', allow_pickle=False) as stored:
            wrench, finger = stored['wrist_wrench'], stored['finger_force']
        self.assertEqual(f.EXTRA_NAMES, ('fx', 'fy', 'fz', 'tx', 'ty', 'tz', 'finger_left', 'finger_right'))
        np.testing.assert_array_equal(raw['values'][0, :, :6], wrench)
        np.testing.assert_array_equal(raw['values'][0, :, 6:], finger)

    def test_a_missing_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            (root/'episodes/episode_200.npz').unlink()
            with self.assertRaises(ValueError):
                self.f.load_raw(root, verify=False)

    def test_an_unexpected_extra_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            shutil.copy(root/'episodes/episode_539.npz', root/'episodes/episode_540.npz')
            with self.assertRaises(ValueError):
                self.f.load_raw(root, verify=False)

    def test_a_duplicated_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            shutil.copy(root/'episodes/episode_004.npz', root/'episodes/episode_005.npz')  # 005 now says episode 4
            with self.assertRaises(ValueError):
                self.f.load_raw(root, verify=False)

    def test_a_shifted_frame_index_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            rewrite_episode(root, 9, frame_index=np.arange(1, 453))
            with self.assertRaises(ValueError):
                self.f.load_raw(root, verify=False)

    def test_a_short_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_sidecar(tmp)
            with np.load(root/'episodes/episode_009.npz', allow_pickle=False) as stored:
                short = {key: stored[key][:451] for key in ('wrist_wrench', 'finger_force', 'frame_index')}
            rewrite_episode(root, 9, **short)
            with self.assertRaises(ValueError):
                self.f.load_raw(root, verify=False)


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


class RealConstantsTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.f = forces()

    def test_clip_ranges_are_s0s_sealed_report_values(self):
        report = json.loads((self.f.SIDECAR/'report.json').read_text())['clip_percentiles']
        constants = self.f.constants()
        self.assertEqual(constants['low'].tolist(), report['p0.1'])
        self.assertEqual(constants['high'].tolist(), report['p99.9'])
        self.assertEqual(len(report['p0.1']), 8)

    def test_constants_come_from_the_480_training_episodes(self):
        f = self.f
        raw = f.load_raw()
        constants = f.constants()
        rows = np.clip(raw['values'][:480].reshape(-1, 8), constants['low'], constants['high'])
        self.assertEqual(len(rows), 216_960)
        np.testing.assert_allclose(constants['mean'], rows.mean(0), rtol=0, atol=1e-9)
        np.testing.assert_allclose(constants['std'], rows.std(0), rtol=0, atol=1e-9)
        everything = np.clip(raw['values'].reshape(-1, 8), constants['low'], constants['high'])
        self.assertFalse(np.allclose(constants['mean'], everything.mean(0), rtol=0, atol=1e-12))

    def test_constants_are_pinned_by_digest(self):
        f = self.f
        self.assertEqual(f.constants_digest(f.constants()), f.CONSTANTS_SHA256)
        changed = {key: value.copy() for key, value in f.constants().items()}
        changed['mean'][0] += 1e-9
        self.assertNotEqual(f.constants_digest(changed), f.CONSTANTS_SHA256)

    def test_constants_that_differ_from_the_pin_are_refused(self):
        f = self.f
        f._loaded.cache_clear()
        self.addCleanup(f._loaded.cache_clear)
        with patch.object(f, 'CONSTANTS_SHA256', '0'*64):
            with self.assertRaises(ValueError):
                f.constants()

    def test_the_training_rows_standardise_to_zero_mean_and_unit_spread(self):
        table = self.f.table()
        self.assertEqual(table.shape, (540, 452, 8))
        self.assertEqual(table.dtype, np.float32)
        rows = table[:480].reshape(-1, 8).astype(np.float64)
        np.testing.assert_allclose(rows.mean(0), 0., atol=1e-5)
        np.testing.assert_allclose(rows.std(0), 1., atol=1e-5)

    def test_extra_state_is_the_clipped_standardised_sidecar_row(self):
        f = self.f
        raw = f.load_raw()['values']
        c = f.constants()
        for episode, frame in ((0, 0), (0, 451), (271, 200), (479, 17), (480, 0), (539, 451)):
            expected = ((np.clip(raw[episode, frame], c['low'], c['high'])-c['mean'])/c['std']).astype(np.float32)
            got = f.extra_state(episode, frame)
            self.assertEqual(got.dtype, np.float32)
            self.assertEqual(got.shape, (8,))
            self.assertEqual(got.tobytes(), expected.tobytes(), (episode, frame))

    def test_extra_state_refuses_indices_outside_the_sidecar(self):
        for episode, frame in ((540, 0), (-1, 0), (0, 452), (0, -1)):
            with self.assertRaises(ValueError):
                self.f.extra_state(episode, frame)
        for bad in (1.0, True, '0'):
            with self.assertRaises(ValueError):
                self.f.extra_state(bad, 0)

    def test_expected_extras_maps_dataset_rows_to_episode_and_frame(self):
        f = self.f
        rows = [0, 451, 452, 203_039, 244_079]
        got = f.expected_extras(rows)
        self.assertEqual(got.shape, (5, 8))
        for k, row in enumerate(rows):
            np.testing.assert_array_equal(got[k], f.extra_state(row//452, row % 452))
        with self.assertRaises(ValueError):
            f.expected_extras([244_080])


if __name__ == '__main__':
    unittest.main()
