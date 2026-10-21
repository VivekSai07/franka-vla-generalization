import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.egl_rendering import compare


def stack(n=4, h=3, w=5, value=100):
    return np.full((n, 2, h, w, 3), value, np.uint8)       # (images, cameras, H, W, rgb)


class ImageComparisonTests(unittest.TestCase):
    def test_identical_stacks_differ_nowhere(self):
        out = compare.compare_images(stack(), stack())
        self.assertEqual((out['images_differing'], out['pixels_differing'], out['max_abs'], out['mean_abs']), (0, 0, 0, 0.0))
        self.assertEqual(out['images'], 8, 'an image is one camera at one tick: 4 ticks x 2 cameras')

    def test_one_channel_of_one_pixel_one_level(self):
        a, b = stack(), stack()
        b[2, 1, 0, 0, 1] += 1
        out = compare.compare_images(a, b)
        self.assertEqual((out['images_differing'], out['pixels_differing'], out['max_abs']), (1, 1, 1))
        self.assertAlmostEqual(out['mean_abs'], 1/b.size)
        self.assertAlmostEqual(out['pixel_fraction'], 1/(4*2*3*5))

    def test_a_pixel_counts_once_however_many_channels_differ(self):
        a, b = stack(), stack()
        b[0, 0, 1, 1, :] = [101, 102, 103]
        out = compare.compare_images(a, b)
        self.assertEqual(out['pixels_differing'], 1)
        self.assertEqual(out['max_abs'], 3)

    def test_unsigned_difference_does_not_wrap(self):
        a, b = stack(value=250), stack(value=5)
        self.assertEqual(compare.compare_images(a, b)['max_abs'], 245)
        self.assertEqual(compare.compare_images(b, a)['max_abs'], 245)

    def test_images_differing_counts_images_not_pixels(self):
        a, b = stack(), stack()
        b[1, 0, :, :, 0] += 2              # a whole camera plane of one image
        b[3, 1, 0, 0, 2] += 1
        out = compare.compare_images(a, b)
        self.assertEqual(out['images_differing'], 2)
        self.assertEqual(out['pixels_differing'], 3*5 + 1)

    def test_shape_mismatch_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            compare.compare_images(stack(n=4), stack(n=3))

    def test_shapes_that_would_broadcast_are_still_an_error(self):
        # (1 tick) against (4 ticks) broadcasts in numpy and would compare one frame four times without complaint
        with self.assertRaises(ValueError):
            compare.compare_images(stack(n=1), stack(n=4))

    def test_per_camera_split(self):
        a, b = stack(), stack()
        b[:, 1, 0, 0, 0] += 4              # camera 1 only
        out = compare.compare_images(a, b)
        self.assertEqual([c['pixels_differing'] for c in out['per_camera']], [0, 4])
        self.assertEqual([c['max_abs'] for c in out['per_camera']], [0, 4])


class PhysicsComparisonTests(unittest.TestCase):
    def arrays(self):
        return dict(qpos=np.arange(6, dtype=np.float64).reshape(2, 3), action=np.ones((2, 8), np.float32), reason='success')

    def test_equal_arrays_differ_nowhere(self):
        self.assertEqual(compare.physics_differences(self.arrays(), self.arrays()), [])

    def test_the_last_bit_of_a_float_is_a_difference(self):
        a, b = self.arrays(), self.arrays()
        b['qpos'][1, 2] = np.nextafter(b['qpos'][1, 2], np.inf)
        self.assertEqual(compare.physics_differences(a, b), ['qpos'])

    def test_dtype_and_shape_and_missing_keys_are_differences(self):
        a, b = self.arrays(), self.arrays()
        b['action'] = b['action'].astype(np.float64)
        b['extra'] = np.zeros(1)
        self.assertEqual(compare.physics_differences(a, b), ['action', 'extra'])
        c = self.arrays()
        c['qpos'] = c['qpos'][:1]
        self.assertEqual(compare.physics_differences(a, c), ['qpos'])

    def test_a_changed_outcome_string_is_a_difference(self):
        a, b = self.arrays(), self.arrays()
        b['reason'] = 'timeout'
        self.assertEqual(compare.physics_differences(a, b), ['reason'])

    def test_nan_in_the_same_place_is_not_a_difference_but_a_moved_nan_is(self):
        a, b = self.arrays(), self.arrays()
        a['qpos'][0, 0] = b['qpos'][0, 0] = np.nan
        self.assertEqual(compare.physics_differences(a, b), [])
        b['qpos'][0, 0], b['qpos'][0, 1] = 0.0, np.nan
        self.assertEqual(compare.physics_differences(a, b), ['qpos'])


class RepeatabilityTests(unittest.TestCase):
    def stats(self, max_abs, images_differing, images=904):
        return dict(max_abs=max_abs, images_differing=images_differing, images=images)

    def test_within_both_limits_passes(self):
        self.assertTrue(compare.repeatable([self.stats(1, 6), self.stats(2, 9)]))        # 9/904 = 0.99%

    def test_the_level_limit_is_inclusive_at_two_and_fails_at_three(self):
        self.assertTrue(compare.repeatable([self.stats(2, 0)]))
        self.assertFalse(compare.repeatable([self.stats(3, 0)]))

    def test_the_image_fraction_limit_is_one_percent_and_exclusive_above(self):
        self.assertTrue(compare.repeatable([self.stats(1, 9, images=900)]))             # exactly 1%
        self.assertFalse(compare.repeatable([self.stats(1, 10, images=900)]))

    def test_one_bad_pair_fails_the_set_and_an_empty_set_is_not_a_pass(self):
        self.assertFalse(compare.repeatable([self.stats(1, 0), self.stats(9, 0)]))
        self.assertFalse(compare.repeatable([]))


if __name__ == '__main__':
    unittest.main()
