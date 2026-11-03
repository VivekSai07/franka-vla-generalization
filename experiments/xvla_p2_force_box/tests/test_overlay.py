"""The overlay: a magenta band of 4 px outside the half-open pixel box, drawn on float32 CHW frames in [0, 1].

Hand-computed pixel sets, not a second implementation: the expected counts and corner pixels below are arithmetic on the
spec's rule (D3): band = [x0-4, x1+4) x [y0-4, y1+4) minus [x0, x1) x [y0, y1), clipped to 512, x = u (right), y = v (down).
"""
import importlib
import unittest

import numpy as np
import torch

MAGENTA = (1.0, 0.0, 1.0)


def overlay():
    return importlib.import_module('experiments.xvla_p2_force_box.overlay')


def frame(value=0.5):
    return torch.full((3, 512, 512), value, dtype=torch.float32)


def random_frame(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((3, 512, 512), generator=generator, dtype=torch.float32)


class PixelBoxTests(unittest.TestCase):
    def setUp(self):
        self.o = overlay()

    def test_the_constants_are_the_specs(self):
        self.assertEqual((self.o.THICKNESS, self.o.COLOUR, self.o.SIZE), (4, MAGENTA, 512))

    def test_the_pixel_box_is_floor_of_the_minima_and_ceil_of_the_maxima(self):
        o = self.o
        self.assertEqual(o.pixel_box([0.25, 0.125, 0.5, 0.375, 1.0]), (128, 64, 256, 192))
        # 512*0.2501 = 128.0512 -> floor 128; 512*0.5001 = 256.0512 -> ceil 257; 512*0.1251 = 64.05 -> 64; 512*0.3751 = 192.05 -> 193
        self.assertEqual(o.pixel_box([0.2501, 0.1251, 0.5001, 0.3751, 1.0]), (128, 64, 257, 193))

    def test_a_box_that_is_not_in_frame_has_no_pixel_box(self):
        self.assertIsNone(self.o.pixel_box([0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertIsNone(self.o.pixel_box([0.25, 0.125, 0.5, 0.375, 0.0]))

    def test_invalid_boxes_are_refused(self):
        o = self.o
        bad = ([0.25, 0.125, 0.5, 0.375], [0.25, 0.125, 0.5, 0.375, 1.0, 0.0], [-0.01, 0.1, 0.5, 0.5, 1.0],
               [0.1, 0.1, 1.01, 0.5, 1.0], [float('nan'), 0.1, 0.5, 0.5, 1.0], [0.1, 0.1, float('inf'), 0.5, 1.0],
               [0.1, 0.1, 0.5, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5, 2.0], [0.5, 0.1, 0.5, 0.5, 1.0],
               [0.1, 0.5, 0.5, 0.5, 1.0], [0.6, 0.1, 0.5, 0.5, 1.0])
        for box in bad:
            with self.subTest(box=box), self.assertRaises(ValueError):
                o.pixel_box(box)


class BandTests(unittest.TestCase):
    def setUp(self):
        self.o = overlay()

    def test_an_interior_box_has_a_band_of_exactly_the_hand_computed_pixels(self):
        band = self.o.band([0.25, 0.125, 0.5, 0.375, 1.0])  # pixel box x [128, 256), y [64, 192)
        self.assertEqual((band.shape, band.dtype), ((512, 512), np.bool_))
        self.assertEqual(int(band.sum()), 136*136-128*128)  # 2112
        # indexed [y, x]
        for y, x in ((60, 124), (60, 259), (195, 124), (195, 259), (63, 130), (192, 130), (130, 124), (130, 259), (130, 127)):
            self.assertTrue(band[y, x], (y, x))
        for y, x in ((59, 130), (196, 130), (130, 123), (130, 260), (64, 128), (191, 255), (100, 200), (0, 0), (511, 511)):
            self.assertFalse(band[y, x], (y, x))

    def test_u_is_the_column_and_v_is_the_row(self):
        band = self.o.band([0.25, 0.5, 0.375, 0.75, 1.0])  # x [128, 192), y [256, 384): wide in y, narrow in x
        self.assertEqual(int(band.sum()), (64+8)*(128+8)-64*128)
        self.assertTrue(band[256, 124] and band[383+4, 150] and band[300, 195])
        self.assertFalse(band[300, 150])
        self.assertFalse(band[150, 300])

    def test_a_box_touching_the_top_left_corner_is_clipped_to_the_image(self):
        band = self.o.band([0.0, 0.0, 0.25, 0.25, 1.0])  # pixel box [0, 128)^2
        self.assertEqual(int(band.sum()), 132*132-128*128)  # 1040
        self.assertTrue(band[0, 128] and band[131, 0] and band[131, 131])
        self.assertFalse(band[0, 0] or band[127, 127] or band[132, 0] or band[0, 132])

    def test_a_box_touching_the_bottom_right_corner_is_clipped_to_the_image(self):
        band = self.o.band([0.75, 0.75, 1.0, 1.0, 1.0])  # pixel box [384, 512)^2
        self.assertEqual(int(band.sum()), 132*132-128*128)
        self.assertTrue(band[380, 511] and band[511, 380] and band[380, 380])
        self.assertFalse(band[511, 511] or band[384, 384] or band[379, 400] or band[400, 379])

    def test_a_box_covering_the_whole_image_has_no_band(self):
        self.assertEqual(int(self.o.band([0.0, 0.0, 1.0, 1.0, 1.0]).sum()), 0)

    def test_a_one_pixel_box_has_a_9_by_9_band_minus_the_pixel(self):
        band = self.o.band([0.5, 0.5, 0.5+1/512, 0.5+1/512, 1.0])  # pixel box x [256, 257), y [256, 257)
        self.assertEqual(int(band.sum()), 9*9-1)
        self.assertFalse(band[256, 256])
        self.assertTrue(band[252, 252] and band[260, 260] and band[256, 252] and band[256, 260])
        self.assertFalse(band[251, 256] or band[261, 256])

    def test_fractional_edges_round_outwards(self):
        band = self.o.band([0.2501, 0.1251, 0.5001, 0.3751, 1.0])  # x [128, 257), y [64, 193)
        self.assertEqual(int(band.sum()), (129+8)*(129+8)-129*129)
        self.assertFalse(band[130, 256] or band[192, 130])  # the last pixel column and row of the box are inside it
        self.assertTrue(band[130, 257] and band[130, 260] and band[193, 130] and band[196, 130] and band[130, 124])
        self.assertFalse(band[130, 261] or band[197, 130] or band[130, 123] or band[59, 130])

    def test_a_box_that_is_not_in_frame_has_an_empty_band(self):
        self.assertEqual(int(self.o.band([0.0, 0.0, 0.0, 0.0, 0.0]).sum()), 0)
        self.assertEqual(int(self.o.band([0.25, 0.125, 0.5, 0.375, 0.0]).sum()), 0)


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.o = overlay()
        self.box = [0.25, 0.125, 0.5, 0.375, 1.0]

    def test_band_pixels_are_exactly_magenta_and_every_other_pixel_is_unchanged(self):
        image = random_frame()
        drawn = self.o.draw(image, self.box)
        band = torch.from_numpy(self.o.band(self.box))
        self.assertEqual((drawn.dtype, tuple(drawn.shape)), (torch.float32, (3, 512, 512)))
        for channel, value in enumerate(MAGENTA):
            self.assertTrue(torch.equal(drawn[channel][band], torch.full((int(band.sum()),), value)))
            self.assertTrue(torch.equal(drawn[channel][~band], image[channel][~band]))

    def test_nothing_inside_the_pixel_box_changes(self):
        image = random_frame(1)
        drawn = self.o.draw(image, self.box)
        self.assertTrue(torch.equal(drawn[:, 64:192, 128:256], image[:, 64:192, 128:256]))

    def test_the_input_is_not_mutated_and_the_result_is_a_copy(self):
        image = random_frame(2)
        before = image.clone()
        drawn = self.o.draw(image, self.box)
        self.assertTrue(torch.equal(image, before))
        self.assertNotEqual(drawn.data_ptr(), image.data_ptr())
        self.assertFalse(torch.equal(drawn, image))

    def test_a_box_not_in_frame_returns_an_equal_copy(self):
        image = random_frame(3)
        drawn = self.o.draw(image, [0.0]*5)
        self.assertTrue(torch.equal(drawn, image))
        self.assertNotEqual(drawn.data_ptr(), image.data_ptr())

    def test_frames_that_are_not_float32_chw_512_in_the_unit_range_are_refused(self):
        o = self.o
        bad = (frame().double(), frame().half(), torch.zeros(3, 256, 256), torch.zeros(512, 512, 3), torch.zeros(4, 512, 512),
               torch.zeros(1, 3, 512, 512), frame(1.5), frame(-0.5), torch.full((3, 512, 512), float('nan')),
               (frame()*255).to(torch.uint8))
        for image in bad:
            with self.subTest(dtype=image.dtype, shape=tuple(image.shape)), self.assertRaises(ValueError):
                o.draw(image, self.box)


class NearMagentaTests(unittest.TestCase):
    """The colour check's measure: pixels within an L-infinity distance of magenta, on frames that are NOT drawn."""

    def setUp(self):
        self.o = overlay()

    def test_exact_and_near_magenta_pixels_are_counted_and_others_are_not(self):
        image = frame(0.5)
        image[:, 0, 0] = torch.tensor([1.0, 0.0, 1.0])  # exact
        image[:, 0, 1] = torch.tensor([0.95, 0.05, 0.95])  # near
        image[:, 0, 2] = torch.tensor([0.85, 0.0, 1.0])  # outside in red
        image[:, 0, 3] = torch.tensor([1.0, 0.15, 1.0])  # outside in green
        image[:, 0, 4] = torch.tensor([1.0, 0.0, 0.85])  # outside in blue
        image[:, 0, 5] = torch.tensor([1.0, 1.0, 1.0])  # white is not magenta
        image[:, 0, 6] = torch.tensor([0.0, 0.0, 0.0])
        self.assertEqual(self.o.near_magenta(image, 0.1), 2)
        self.assertEqual(self.o.near_magenta(image, 0.0), 1)

    def test_a_batch_is_counted_over_all_its_frames(self):
        batch = torch.stack([frame(0.5), frame(0.5)])
        batch[0, :, 0, 0] = torch.tensor([1.0, 0.0, 1.0])
        batch[1, :, 5, 5] = torch.tensor([1.0, 0.0, 1.0])
        batch[1, :, 5, 6] = torch.tensor([1.0, 0.0, 1.0])
        self.assertEqual(self.o.near_magenta(batch, 0.1), 3)

    def test_a_drawn_frame_has_exactly_its_band_pixels_near_magenta_on_a_grey_frame(self):
        box = [0.25, 0.125, 0.5, 0.375, 1.0]
        self.assertEqual(self.o.near_magenta(self.o.draw(frame(0.5), box), 0.1), 136*136-128*128)


class DrawFrameTests(unittest.TestCase):
    def setUp(self):
        self.o = overlay()
        self.front, self.gripper = [0.25, 0.125, 0.5, 0.375, 1.0], [0.5, 0.5, 0.75, 0.75, 1.0]
        self.row = {'observation.images.image': random_frame(4), 'observation.images.image2': random_frame(5),
                    'observation.state': torch.zeros(17), 'task': 'x'}

    def test_front_goes_to_image_and_gripper_to_image2(self):
        o = self.o
        out = o.draw_frame(self.row, np.array([self.front, self.gripper]))
        self.assertTrue(torch.equal(out['observation.images.image'], o.draw(self.row['observation.images.image'], self.front)))
        self.assertTrue(torch.equal(out['observation.images.image2'], o.draw(self.row['observation.images.image2'], self.gripper)))
        self.assertFalse(torch.equal(out['observation.images.image'], o.draw(self.row['observation.images.image'], self.gripper)))

    def test_other_fields_pass_through_and_the_input_row_is_untouched(self):
        before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.row.items()}
        out = self.o.draw_frame(self.row, np.array([self.front, self.gripper]))
        self.assertIsNot(out, self.row)
        self.assertIs(out['observation.state'], self.row['observation.state'])
        self.assertEqual(out['task'], 'x')
        for key, value in before.items():
            same = torch.equal(self.row[key], value) if torch.is_tensor(value) else self.row[key] == value
            self.assertTrue(same, key)

    def test_the_target_box_must_be_two_cameras_of_five_values(self):
        for bad in (np.zeros((5,)), np.zeros((1, 5)), np.zeros((3, 5)), np.zeros((2, 4)), np.zeros((2, 5, 1))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.o.draw_frame(self.row, bad)

    def test_a_row_without_both_camera_frames_is_refused(self):
        for key in ('observation.images.image', 'observation.images.image2'):
            row = {k: v for k, v in self.row.items() if k != key}
            with self.subTest(missing=key), self.assertRaises((ValueError, KeyError)):
                self.o.draw_frame(row, np.array([self.front, self.gripper]))


if __name__ == '__main__':
    unittest.main()
