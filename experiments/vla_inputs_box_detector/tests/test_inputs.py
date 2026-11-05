"""Frame preprocessing, box geometry, exact translation augmentation and the instruction parser. Pure numpy: either image, CPU."""
import unittest

import numpy as np

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import inputs


def render(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(2)]


class Pinned(unittest.TestCase):
    def test_the_constants_are_fixed_before_any_run(self):
        self.assertEqual((k.STRIDE_K, k.FRONT_CROP, k.FRONT_SHAPE, k.GRIPPER_SHAPE, k.GRIPPER_FACTOR, k.HEAD_STRIDE),
                         (8, (128, 416, 16, 496), (288, 480), (256, 256), 2, 4))
        self.assertEqual(len(k.CACHE_FRAMES), 57)
        self.assertEqual((k.CACHE_FRAMES[0], k.CACHE_FRAMES[-1]), (0, 448))
        self.assertEqual((k.PRESENCE_THRESHOLD, k.MIN_VISIBLE_PX), (0.5, 2.0))
        self.assertEqual((k.GATE_MEAN_IOU, k.GATE_SHARE_BELOW_HALF, k.GATE_IOU_FLOOR, k.GATE_PRESENCE_F1), (0.85, 0.03, 0.5, 0.90))
        self.assertEqual((k.TRAIN_EPISODES[0], k.TRAIN_EPISODES[-1], k.VALIDATION_EPISODES[0], k.VALIDATION_EPISODES[-1]), (0, 479, 480, 539))
        self.assertEqual((k.SHIFT_PROBABILITY, k.SHIFT_MAX_FRACTION, k.EPOCHS, k.BATCH, k.SEED, k.MAX_PARAMETERS), (0.5, 0.6, 60, 32, 20261105, 50_000_000))
        self.assertEqual(k.CACHE_DISK_GUARD_GIB, 65)

    def test_the_object_order_is_the_scenes(self):
        from experiments.props_scene import constants as scene
        self.assertEqual(k.OBJECTS, scene.OBJECTS)
        self.assertEqual(k.BINS, scene.BINS)

    def test_the_crop_geometry_is_consistent(self):
        y0, y1, x0, x1 = k.FRONT_CROP
        self.assertEqual((y1 - y0, x1 - x0), k.FRONT_SHAPE)
        self.assertEqual(tuple(s * k.GRIPPER_FACTOR for s in k.GRIPPER_SHAPE), (512, 512))
        for shape in (k.FRONT_SHAPE, k.GRIPPER_SHAPE):
            self.assertTrue(all(s % 32 == 0 for s in shape), shape)  # the backbone's total stride

    def test_the_crop_contains_the_extents_of_all_sealed_target_boxes(self):
        """S0's front boxes (measured over all 540 episodes): u 0.100..0.928, v 0.341..0.711."""
        y0, y1, x0, x1 = k.FRONT_CROP
        self.assertLessEqual(x0, 0.100 * 512)
        self.assertGreaterEqual(x1, 0.928 * 512)
        self.assertLessEqual(y0, 0.341 * 512)
        self.assertGreaterEqual(y1, 0.711 * 512)


class Preprocess(unittest.TestCase):
    def test_front_is_the_fixed_native_resolution_crop(self):
        front_image, _ = render()
        front, _gripper = inputs.preprocess(render())
        self.assertEqual((front.shape, front.dtype), ((288, 480, 3), np.uint8))
        np.testing.assert_array_equal(front, front_image[128:416, 16:496])

    def test_gripper_is_the_two_by_two_mean_rounded_half_up(self):
        _, image = render(1)
        _front, gripper = inputs.preprocess(render(1))
        self.assertEqual((gripper.shape, gripper.dtype), ((256, 256, 3), np.uint8))
        block = image[:2, :2].astype(np.float64).reshape(4, 3)
        np.testing.assert_array_equal(gripper[0, 0], np.floor(block.mean(axis=0) + 0.5).astype(np.uint8))
        manual = np.stack([[np.floor(image[2*i:2*i+2, 2*j:2*j+2].reshape(4, 3).astype(np.float64).mean(axis=0) + 0.5) for j in (0, 97, 255)]
                           for i in (0, 130, 255)]).astype(np.uint8)
        np.testing.assert_array_equal(gripper[[0, 130, 255]][:, [0, 97, 255]], manual)

    def test_a_flat_image_is_unchanged_and_the_input_is_not_modified(self):
        flat = [np.full((512, 512, 3), 77, np.uint8)] * 2
        before = [f.copy() for f in flat]
        front, gripper = inputs.preprocess(flat)
        self.assertTrue((front == 77).all() and (gripper == 77).all())
        for a, b in zip(flat, before):
            np.testing.assert_array_equal(a, b)

    def test_anything_but_two_uint8_512_images_is_refused(self):
        good = render()
        for bad in ([good[0]], [good[0], good[1], good[1]], [good[0], good[1].astype(np.float32)], [good[0], good[1][:256]],
                    [good[0], good[1][:, :, :1]]):
            with self.assertRaises(ValueError):
                inputs.preprocess(bad)


class Boxes(unittest.TestCase):
    def test_boxes_map_to_input_pixels_by_hand(self):
        box = np.array([0.25, 0.5, 0.5, 0.75, 1.0])
        np.testing.assert_array_equal(inputs.front_box_to_input(box), [112, 128, 240, 256])
        np.testing.assert_array_equal(inputs.gripper_box_to_input(box), [64, 128, 128, 192])

    def test_the_round_trip_back_to_the_full_frame(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            lo = 0.2 + 0.3 * rng.random(2)
            box = np.array([lo[0], 0.35 + 0.2 * rng.random(), lo[0] + 0.05 + 0.1 * rng.random(), 0.0, 1.0])
            box[3] = box[1] + 0.05 + 0.1 * rng.random()
            front = inputs.front_box_to_input(box)
            back = inputs.input_to_front_box(front)
            np.testing.assert_allclose(back[:4], box[:4], atol=1e-12)
            self.assertEqual(back[4], 1.0)
            np.testing.assert_allclose(inputs.input_to_gripper_box(inputs.gripper_box_to_input(box))[:4], box[:4], atol=1e-12)

    def test_converting_back_clips_to_the_frame_and_an_empty_box_is_absent(self):
        out = inputs.input_to_front_box(np.array([-20.0, 100.0, 40.0, 300.0]))
        np.testing.assert_allclose(out, [0.0, 228 / 512, 56 / 512, 1.0 * 428 / 512, 1.0])
        out = inputs.input_to_gripper_box(np.array([10.0, 10.0, 600.0, 700.0]))
        np.testing.assert_allclose(out, [20 / 512, 20 / 512, 1.0, 1.0, 1.0])
        self.assertEqual(inputs.input_to_gripper_box(np.array([5.0, 5.0, 5.0, 9.0])).tolist(), [0.0] * 5)
        self.assertEqual(inputs.input_to_gripper_box(np.array([9.0, 5.0, 5.0, 9.0])).tolist(), [0.0] * 5)

    def test_an_absent_box_converts_to_no_pixels(self):
        self.assertIsNone(inputs.front_box_to_input(np.zeros(5)))
        self.assertIsNone(inputs.gripper_box_to_input(np.zeros(5)))


class Shift(unittest.TestCase):
    def test_the_image_content_moves_by_the_shift_and_the_rest_is_zero(self):
        image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3) + 1
        out = inputs.shift_image(image, 2, 1)
        self.assertTrue((out[:1] == 0).all() and (out[:, :2] == 0).all())
        np.testing.assert_array_equal(out[1:, 2:], image[:4, :4])
        out = inputs.shift_image(image, -3, -2)
        np.testing.assert_array_equal(out[:3, :3], image[2:, 3:])
        self.assertTrue((out[3:] == 0).all() and (out[:, 3:] == 0).all())
        np.testing.assert_array_equal(inputs.shift_image(image, 0, 0), image)
        self.assertTrue((inputs.shift_image(image, 6, 0) == 0).all() and (inputs.shift_image(image, 0, -5) == 0).all())
        self.assertFalse(np.shares_memory(inputs.shift_image(image, 0, 0), image))

    def test_the_box_moves_with_the_image_and_is_clipped(self):
        box, present = inputs.shift_box(np.array([10.0, 10.0, 20.0, 20.0]), 5, -5, (100, 100))
        self.assertTrue(present)
        np.testing.assert_array_equal(box, [15, 5, 25, 15])
        box, present = inputs.shift_box(np.array([10.0, 10.0, 20.0, 20.0]), -4, 0, (100, 100))
        self.assertTrue(present)
        np.testing.assert_array_equal(box, [6, 10, 16, 20])
        box, present = inputs.shift_box(np.array([10.0, 10.0, 20.0, 20.0]), -15, 0, (100, 100))
        self.assertTrue(present)
        np.testing.assert_array_equal(box, [0, 10, 5, 20])

    def test_a_box_with_no_area_or_less_than_two_pixels_left_is_absent(self):
        shape = (100, 100)
        for dx, dy, expected in ((-20, 0, False), (-19, 0, False), (-18, 0, True), (0, 95, False), (0, 89, False), (0, 88, True), (88, 0, True),
                                 (89, 0, False), (91, 0, False), (200, 200, False), (-200, 0, False)):
            box, present = inputs.shift_box(np.array([10.0, 10.0, 20.0, 20.0]), dx, dy, shape)
            self.assertEqual(present, expected, (dx, dy))
            if not present:
                self.assertEqual(box.tolist(), [0.0] * 4)

    def test_image_and_box_agree_exactly_on_random_shifts(self):
        """A bright rectangle on a dark image: after shifting both, the bright pixels are exactly the shifted, clipped box."""
        rng = np.random.default_rng(3)
        shape = (60, 80)
        for _ in range(400):
            x0, y0 = rng.integers(0, 50), rng.integers(0, 30)
            x1, y1 = x0 + rng.integers(3, 25), y0 + rng.integers(3, 25)
            x1, y1 = min(x1, 80), min(y1, 60)
            image = np.zeros((*shape, 3), np.uint8)
            image[y0:y1, x0:x1] = 255
            dx, dy = int(rng.integers(-70, 70)), int(rng.integers(-50, 50))
            moved = inputs.shift_image(image, dx, dy)
            box, present = inputs.shift_box(np.array([x0, y0, x1, y1], dtype=float), dx, dy, shape)
            ys, xs = np.nonzero(moved[..., 0])
            if len(xs) == 0:
                self.assertFalse(present)
                continue
            seen = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            if present:
                np.testing.assert_array_equal(seen, box)
            else:
                self.assertTrue(seen[2] - seen[0] < k.MIN_VISIBLE_PX or seen[3] - seen[1] < k.MIN_VISIBLE_PX)

    def test_a_non_integer_shift_is_refused(self):
        with self.assertRaises(ValueError):
            inputs.shift_image(np.zeros((4, 4, 3), np.uint8), 1.5, 0)
        with self.assertRaises(ValueError):
            inputs.shift_box(np.array([0.0, 0.0, 2.0, 2.0]), 0, 0.5, (4, 4))


class Instruction(unittest.TestCase):
    def test_the_ten_trained_and_the_two_held_out_instructions_parse_to_their_object(self):
        from experiments.props_p2_data.plan import HELD_OUT_PAIRS, TRAINED_PAIRS, instruction
        for obj, colour in (*TRAINED_PAIRS, *HELD_OUT_PAIRS):
            self.assertEqual(inputs.parse_instruction(instruction(obj, colour)), k.OBJECTS.index(obj))
        self.assertEqual(len(TRAINED_PAIRS), 10)

    def test_everything_else_is_refused(self):
        for bad in ("put the cube in the blue bin", "put the can in the red bin", "Put the can in the blue bin", "put the can in the blue bin ",
                    "put  the can in the blue bin", "put the can on the blue bin", "put the can in the blue", "", "can", "put the can in the blue bin.",
                    None, 3, b"put the can in the blue bin"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    inputs.parse_instruction(bad)


if __name__ == "__main__":
    unittest.main()
