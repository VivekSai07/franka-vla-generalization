"""The four corruptions of the oracle box (arms A6-A9) and the constants that define them. Pure numpy and torch: either image, CPU."""
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from experiments.vla_inputs_arm_xvla_force_box_noisy import corruption
from experiments.xvla_p2_force_box import overlay

FRONT = np.array([0.40, 0.42, 0.46, 0.50, 1.0])
GRIPPER = np.array([0.30, 0.35, 0.70, 0.80, 1.0])
BOX = np.stack([FRONT, GRIPPER])
OTHER = np.stack([np.array([0.10, 0.10, 0.15, 0.17, 1.0]), np.array([0.55, 0.20, 0.90, 0.60, 1.0])])
ABSENT = np.zeros(5)
CONDITIONS = ("jitter_small", "jitter_large", "dropout", "swap")


def apply(condition, box=BOX, tick=0, distractor=OTHER):
    return corruption.apply(condition, box, tick, distractor_box=distractor if condition == "swap" else None)


def random_boxes(n, seed=0):
    """Valid in-frame boxes of all sizes, many touching an edge, a few only a pixel wide."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pair = []
        for _camera in range(2):
            lo, hi = np.sort(rng.random((2, 2)), axis=0)
            if rng.random() < 0.3:
                lo[0] = 0.0
            if rng.random() < 0.3:
                hi[1] = 1.0
            if rng.random() < 0.1:
                hi = lo + 1 / 512
            pair.append([lo[0], lo[1], min(hi[0], 1.0), min(hi[1], 1.0), 1.0])
        out.append(np.array(pair))
    return out


class ConstantsAreFixedBeforeAnyRun(unittest.TestCase):
    def test_the_conditions_and_the_values_the_spec_names(self):
        self.assertEqual(set(corruption.CONDITIONS), set(CONDITIONS))
        self.assertEqual(corruption.CONDITIONS["jitter_small"], dict(kind="jitter", sigma_px=6.0))
        self.assertEqual(corruption.CONDITIONS["jitter_large"], dict(kind="jitter", sigma_px=18.0))
        self.assertEqual(corruption.CONDITIONS["dropout"], dict(kind="dropout", probability=0.5))
        self.assertEqual(corruption.CONDITIONS["swap"], dict(kind="swap"))
        self.assertEqual((corruption.IMAGE, corruption.MIN_SIDE_PX), (512, 2.0))

    def test_each_arm_name_maps_to_one_condition(self):
        self.assertEqual(corruption.ARMS, {"A6": "jitter_small", "A7": "jitter_large", "A8": "dropout", "A9": "swap"})


class OutputIsAlwaysADrawableBox(unittest.TestCase):
    def check(self, out, label, min_side=True):
        self.assertEqual((out.shape, out.dtype), ((2, 5), np.float64), label)
        self.assertTrue(np.isfinite(out).all(), label)
        for camera in range(2):
            box = out[camera]
            if box[4] == 0.0:
                self.assertEqual(box.tolist(), [0.0] * 5, label)
                continue
            self.assertEqual(box[4], 1.0, label)
            self.assertTrue((box[:4] >= 0).all() and (box[:4] <= 1).all(), (label, box))
            if min_side:  # only a noised box is normalised; dropout and swap pass boxes through as they come
                self.assertLessEqual(box[0] + corruption.MIN_SIDE_PX / 512 - 1e-12, box[2], (label, box))
                self.assertLessEqual(box[1] + corruption.MIN_SIDE_PX / 512 - 1e-12, box[3], (label, box))
            self.assertIsNotNone(overlay.pixel_box(box), label)  # raises on anything the overlay would refuse

    def test_jitter_on_random_edge_touching_and_one_pixel_boxes(self):
        for condition in ("jitter_small", "jitter_large"):
            for i, box in enumerate(random_boxes(400)):
                self.check(apply(condition, box, tick=i), (condition, i))

    def test_jitter_on_a_box_that_noise_pushes_off_the_image(self):
        corner = np.stack([np.array([0.999, 0.999, 1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.002, 0.002, 1.0])])
        for tick in range(200):
            self.check(apply("jitter_large", corner, tick), tick)

    def test_dropout_and_swap_outputs(self):
        for i, box in enumerate(random_boxes(100)):
            self.check(apply("dropout", box, tick=i), i, min_side=False)
            self.check(apply("swap", box, tick=i), i, min_side=False)

    def test_a_box_that_is_not_in_frame_stays_absent_under_jitter(self):
        box = np.stack([ABSENT, GRIPPER])
        out = apply("jitter_large", box, 3)
        self.assertEqual(out[0].tolist(), [0.0] * 5)
        self.assertEqual(out[1, 4], 1.0)
        both = apply("jitter_large", np.stack([ABSENT, ABSENT]), 3)
        self.assertEqual(both.tolist(), np.zeros((2, 5)).tolist())

    def test_inputs_that_are_not_a_box_pair_are_refused(self):
        for bad in (np.zeros(5), np.zeros((3, 5)), np.full((2, 5), np.nan), np.stack([FRONT, [0.1, 0.1, 0.2, 0.2, 0.5]]),
                    np.stack([FRONT, [0.1, 0.1, 0.2, 1.2, 1.0]])):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    apply("jitter_small", bad)
        with self.assertRaises(ValueError):
            corruption.apply("nonsense", BOX, 0)

    def test_the_input_is_never_modified_and_the_output_never_aliases_it(self):
        for condition in CONDITIONS:
            before = BOX.copy()
            out = apply(condition)
            np.testing.assert_array_equal(BOX, before)
            self.assertFalse(np.shares_memory(out, BOX))
            self.assertFalse(np.shares_memory(out, OTHER))


class DeterminismDoesNotDependOnOrder(unittest.TestCase):
    def test_same_inputs_same_output_whatever_ran_in_between(self):
        for condition in ("jitter_small", "jitter_large", "dropout"):
            first = apply(condition, BOX, 17)
            np.random.seed(123)
            apply(condition, BOX, 3)
            apply(condition, OTHER, 99)
            np.random.rand(10)
            self.assertEqual(apply(condition, BOX, 17).tobytes(), first.tobytes(), condition)

    def test_the_seed_is_the_sha256_of_condition_box_bytes_and_tick(self):
        expected = int.from_bytes(hashlib.sha256(b"jitter_small\0" + BOX.astype(np.float64).tobytes() + (5).to_bytes(8, "little")).digest()[:8], "big")
        rng = np.random.default_rng(expected)
        noise = rng.normal(0.0, 6.0 / 512, size=(2, 4))
        out = apply("jitter_small", BOX, 5)
        want = BOX[:, :4] + noise
        np.testing.assert_allclose(out[:, :4], want, atol=1e-15)

    def test_a_different_tick_box_or_condition_gives_different_noise(self):
        base = apply("jitter_large", BOX, 4)
        self.assertNotEqual(apply("jitter_large", BOX, 5).tobytes(), base.tobytes())
        self.assertNotEqual(apply("jitter_large", OTHER, 4).tobytes(), base.tobytes())
        small, large = apply("jitter_small", BOX, 4), apply("jitter_large", BOX, 4)
        self.assertNotEqual(np.abs(small - BOX).tobytes(), np.abs(large - BOX).tobytes())

    def test_the_noise_does_not_depend_on_the_order_the_ticks_are_asked_in(self):
        forward = [apply("jitter_small", BOX, t).tobytes() for t in range(20)]
        backward = [apply("jitter_small", BOX, t).tobytes() for t in reversed(range(20))][::-1]
        self.assertEqual(forward, backward)


class TheStatisticsAreWhatTheConstantsSay(unittest.TestCase):
    def test_jitter_noise_has_the_stated_standard_deviation_in_pixels_for_every_coordinate(self):
        wide = np.stack([np.array([0.30, 0.30, 0.70, 0.70, 1.0])] * 2)  # far from every edge and from the minimum side
        for condition, sigma in (("jitter_small", 6.0), ("jitter_large", 18.0)):
            noise = np.stack([(apply(condition, wide, t)[:, :4] - wide[:, :4]) * 512 for t in range(10_000)])
            self.assertEqual(noise.shape, (10_000, 2, 4))
            for camera in range(2):
                for coordinate in range(4):
                    sample = noise[:, camera, coordinate]
                    self.assertAlmostEqual(float(sample.std()), sigma, delta=0.04 * sigma, msg=(condition, camera, coordinate))
                    self.assertLess(abs(float(sample.mean())), 0.04 * sigma, (condition, camera, coordinate))

    def test_the_two_cameras_and_four_coordinates_get_independent_draws(self):
        wide = np.stack([np.array([0.30, 0.30, 0.70, 0.70, 1.0])] * 2)
        noise = np.stack([(apply("jitter_small", wide, t)[:, :4] - wide[:, :4]).reshape(8) for t in range(4000)])
        corr = np.corrcoef(noise.T)
        self.assertLess(float(np.abs(corr - np.eye(8)).max()), 0.08)

    def test_dropout_removes_the_gripper_box_about_half_the_time_and_never_touches_the_front_box(self):
        gone = 0
        for t in range(10_000):
            out = apply("dropout", BOX, t)
            self.assertEqual(out[0].tobytes(), FRONT.tobytes())
            if out[1].tolist() == [0.0] * 5:
                gone += 1
            else:
                self.assertEqual(out[1].tobytes(), GRIPPER.tobytes())
        self.assertAlmostEqual(gone / 10_000, 0.5, delta=0.02)

    def test_a_dropout_probability_of_zero_or_one_is_never_or_always(self):
        for probability, expected in ((0.0, 0), (1.0, 500)):
            with patch.dict(corruption.CONDITIONS, {"dropout": dict(kind="dropout", probability=probability)}):
                self.assertEqual(sum(apply("dropout", BOX, t)[1, 4] == 0.0 for t in range(500)), expected)

    def test_swap_draws_the_distractors_box_on_both_cameras_whatever_the_target_box(self):
        for box in (BOX, OTHER, np.zeros((2, 5))):
            np.testing.assert_array_equal(apply("swap", box, 9), OTHER)

    def test_swap_without_a_distractor_box_is_refused(self):
        with self.assertRaises(ValueError):
            corruption.apply("swap", BOX, 0)
        with self.assertRaises(ValueError):
            corruption.apply("swap", BOX, 0, distractor_box=np.zeros(5))

    def test_a_corruption_with_its_constant_at_zero_is_the_identity(self):
        with patch.dict(corruption.CONDITIONS, {"jitter_small": dict(kind="jitter", sigma_px=0.0)}):
            for t in range(50):
                self.assertEqual(apply("jitter_small", BOX, t).tobytes(), BOX.tobytes())
        with patch.dict(corruption.CONDITIONS, {"dropout": dict(kind="dropout", probability=0.0)}):
            for t in range(50):
                self.assertEqual(apply("dropout", BOX, t).tobytes(), BOX.tobytes())
        self.assertEqual(corruption.apply(corruption.IDENTITY, BOX, 0).tobytes(), BOX.tobytes())
        self.assertFalse(np.shares_memory(corruption.apply(corruption.IDENTITY, BOX, 0), BOX))


class NearestDistractor(unittest.TestCase):
    positions = {"milk": (0.0, 0.0, 0.0), "can": (0.3, 0.0, 0.0), "bread": (0.0, 0.12, 0.05), "far": (1.0, 1.0, 1.0)}

    def test_the_nearest_prop_other_than_the_target_in_3d(self):
        self.assertEqual(corruption.nearest_distractor("milk", self.positions), "bread")
        self.assertEqual(corruption.nearest_distractor("can", self.positions), "milk")  # 0.30 m, bread is 0.33 m

    def test_the_height_counts_and_a_tie_goes_to_the_first_name(self):
        self.assertEqual(corruption.nearest_distractor("a", {"a": (0, 0, 0), "c": (0, 0, 0.1), "b": (0.1, 0, 0)}), "b")
        self.assertEqual(corruption.nearest_distractor("a", {"a": (0, 0, 0), "c": (0.1, 0, 0), "b": (0.0, 0, 0.2)}), "c")

    def test_it_refuses_without_a_target_or_without_any_other_prop(self):
        for bad in ({"milk": (0, 0, 0)}, {"can": (0, 0, 0)}):
            with self.assertRaises(ValueError):
                corruption.nearest_distractor("milk", bad)


class MeanIoU(unittest.TestCase):
    def test_iou_of_known_boxes(self):
        a = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
        self.assertEqual(corruption.iou(a, a), 1.0)
        self.assertEqual(corruption.iou(a, np.array([0.5, 0.5, 1.0, 1.0, 1.0])), 0.0)
        self.assertAlmostEqual(corruption.iou(a, np.array([0.25, 0.0, 0.75, 0.5, 1.0])), 1 / 3)
        self.assertEqual(corruption.iou(ABSENT, ABSENT), 1.0)  # both absent: nothing to disagree on
        self.assertEqual(corruption.iou(a, ABSENT), 0.0)

    def test_the_mean_iou_decreases_with_the_noise(self):
        boxes = [BOX] * 300
        values = {c: corruption.mean_iou(c, boxes) for c in ("jitter_small", "jitter_large")}
        self.assertGreater(values["jitter_small"], values["jitter_large"])
        self.assertTrue(0.0 < values["jitter_large"] < values["jitter_small"] < 1.0)
        self.assertAlmostEqual(corruption.mean_iou("dropout", [BOX] * 2000), 0.75, delta=0.02)  # front 1.0, gripper 1 or 0


if __name__ == "__main__":
    unittest.main()
