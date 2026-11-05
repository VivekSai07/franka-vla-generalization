"""The class-conditional two-camera centre-point box detector: shapes, parameter budget, targets, loss, decoding, and that it learns.

Admission or robosuite image, CPU, tiny inputs. (The real inputs are 288 x 480 and 256 x 256; the network is fully convolutional.)
"""
import unittest

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import model as m


class Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.net = m.BoxDetector()

    def test_the_parameter_budget(self):
        count = m.count_parameters(self.net)
        self.assertGreater(count, 10_000_000)
        self.assertLessEqual(count, k.MAX_PARAMETERS)

    def test_the_real_input_sizes_give_stride_four_maps_with_one_channel_per_object(self):
        self.net.eval()
        with torch.no_grad():
            front, gripper = self.net(torch.zeros(2, 3, *k.FRONT_SHAPE), torch.zeros(2, 3, *k.GRIPPER_SHAPE))
        self.assertEqual(tuple(front["heat"].shape), (2, 6, 72, 120))
        self.assertEqual(tuple(front["offset"].shape), (2, 2, 72, 120))
        self.assertEqual(tuple(front["size"].shape), (2, 2, 72, 120))
        self.assertEqual(tuple(gripper["heat"].shape), (2, 6, 64, 64))

    def test_the_two_cameras_have_their_own_weights_and_nothing_is_pretrained(self):
        a, b = self.net.front.state_dict(), self.net.gripper.state_dict()
        self.assertEqual(sorted(a), sorted(b))
        self.assertTrue(any(not torch.equal(a[key], b[key]) for key in a if a[key].dtype.is_floating_point))
        first = self.net.front.stem[0].weight
        self.assertLess(abs(float(first.mean())), 0.05)  # a fresh initialisation, not a downloaded one

    def test_the_heat_map_starts_near_the_prior(self):
        self.net.eval()
        with torch.no_grad():
            front, _ = self.net(torch.zeros(1, 3, *k.FRONT_SHAPE), torch.zeros(1, 3, *k.GRIPPER_SHAPE))
        self.assertLess(float(torch.sigmoid(front["heat"]).mean()), 0.2)


class Normalise(unittest.TestCase):
    def test_uint8_nhwc_becomes_a_float_nchw_with_the_pinned_mean_and_std(self):
        image = np.zeros((2, 4, 6, 3), np.uint8)
        image[0, :, :, 0] = 255
        out = m.normalise(torch.from_numpy(image))
        self.assertEqual((tuple(out.shape), out.dtype), ((2, 3, 4, 6), torch.float32))
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), (1.0 - m.MEAN[0]) / m.STD[0], places=6)
        self.assertAlmostEqual(float(out[1, 2, 0, 0]), (0.0 - m.MEAN[2]) / m.STD[2], places=6)


class Targets(unittest.TestCase):
    def test_a_present_box_by_hand(self):
        t = m.make_target(np.array([40.0, 24.0, 64.0, 56.0]), (288, 480))
        self.assertTrue(t["present"])
        self.assertEqual((t["iy"], t["ix"]), (10, 13))
        np.testing.assert_allclose(t["offset"], [0.0, 0.0])
        np.testing.assert_allclose(t["size"], [24 / 16, 32 / 16])
        self.assertEqual(t["heat"].shape, (72, 120))
        self.assertEqual(float(t["heat"][10, 13]), 1.0)
        self.assertEqual(int((t["heat"] == 1.0).sum()), 1)
        self.assertGreater(float(t["heat"][10, 14]), 0.0)
        self.assertLess(float(t["heat"][10, 14]), 1.0)
        self.assertEqual(float(t["heat"][10, 14]), float(t["heat"][10, 12]))  # symmetric about the centre cell
        self.assertEqual(float(t["heat"][0, 0]), 0.0)

    def test_a_non_integer_centre_splits_into_a_cell_and_an_offset(self):
        t = m.make_target(np.array([41.0, 25.0, 66.0, 59.0]), (288, 480))  # centre (53.5, 42) -> cell (13, 10), offset (0.375, 0.5)
        self.assertEqual((t["ix"], t["iy"]), (13, 10))
        np.testing.assert_allclose(t["offset"], [0.375, 0.5])

    def test_an_absent_box_has_an_empty_heat_map_and_no_box_target(self):
        t = m.make_target(None, (288, 480))
        self.assertFalse(t["present"])
        self.assertEqual(float(t["heat"].max()), 0.0)

    def test_a_box_at_the_image_edge_keeps_its_centre_inside_the_map(self):
        t = m.make_target(np.array([0.0, 0.0, 6.0, 6.0]), (64, 64))
        self.assertEqual((t["ix"], t["iy"]), (0, 0))
        t = m.make_target(np.array([58.0, 58.0, 64.0, 64.0]), (64, 64))
        self.assertEqual((t["ix"], t["iy"]), (15, 15))

    def test_the_gaussian_widens_with_the_box_size(self):
        small = m.make_target(np.array([40.0, 40.0, 56.0, 56.0]), (128, 128))["heat"]
        large = m.make_target(np.array([8.0, 8.0, 120.0, 120.0]), (128, 128))["heat"]
        self.assertGreater(int((large > 0.1).sum()), int((small > 0.1).sum()))


def perfect_outputs(boxes, classes, shape, strength=12.0):
    """Network outputs that reproduce `boxes` (pixel boxes or None) exactly for the target class, for the loss and decode tests."""
    n = len(boxes)
    h, w = shape[0] // k.HEAD_STRIDE, shape[1] // k.HEAD_STRIDE
    heat = torch.full((n, 6, h, w), -strength)
    offset, size = torch.zeros(n, 2, h, w), torch.zeros(n, 2, h, w)
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        if box is None:
            continue
        t = m.make_target(box, shape)
        heat[i, cls, t["iy"], t["ix"]] = strength
        offset[i, :, t["iy"], t["ix"]] = torch.tensor(t["offset"], dtype=torch.float32)
        size[i, :, t["iy"], t["ix"]] = torch.tensor(t["size"], dtype=torch.float32)
    return dict(heat=heat, offset=offset, size=size)


BOXES = [np.array([40.0, 24.0, 64.0, 56.0]), None, np.array([100.0, 80.0, 130.0, 110.0])]
CLASSES = [3, 0, 5]
SHAPE = (288, 480)


class Loss(unittest.TestCase):
    def batch(self, boxes=BOXES, classes=CLASSES):
        return m.collate_targets([m.make_target(b, SHAPE) for b in boxes], classes)

    def test_a_perfect_prediction_has_almost_no_loss_and_a_wrong_one_a_lot(self):
        targets = self.batch()
        perfect = m.loss(perfect_outputs(BOXES, CLASSES, SHAPE), targets)
        self.assertLess(float(perfect["total"]), 1e-3)
        shifted = perfect_outputs([np.array([60.0, 60.0, 84.0, 92.0]), None, BOXES[2]], CLASSES, SHAPE)
        self.assertGreater(float(m.loss(shifted, targets)["total"]), 1.0)

    def test_an_absent_sample_trains_only_presence(self):
        targets = self.batch([None, None], [0, 1])
        outputs = perfect_outputs([None, None], [0, 1], SHAPE)
        for key in ("offset", "size"):
            outputs[key] = torch.randn_like(outputs[key]).requires_grad_(True)
        outputs["heat"] = outputs["heat"].clone().requires_grad_(True)
        parts = m.loss(outputs, targets)
        self.assertEqual(float(parts["box"]), 0.0)
        parts["total"].backward()
        self.assertEqual(float(outputs["offset"].grad.abs().sum()), 0.0)
        self.assertEqual(float(outputs["size"].grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(parts["total"]))

    def test_a_confident_false_alarm_on_an_absent_frame_is_penalised(self):
        targets = self.batch([None], [2])
        quiet = perfect_outputs([None], [2], SHAPE)
        alarm = perfect_outputs([BOXES[0]], [2], SHAPE)
        self.assertGreater(float(m.loss(alarm, targets)["total"]), float(m.loss(quiet, targets)["total"]) + 1.0)

    def test_only_the_target_class_channel_is_supervised(self):
        targets = self.batch()
        outputs = perfect_outputs(BOXES, CLASSES, SHAPE)
        other = {k_: v.clone() for k_, v in outputs.items()}
        for i, cls in enumerate(CLASSES):
            other["heat"][i, (cls + 1) % 6] = 12.0  # shouting in a channel that is not the instructed one changes nothing
        self.assertEqual(float(m.loss(other, targets)["total"]), float(m.loss(outputs, targets)["total"]))

    def test_the_box_loss_sees_a_wrong_offset_or_size_at_the_centre_cell_only(self):
        targets = self.batch([BOXES[0]], [3])
        outputs = perfect_outputs([BOXES[0]], [3], SHAPE)
        wrong = {k_: v.clone() for k_, v in outputs.items()}
        wrong["size"][0, 0, 10, 13] += 1.0
        self.assertAlmostEqual(float(m.loss(wrong, targets)["box"]), 1.0, places=5)
        elsewhere = {k_: v.clone() for k_, v in outputs.items()}
        elsewhere["size"][0, 0, 3, 3] += 5.0
        self.assertEqual(float(m.loss(elsewhere, targets)["box"]), 0.0)


class Decode(unittest.TestCase):
    def test_a_perfect_output_decodes_to_its_box_and_presence(self):
        outputs = perfect_outputs(BOXES, CLASSES, SHAPE)
        boxes, logits = m.decode(outputs, torch.tensor(CLASSES))
        np.testing.assert_allclose(boxes[0].numpy(), BOXES[0], atol=1e-5)
        np.testing.assert_allclose(boxes[2].numpy(), BOXES[2], atol=1e-5)
        self.assertGreaterEqual(float(torch.sigmoid(logits[0])), k.PRESENCE_THRESHOLD)
        self.assertLess(float(torch.sigmoid(logits[1])), k.PRESENCE_THRESHOLD)

    def test_it_reads_the_instructed_channel_not_the_loudest(self):
        outputs = perfect_outputs([BOXES[0]], [3], SHAPE)
        outputs["heat"][0, 1, 30, 60] = 20.0  # a louder peak elsewhere in another object's channel
        boxes, logits = m.decode(outputs, torch.tensor([3]))
        np.testing.assert_allclose(boxes[0].numpy(), BOXES[0], atol=1e-5)
        self.assertAlmostEqual(float(logits[0]), 12.0, places=4)
        other_box, other_logit = m.decode(outputs, torch.tensor([1]))
        self.assertAlmostEqual(float(other_logit[0]), 20.0, places=4)

    def test_the_presence_threshold_is_a_sigmoid_of_one_half(self):
        self.assertEqual(k.PRESENCE_THRESHOLD, 0.5)
        self.assertTrue(bool(m.is_present(torch.tensor([0.0, 0.001, -0.001]))[:2].all()))
        self.assertFalse(bool(m.is_present(torch.tensor([-0.001]))[0]))


class Learns(unittest.TestCase):
    def test_a_few_steps_find_a_bright_rectangle_and_the_empty_frames(self):
        """A synthetic scene (dark frame, one bright rectangle of the instructed object or nothing): the loss must fall and the box be found."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        shape, steps = (64, 96), 90
        net = m.BoxDetector(widths=(16, 32, 64, 128), head_channels=32)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)

        def sample(n):
            images, boxes = np.zeros((n, *shape, 3), np.uint8), []
            for i in range(n):
                if rng.random() < 0.25:
                    boxes.append(None)
                    continue
                w, h = rng.integers(10, 24), rng.integers(10, 24)
                x, y = rng.integers(0, shape[1] - w), rng.integers(0, shape[0] - h)
                images[i, y:y + h, x:x + w] = 255
                boxes.append(np.array([x, y, x + w, y + h], dtype=float))
            return images, boxes

        history = []
        net.train()
        for _ in range(steps):
            images, boxes = sample(16)
            x = m.normalise(torch.from_numpy(images))
            out, _ = net(x, x)
            targets = m.collate_targets([m.make_target(b, shape) for b in boxes], [0] * 16)
            parts = m.loss(out, targets)
            optimizer.zero_grad()
            parts["total"].backward()
            optimizer.step()
            history.append(float(parts["total"]))
        self.assertLess(np.mean(history[-10:]), 0.5 * np.mean(history[:10]))
        net.eval()
        images, boxes = sample(64)
        with torch.no_grad():
            out, _ = net(m.normalise(torch.from_numpy(images)), m.normalise(torch.from_numpy(images)))
        decoded, logits = m.decode(out, torch.zeros(64, dtype=torch.long))
        present = m.is_present(logits).numpy()
        truth = np.array([b is not None for b in boxes])
        self.assertGreater((present == truth).mean(), 0.9)
        ious = [m.pixel_iou(decoded[i].numpy(), boxes[i]) for i in range(64) if truth[i] and present[i]]
        self.assertGreater(np.mean(ious), 0.5)


class PixelIou(unittest.TestCase):
    def test_iou_of_pixel_boxes(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        self.assertEqual(m.pixel_iou(a, a), 1.0)
        self.assertEqual(m.pixel_iou(a, np.array([10.0, 0.0, 20.0, 10.0])), 0.0)
        self.assertAlmostEqual(m.pixel_iou(a, np.array([5.0, 0.0, 15.0, 10.0])), 1 / 3)
        self.assertEqual(m.pixel_iou(a, np.array([20.0, 20.0, 20.0, 30.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
