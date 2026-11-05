"""The detector gate (spec D5): IoU on decoded and on raw frames, presence F1 on synthetic out-of-view frames, and the decision.

Admission image, CPU; stand-in predictors prove the gate passes a perfect detector and fails each kind of bad one.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import data, gate, inputs

import test_data  # noqa: E402  (the synthetic cache of this package's data tests)


class Oracle:
    """A predictor that returns the labels of the rows it is asked about (found by the frames' content): a perfect detector."""

    def __init__(self, cache, perturb=None, always_present=False, never_present=False):
        self.cache, self.perturb, self.always_present, self.never_present = cache, perturb, always_present, never_present
        self.calls = 0

    def predict_arrays(self, fronts, grippers, classes):
        self.calls += 1
        out = np.zeros((len(fronts), 2, 5))
        for i in range(len(fronts)):
            for camera, frames in ((0, fronts), (1, grippers)):
                ys, xs = np.nonzero(frames[i][..., 0])
                if len(xs) == 0 or (xs.max() + 1 - xs.min() < k.MIN_VISIBLE_PX) or (ys.max() + 1 - ys.min() < k.MIN_VISIBLE_PX):
                    present, box = False, np.zeros(5)
                else:
                    pixels = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
                    box = (inputs.input_to_front_box if camera == 0 else inputs.input_to_gripper_box)(pixels)
                    present = True
                if self.perturb is not None and present:
                    box = self.perturb(box, camera)
                if self.always_present and not present:
                    box = np.array([0.4, 0.4, 0.5, 0.5, 1.0])
                if self.never_present:
                    box = np.zeros(5)
                out[i, camera] = box
        return out, np.where(out[:, :, 4] > 0, 5.0, -5.0).astype(np.float32)


def shifted(box, camera, dx=0.0, dy=0.0):
    out = box.copy()
    out[[0, 2]] += dx
    out[[1, 3]] += dy
    return np.clip(out, 0.0, 1.0)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name) / "cache"
        test_data.synthetic(directory, rows=[(480 + e, f) for e in range(2) for f in (0, 8)] + [(e, f) for e in range(2) for f in (0, 8)])
        self.cache = data.open_cache(directory)
        self.rows = np.arange(4)  # the validation episodes
        self.cache_obj = self.cache


class Iou(unittest.TestCase):
    def test_iou_of_boxes_in_the_full_frame_and_an_absent_prediction_scores_zero(self):
        a = np.array([0.1, 0.1, 0.3, 0.3, 1.0])
        self.assertEqual(gate.iou(a, a), 1.0)
        self.assertAlmostEqual(gate.iou(np.array([0.2, 0.1, 0.4, 0.3, 1.0]), a), 1 / 3)
        self.assertEqual(gate.iou(np.zeros(5), a), 0.0)
        self.assertEqual(gate.iou(a, np.array([0.5, 0.5, 0.6, 0.6, 1.0])), 0.0)


class DecodedReport(Fixture):
    def test_a_perfect_detector_scores_one_everywhere(self):
        report = gate.decoded_report(Oracle(self.cache), self.cache, self.rows, batch=3)
        for camera in ("front", "gripper"):
            self.assertGreater(report[camera]["mean_iou"], 0.95)  # the fixture draws integer rectangles for fractional labels
            self.assertEqual(report[camera]["share_below_half"], 0.0)
            self.assertEqual(report[camera]["frames"], 4)

    def test_a_shifted_front_box_costs_iou_and_below_half_frames_are_counted(self):
        report = gate.decoded_report(Oracle(self.cache, perturb=lambda b, c: shifted(b, c, dx=0.5 * (b[2] - b[0])) if c == 0 else b), self.cache, self.rows)
        self.assertLess(report["front"]["mean_iou"], 0.6)
        self.assertGreater(report["front"]["share_below_half"], 0.5)
        self.assertGreater(report["gripper"]["mean_iou"], 0.95)

    def test_a_detector_that_never_fires_scores_zero_and_every_frame_is_below_half(self):
        report = gate.decoded_report(Oracle(self.cache, never_present=True), self.cache, self.rows)
        for camera in ("front", "gripper"):
            self.assertEqual(report[camera]["mean_iou"], 0.0)
            self.assertEqual(report[camera]["share_below_half"], 1.0)
            self.assertEqual(report[camera]["share_predicted_absent"], 1.0)

    def test_only_validation_rows_are_measured(self):
        with self.assertRaises(ValueError):
            gate.decoded_report(Oracle(self.cache), self.cache, np.array([0, 5]))


class SyntheticReport(Fixture):
    def test_the_synthetic_frames_are_deterministic_and_have_both_classes_with_exact_labels(self):
        a = gate.synthetic_set(self.cache, self.rows)
        b = gate.synthetic_set(self.cache, self.rows)
        np.testing.assert_array_equal(a["front"], b["front"])
        np.testing.assert_array_equal(a["present"], b["present"])
        self.assertTrue(a["present"].any() and (~a["present"]).any())
        # exact labels: a frame is absent iff less than MIN_VISIBLE_PX of the white rectangle remains
        for camera, key in ((0, "front"), (1, "gripper")):
            for i in range(len(self.rows)):
                ys, xs = np.nonzero(a[key][i][..., 0])
                visible = len(xs) > 0 and xs.max() + 1 - xs.min() >= k.MIN_VISIBLE_PX and ys.max() + 1 - ys.min() >= k.MIN_VISIBLE_PX
                self.assertEqual(bool(a["present"][i, camera]), bool(visible), (key, i))

    def test_a_perfect_detector_has_f1_one_in_both_classes(self):
        report = gate.synthetic_report(Oracle(self.cache), self.cache, self.rows)
        for camera in ("front", "gripper"):
            self.assertEqual(report[camera]["f1_present"], 1.0)
            self.assertEqual(report[camera]["f1_absent"], 1.0)
            self.assertEqual(report[camera]["min_f1"], 1.0)

    def test_a_detector_that_always_says_present_fails_the_absent_class(self):
        report = gate.synthetic_report(Oracle(self.cache, always_present=True), self.cache, self.rows)
        for camera in ("front", "gripper"):
            self.assertEqual(report[camera]["f1_absent"], 0.0)
            self.assertEqual(report[camera]["min_f1"], 0.0)

    def test_a_detector_that_never_says_present_fails_the_present_class(self):
        report = gate.synthetic_report(Oracle(self.cache, never_present=True), self.cache, self.rows)
        for camera in ("front", "gripper"):
            self.assertEqual(report[camera]["f1_present"], 0.0)

    def test_f1_by_hand(self):
        self.assertAlmostEqual(gate.f1(tp=8, fp=2, fn=2), 0.8)
        self.assertEqual(gate.f1(tp=0, fp=0, fn=0), 0.0)
        self.assertEqual(gate.f1(tp=0, fp=3, fn=3), 0.0)


def decoded(front=(0.9, 0.01), gripper=(0.95, 0.0)):
    return dict(front=dict(mean_iou=front[0], share_below_half=front[1], frames=100), gripper=dict(mean_iou=gripper[0], share_below_half=gripper[1], frames=100))


def synthetic(front=0.95, gripper=0.95):
    return dict(front=dict(min_f1=front), gripper=dict(min_f1=gripper))


class Decide(unittest.TestCase):
    def test_all_thresholds_met_passes(self):
        verdict = gate.decide(decoded(), decoded(), synthetic())
        self.assertTrue(verdict["passed"])
        self.assertEqual(len(verdict["checks"]), 10)
        self.assertTrue(all(c["ok"] for c in verdict["checks"]))

    def test_the_thresholds_are_the_pinned_ones_and_the_boundary_is_inclusive(self):
        self.assertTrue(gate.decide(decoded((0.85, 0.03)), decoded((0.85, 0.03)), synthetic(0.90, 0.90))["passed"])
        self.assertFalse(gate.decide(decoded((0.8499, 0.01)), decoded(), synthetic())["passed"])
        self.assertFalse(gate.decide(decoded(), decoded((0.9, 0.0301)), synthetic())["passed"])
        self.assertFalse(gate.decide(decoded(), decoded(), synthetic(0.8999, 0.95))["passed"])

    def test_every_single_failure_fails_the_gate_and_is_named(self):
        for name, args in (("decoded front iou", (decoded(front=(0.8, 0.0)), decoded(), synthetic())),
                           ("decoded gripper below half", (decoded(gripper=(0.95, 0.05)), decoded(), synthetic())),
                           ("raw front iou", (decoded(), decoded(front=(0.5, 0.0)), synthetic())),
                           ("raw gripper iou", (decoded(), decoded(gripper=(0.5, 0.0)), synthetic())),
                           ("synthetic front presence", (decoded(), decoded(), synthetic(front=0.5))),
                           ("synthetic gripper presence", (decoded(), decoded(), synthetic(gripper=0.5)))):
            with self.subTest(name):
                verdict = gate.decide(*args)
                self.assertFalse(verdict["passed"])
                self.assertTrue(any(not c["ok"] for c in verdict["checks"]))

    def test_a_missing_measurement_fails_the_gate(self):
        for args in ((None, decoded(), synthetic()), (decoded(), None, synthetic()), (decoded(), decoded(), None)):
            self.assertFalse(gate.decide(*args)["passed"])

    def test_the_numbers_are_recorded_next_to_their_thresholds(self):
        verdict = gate.decide(decoded((0.8, 0.0)), decoded(), synthetic())
        failing = [c for c in verdict["checks"] if not c["ok"]]
        self.assertEqual(len(failing), 1)
        self.assertEqual((failing[0]["value"], failing[0]["threshold"]), (0.8, 0.85))


class MergeRaw(unittest.TestCase):
    def test_per_episode_records_merge_into_one_report(self):
        records = [dict(episode=480, front=[1.0, 0.4], gripper=[0.9, 0.9]), dict(episode=481, front=[0.8], gripper=[0.2])]
        report = gate.merge_raw(records)
        self.assertEqual(report["front"]["frames"], 3)
        self.assertAlmostEqual(report["front"]["mean_iou"], (1.0 + 0.4 + 0.8) / 3)
        self.assertAlmostEqual(report["front"]["share_below_half"], 1 / 3)
        self.assertAlmostEqual(report["gripper"]["share_below_half"], 1 / 3)
        self.assertEqual(report["episodes"], [480, 481])

    def test_duplicate_episodes_are_refused(self):
        with self.assertRaises(ValueError):
            gate.merge_raw([dict(episode=480, front=[1.0], gripper=[1.0]), dict(episode=480, front=[1.0], gripper=[1.0])])


if __name__ == "__main__":
    unittest.main()
