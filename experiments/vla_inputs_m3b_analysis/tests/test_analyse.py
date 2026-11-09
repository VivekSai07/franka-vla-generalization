"""M3b analysis: recovery R and its interval, the paired tests, Holm, failure classes, the detector's per-tick error, the refusals.

Pure numpy and json on synthetic campaign roots; the admission image, no simulator. Hand-computed answers throughout.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.vla_inputs_m3b_analysis import analyse

OBJECTS = ("bottle", "bread", "can", "cereal", "lemon", "milk")
CORE = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"


def layouts(n=573):
    return [dict(layout_id=f"L{i:03d}", object=OBJECTS[i % 6]) for i in range(n)]


def rows(successes, n=573, causes=None):
    """One row per layout; layouts in `successes` (a set of positions) succeed."""
    out = []
    for i, layout in enumerate(layouts(n)):
        ok = i in successes
        cause = "success" if ok else (causes or {}).get(i, "never_lifted")
        out.append(dict(layout, task_success=ok, cause=cause, wrong_object=cause == "wrong_object", kind="trained",
                        distractor_count=1, blue_slot="left", ordinal=i + 1, bin="blue"))
    return out


def with_counts(floor, ceiling, arm, n=573):
    """(a2, a3, a5) position sets: A2 passes the first `floor`, A3 the first `ceiling`, A5 the first `ceiling - missed...` per `arm`."""
    return set(range(floor)), set(range(ceiling)), set(arm)


class Recovery(unittest.TestCase):
    def test_s_559_against_523_and_571_is_exactly_three_quarters(self):
        a2, a3 = set(range(523)), set(range(571))
        a5 = set(range(523)) | set(range(523, 559))
        got = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=200)
        self.assertEqual((got["successes"], got["floor"], got["ceiling"], got["gap"]), (559, 523, 571, 48))
        self.assertEqual(got["R"], 0.75)

    def test_r_for_other_counts(self):
        a2, a3 = set(range(523)), set(range(571))
        for s, expected in ((523, 0.0), (571, 1.0), (547, 0.5), (500, -23 / 48), (575 - 4, 1.0)):
            a5 = set(range(min(s, 573)))
            got = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=100)
            self.assertAlmostEqual(got["R"], (s - 523) / 48, places=12, msg=s)

    def test_an_arm_equal_to_a3_gives_one_and_equal_to_a2_gives_zero_with_degenerate_intervals(self):
        a2, a3 = set(range(0, 40, 2)), set(range(0, 40, 2)) | set(range(1, 40, 4))
        top = analyse.recovery(rows(a3, n=60), rows(a2, n=60), rows(a3, n=60), resamples=500)
        bottom = analyse.recovery(rows(a2, n=60), rows(a2, n=60), rows(a3, n=60), resamples=500)
        self.assertEqual((top["R"], top["lo"], top["hi"]), (1.0, 1.0, 1.0))
        self.assertEqual((bottom["R"], bottom["lo"], bottom["hi"]), (0.0, 0.0, 0.0))

    def test_the_bootstrap_is_stratified_by_object_so_the_denominator_stays_fixed_within_a_stratum(self):
        """Ten bottle layouts that A3 passes and A2 fails (A5 passes five of them); ninety lemon layouts everyone passes.
        Stratified, every resample holds exactly ten bottle layouts, so R* = k / 10 with k ~ Binomial(10, 1/2): the 2.5 and 97.5
        percentiles are 0.2 and 0.8. Unstratified the denominator itself varies and the interval is wider."""
        def build(passing_bottles, n_bottle=10, n_lemon=90):
            out = []
            for i in range(n_bottle):
                out.append(dict(layout_id=f"b{i}", object="bottle", task_success=i in passing_bottles, cause="success" if i in passing_bottles else "never_lifted",
                                wrong_object=False, kind="trained", distractor_count=1, blue_slot="left"))
            for i in range(n_lemon):
                out.append(dict(layout_id=f"l{i}", object="lemon", task_success=True, cause="success", wrong_object=False, kind="trained",
                                distractor_count=1, blue_slot="left"))
            return out
        got = analyse.recovery(build(set(range(5))), build(set()), build(set(range(10))), resamples=10_000)
        self.assertEqual(got["R"], 0.5)
        self.assertAlmostEqual(got["lo"], 0.2, places=9)
        self.assertAlmostEqual(got["hi"], 0.8, places=9)
        self.assertEqual(got["degenerate_resamples"], 0)

    def test_numerator_and_denominator_are_resampled_together_over_the_same_layouts(self):
        """If A5 passes exactly the layouts A3 passes and A2 fails, R* is 1 on every resample in which A3 beats A2 at all."""
        n = 40
        a2 = set(range(10))
        a3 = a2 | set(range(10, 30))
        a5 = a3
        got = analyse.recovery(rows(a5, n=n), rows(a2, n=n), rows(a3, n=n), resamples=2000)
        self.assertEqual((got["lo"], got["hi"]), (1.0, 1.0))

    def test_the_same_seed_gives_the_same_interval_and_another_seed_a_different_one(self):
        a2, a3, a5 = set(range(300)), set(range(500)), set(range(420))
        first = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=1000, seed=0)
        again = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=1000, seed=0)
        other = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=1000, seed=1)
        self.assertEqual((first["lo"], first["hi"]), (again["lo"], again["hi"]))
        self.assertNotEqual((first["lo"], first["hi"]), (other["lo"], other["hi"]))

    def test_a_ceiling_that_does_not_beat_the_floor_has_no_recovery(self):
        same = set(range(100))
        with self.assertRaisesRegex(ValueError, "undefined"):
            analyse.recovery(rows(same, n=200), rows(same, n=200), rows(same, n=200))
        with self.assertRaisesRegex(ValueError, "undefined"):
            analyse.recovery(rows(set(range(50)), n=200), rows(set(range(100)), n=200), rows(set(range(90)), n=200))

    def test_arms_over_different_layouts_are_refused(self):
        good = rows(set(range(10)), n=50)
        with self.assertRaisesRegex(ValueError, "layouts"):
            analyse.recovery(rows(set(), n=49), good, good)
        shifted = [dict(r, layout_id=r["layout_id"] + "x") for r in good]
        with self.assertRaisesRegex(ValueError, "layouts"):
            analyse.recovery(shifted, good, good)

    def test_the_floor_and_the_ceiling_are_not_interchangeable(self):
        a2, a3, a5 = set(range(523)), set(range(571)), set(range(559))
        straight = analyse.recovery(rows(a5), rows(a2), rows(a3), resamples=50)
        self.assertEqual((straight["floor"], straight["ceiling"]), (523, 571))
        with self.assertRaisesRegex(ValueError, "undefined"):  # a ceiling below the floor has no recovery
            analyse.recovery(rows(a5), rows(a3), rows(a2), resamples=50)


class PairedTests(unittest.TestCase):
    def test_exact_mcnemar_counts_and_p(self):
        new = rows({0, 1, 2, 5, 6}, n=10)
        control = rows({2, 3, 5, 6, 9}, n=10)
        got = analyse.compare_to(new, control)
        self.assertEqual((got["new_only"], got["control_only"], got["both"], got["neither"]), (2, 2, 3, 3))
        self.assertAlmostEqual(got["mcnemar_exact_p"], 1.0)  # n = 4, b = c = 2: 2 * 11 / 16 capped at 1
        got = analyse.compare_to(rows({0, 1, 2, 3, 4}, n=10), rows({3, 4}, n=10))
        self.assertEqual((got["new_only"], got["control_only"]), (3, 0))
        self.assertAlmostEqual(got["mcnemar_exact_p"], 0.25)  # n = 3: 2 * 1/8
        got = analyse.compare_to(rows({0, 1, 2, 3}, n=10), rows({3, 9}, n=10))
        self.assertEqual((got["new_only"], got["control_only"]), (3, 1))
        self.assertAlmostEqual(got["mcnemar_exact_p"], 0.625)  # n = 4: 2 * (1 + 4) / 16

    def test_the_effect_is_the_paired_difference_with_an_interval(self):
        got = analyse.compare_to(rows(set(range(60)), n=100), rows(set(range(40)), n=100), resamples=500)
        self.assertAlmostEqual(got["effect"]["estimate"], 0.2)
        self.assertLess(got["effect"]["lo"], 0.2)
        self.assertGreater(got["effect"]["hi"], 0.2)
        self.assertEqual((got["new_successes"], got["control_successes"], got["layouts"]), (60, 40, 100))

    def test_different_layouts_are_refused(self):
        with self.assertRaisesRegex(ValueError, "layouts"):
            analyse.compare_to(rows(set(), n=9), rows(set(), n=10))

    def test_holm_over_the_four_corruptions_only(self):
        got = analyse.holm_over_corruptions({"A6": 0.01, "A7": 0.02, "A8": 0.03, "A9": 0.04})
        self.assertEqual(got["adjusted"], {"A6": 0.04, "A7": 0.06, "A8": 0.06, "A9": 0.06})
        self.assertEqual(sorted(got["family"]), ["A6", "A7", "A8", "A9"])

    def test_an_arm_that_was_not_run_enters_as_one_and_an_outsider_is_refused(self):
        got = analyse.holm_over_corruptions({"A6": 0.01, "A9": 0.04})
        self.assertEqual(got["raw"]["A7"], 1.0)
        self.assertEqual(got["missing"], ["A7", "A8"])
        with self.assertRaises(ValueError):
            analyse.holm_over_corruptions({"A5": 0.001, "A6": 0.01})


class DecisionRule(unittest.TestCase):
    def test_the_thresholds_are_pinned_before_any_result(self):
        self.assertEqual((analyse.CLOSED_AT, analyse.PARTIAL_AT, analyse.ALPHA), (0.75, 0.25, 0.05))
        self.assertEqual((analyse.FLOOR_CHECKPOINT, analyse.CEILING_CHECKPOINT, analyse.CHECKPOINT), (88830, 101520, 101520))
        self.assertEqual((analyse.REPLAN_STRIDE, analyse.IOU_BINS), (10, 10))
        self.assertEqual(analyse.CONTROL, "A2")
        self.assertEqual(analyse.CEILING, "A3")
        self.assertEqual(analyse.CORE_SHA256, CORE)

    def test_the_branch_follows_the_point_estimate_and_the_pre_registered_test_against_a2(self):
        significant = dict(effect=dict(estimate=0.1), mcnemar_exact_p=0.001)
        self.assertEqual(analyse.branch(0.75, significant)["branch"], "M3b closes; M4a reconsidered at lower priority")
        self.assertEqual(analyse.branch(0.9, significant)["branch"], "M3b closes; M4a reconsidered at lower priority")
        self.assertEqual(analyse.branch(0.7499, significant)["branch"], "partial: M3b-2 considered (retrain on noisy or predicted boxes)")
        self.assertEqual(analyse.branch(0.25, significant)["branch"], "partial: M3b-2 considered (retrain on noisy or predicted boxes)")
        self.assertEqual(analyse.branch(0.2499, significant)["branch"], "M4a next (the gain needs near-perfect information)")
        not_above = dict(effect=dict(estimate=0.01), mcnemar_exact_p=0.2)
        self.assertEqual(analyse.branch(0.9, not_above)["branch"], "M4a next (the gain needs near-perfect information)")
        negative = dict(effect=dict(estimate=-0.1), mcnemar_exact_p=0.001)
        self.assertEqual(analyse.branch(0.5, negative)["branch"], "M4a next (the gain needs near-perfect information)")

    def test_a_threshold_straddled_by_the_interval_is_reported_as_such(self):
        significant = dict(effect=dict(estimate=0.1), mcnemar_exact_p=0.001)
        got = analyse.branch(0.8, significant, interval=(0.6, 0.95))
        self.assertTrue(got["straddles"])
        got = analyse.branch(0.8, significant, interval=(0.76, 0.95))
        self.assertFalse(got["straddles"])

    def test_the_a9_reading_thresholds(self):
        self.assertEqual(analyse.a9_reading(0.90), "language dominates: the box is supplementary localisation")
        self.assertEqual(analyse.a9_reading(0.8999), "mixed")
        self.assertEqual(analyse.a9_reading(0.40), "mixed")
        self.assertEqual(analyse.a9_reading(0.3999), "the box is followed")

    def test_the_recorded_prediction_is_pinned(self):
        p = analyse.PREDICTION
        self.assertEqual(p["A5"]["R"], (0.6, 0.95))
        self.assertEqual(p["A5"]["classes"], dict(never_lifted=(3, 20), wrong_object=(0, 12), timeout_carrying_or_unstable=(0, 8), dropped_outside=(0, 3)))
        self.assertEqual((p["A6"]["successes"], p["A7"]["successes"], p["A8"]["successes"]), ((560, 571), (500, 565), (480, 571)))
        self.assertEqual(p["A9"]["successes_below"], 458)


class FailureClasses(unittest.TestCase):
    def test_counts_per_class_with_the_pooled_timeout_and_unstable(self):
        causes = {0: "never_lifted", 1: "never_lifted", 2: "wrong_object", 3: "timeout_carrying", 4: "unstable_in_target", 5: "dropped_outside",
                  6: "wrong_bin"}
        got = analyse.failure_classes(rows({7, 8}, n=9, causes=causes))
        self.assertEqual(got["never_lifted"], 2)
        self.assertEqual(got["wrong_object"], 1)
        self.assertEqual(got["timeout_carrying_or_unstable"], 2)
        self.assertEqual(got["dropped_outside"], 1)
        self.assertEqual(got["wrong_bin"], 1)
        self.assertEqual(got["success"], 2)
        self.assertEqual(got["failures"], 7)

    def test_the_counts_are_compared_with_the_predicted_ranges(self):
        counts = dict(never_lifted=3, wrong_object=13, timeout_carrying_or_unstable=0, dropped_outside=4)
        got = analyse.against_prediction("A5", counts)
        self.assertEqual(got["never_lifted"], dict(count=3, range=(3, 20), within=True))
        self.assertFalse(got["wrong_object"]["within"])
        self.assertFalse(got["dropped_outside"]["within"])
        self.assertTrue(got["timeout_carrying_or_unstable"]["within"])


class Iou(unittest.TestCase):
    def test_iou_of_known_boxes(self):
        a = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
        self.assertEqual(analyse.box_iou(a, a), 1.0)
        self.assertEqual(analyse.box_iou(a, np.array([0.5, 0.5, 1.0, 1.0, 1.0])), 0.0)
        self.assertAlmostEqual(analyse.box_iou(a, np.array([0.25, 0.0, 0.75, 0.5, 1.0])), 1 / 3)
        self.assertEqual(analyse.box_iou(np.zeros(5), np.zeros(5)), 1.0)
        self.assertEqual(analyse.box_iou(a, np.zeros(5)), 0.0)
        self.assertEqual(analyse.box_iou(np.zeros(5), a), 0.0)


def oracle_sequence(seed, ticks=25):
    rng = np.random.default_rng(seed)
    start = rng.random((2, 2)) * 0.5
    size = 0.05 + rng.random((2, 2)) * 0.2
    seq = []
    for t in range(ticks):
        lo = start + 0.002 * t
        seq.append(np.concatenate([lo, lo + size, np.ones((2, 1))], axis=1))
    return np.stack(seq)


class DetectorError(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = self.root / "stage"
        self.logs = self.root / "logs"
        self.stage.mkdir()
        self.logs.mkdir()

    def trial(self, ordinal, layout, oracle, predicted, success=True, logfile="shard0.jsonl", name=None):
        folder = self.stage / f"trial_{ordinal:03d}_{layout}"
        folder.mkdir()
        np.savez(folder / "trace.npz", target_box=oracle)
        (folder / "episode.json").write_text(json.dumps(dict(ordinal=ordinal, layout_id=layout, task_success=success)))
        with (self.logs / logfile).open("a") as stream:
            for tick in range(len(oracle)):
                stream.write(json.dumps(dict(trial=name or f"p1-r{ordinal}", tick=tick, predicted=predicted[tick].tolist(),
                                             oracle=oracle[tick].tolist())) + "\n")

    def test_a_perfect_detector_has_iou_one_no_misses_and_the_oracle_s_own_jitter(self):
        oracle = oracle_sequence(1)
        self.trial(1, "A", oracle, oracle.copy())
        got = analyse.detector_error(self.logs, self.stage)
        for camera in ("front", "gripper"):
            self.assertEqual(got["overall"][camera]["mean_iou"], 1.0)
            self.assertEqual(got["overall"][camera]["share_below_half"], 0.0)
            self.assertEqual(got["overall"][camera]["share_predicted_absent"], 0.0)
        self.assertEqual(got["trials_matched"], 1)
        self.assertEqual(got["overall"]["front"]["replan_jitter_px"], got["overall"]["front"]["oracle_replan_jitter_px"])

    def test_a_missing_box_a_shifted_box_and_a_false_alarm_are_counted_per_camera(self):
        oracle = oracle_sequence(2, ticks=20)
        predicted = oracle.copy()
        predicted[:5, 1] = 0.0  # gripper camera: absent for 5 of 20 ticks, oracle present
        width = predicted[5:10, 0, 2] - predicted[5:10, 0, 0]
        predicted[5:10, 0, 0] += width  # front box moved right by its own width: no overlap, IoU 0, for ticks 5..9
        predicted[5:10, 0, 2] += width
        oracle2 = oracle.copy()
        oracle2[15:18, 1] = 0.0  # the oracle loses the gripper box for ticks 15..17 while the prediction still has a box there
        self.trial(1, "A", oracle2, predicted)
        got = analyse.detector_error(self.logs, self.stage)
        gripper, front = got["overall"]["gripper"], got["overall"]["front"]
        self.assertAlmostEqual(gripper["share_predicted_absent"], 5 / 20)
        self.assertAlmostEqual(gripper["share_oracle_absent"], 3 / 20)
        self.assertAlmostEqual(gripper["share_missed"], 5 / 20)       # absent while the oracle is present: ticks 0..4
        self.assertAlmostEqual(gripper["share_false_alarm"], 3 / 20)  # present while the oracle is absent: ticks 15..17
        self.assertAlmostEqual(front["share_below_half"], 5 / 20)
        self.assertEqual(sum(front["iou_histogram"]), 20)
        self.assertEqual(len(front["iou_histogram"]), analyse.IOU_BINS)

    def test_the_replan_jitter_uses_every_tenth_tick_in_pixels(self):
        oracle = oracle_sequence(3, ticks=31)
        predicted = oracle.copy()
        predicted[10, 0, :4] += 4 / 512   # a 4 px jump at replan tick 10 for the front camera: |d| = 4 px on every coordinate
        predicted[20, 0, :4] += 4 / 512
        self.trial(1, "A", oracle, predicted)
        got = analyse.detector_error(self.logs, self.stage)["overall"]["front"]
        # replans at ticks 0, 10, 20, 30: steps 0->10 (+4), 10->20 (+4 -4 -> 0 + oracle drift), 20->30 (-4)
        drift = 10 * 0.002 * 512
        self.assertAlmostEqual(got["oracle_replan_jitter_px"], drift)
        self.assertAlmostEqual(got["replan_jitter_px"], (abs(drift + 4) + abs(drift) + abs(drift - 4)) / 3)

    def test_a_trial_is_matched_to_its_trace_by_the_oracle_sequence_whatever_the_log_order_or_file(self):
        one, two = oracle_sequence(4), oracle_sequence(5)
        self.trial(1, "A", one, one.copy(), logfile="shard1.jsonl", name="p9-r2")
        self.trial(2, "B", two, np.clip(two + 0.2, 0, 1), logfile="shard0.jsonl", name="p3-r1")
        got = analyse.detector_error(self.logs, self.stage)
        by_layout = {t["layout_id"]: t for t in got["per_trial"]}
        self.assertEqual(by_layout["A"]["front"]["mean_iou"], 1.0)
        self.assertLess(by_layout["B"]["front"]["mean_iou"], 1.0)
        self.assertEqual(got["trials_matched"], 2)

    def test_lines_of_several_workers_under_one_trial_id_are_assigned_by_oracle_box(self):
        """The real A5 run: six containers, each pid 1, appended to one file under one trial id, ticks interleaved."""
        one, two = oracle_sequence(20, ticks=30), oracle_sequence(21, ticks=40)
        pred_one, pred_two = one.copy(), np.clip(two + 0.2, 0, 1)
        for ordinal, (layout, oracle) in enumerate((("A", one), ("B", two)), start=1):
            folder = self.stage / f"trial_{ordinal:03d}_{layout}"
            folder.mkdir()
            np.savez(folder / "trace.npz", target_box=oracle)
            (folder / "episode.json").write_text(json.dumps(dict(ordinal=ordinal, layout_id=layout, task_success=layout == "A")))
        lines = []
        for t in range(40):
            for oracle, predicted in ((one, pred_one), (two, pred_two)):
                if t < len(oracle):
                    lines.append(json.dumps(dict(trial="p1-r1", tick=t, predicted=predicted[t].tolist(), oracle=oracle[t].tolist())))
        (self.logs / "worker_1.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "ticks are not"):
            analyse.detector_error(self.logs, self.stage)
        got = analyse.detector_error(self.logs, self.stage, interleaved=True)
        by_layout = {t["layout_id"]: t for t in got["per_trial"]}
        self.assertEqual(got["trials_matched"], 2)
        self.assertEqual(got["log_groups_without_trace"], 0)
        self.assertEqual(by_layout["A"]["front"]["mean_iou"], 1.0)  # trial A was assigned exactly its own lines
        self.assertLess(by_layout["B"]["front"]["mean_iou"], by_layout["A"]["front"]["mean_iou"])
        self.assertEqual((by_layout["A"]["ticks"], by_layout["B"]["ticks"]), (30, 40))

    def test_a_trace_without_a_log_or_a_log_without_a_trace_is_reported_not_ignored(self):
        oracle = oracle_sequence(6)
        self.trial(1, "A", oracle, oracle.copy())
        orphan = oracle_sequence(7)
        with (self.logs / "shard0.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(trial="p1-r99", tick=0, predicted=orphan[0].tolist(), oracle=orphan[0].tolist())) + "\n")
        trace_only = self.stage / "trial_002_B"
        trace_only.mkdir()
        np.savez(trace_only / "trace.npz", target_box=oracle_sequence(8))
        (trace_only / "episode.json").write_text(json.dumps(dict(ordinal=2, layout_id="B", task_success=False)))
        got = analyse.detector_error(self.logs, self.stage)
        self.assertEqual(got["trials_matched"], 1)
        self.assertEqual(got["trials_without_log"], ["B"])
        self.assertEqual(got["log_groups_without_trace"], 1)

    def test_a_log_that_disagrees_with_its_trace_is_refused(self):
        oracle = oracle_sequence(9)
        self.trial(1, "A", oracle, oracle.copy())
        trace = np.load(self.stage / "trial_001_A/trace.npz")["target_box"]
        np.savez(self.stage / "trial_001_A/trace.npz", target_box=trace[:10])  # a trace of other length: it is another trial
        got = analyse.detector_error(self.logs, self.stage)
        self.assertEqual(got["trials_matched"], 0)
        self.assertEqual(got["trials_without_log"], ["A"])

    def test_the_split_by_outcome(self):
        a, b = oracle_sequence(10), oracle_sequence(11)
        self.trial(1, "A", a, a.copy(), success=True)
        self.trial(2, "B", b, np.clip(b + 0.2, 0, 1), success=False)
        got = analyse.detector_error(self.logs, self.stage)
        self.assertEqual(got["by_outcome"]["success"]["front"]["mean_iou"], 1.0)
        self.assertLess(got["by_outcome"]["failure"]["front"]["mean_iou"], 1.0)
        self.assertEqual((got["by_outcome"]["success"]["trials"], got["by_outcome"]["failure"]["trials"]), (1, 1))


class Campaign(unittest.TestCase):
    """A synthetic campaign root: prepared records and complete comparison stages."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.n = 60
        self.sets = {("A2", 88830): set(range(30)), ("A2", 101520): set(range(28)), ("A3", 101520): set(range(58)),
                     ("A5", 101520): set(range(45)), ("A6", 101520): set(range(55)), ("A7", 101520): set(range(48)),
                     ("A8", 101520): set(range(40)), ("A9", 101520): set(range(20))}
        for (arm, checkpoint), successes in self.sets.items():
            self.write(arm, checkpoint, successes)

    def write(self, arm, checkpoint, successes, core=CORE, dropped=("D1",), status="complete", n=None, shift=0):
        n = self.n if n is None else n
        directory = self.root / "arms" / arm
        (directory / "comparison" / f"step_{checkpoint:06d}").mkdir(parents=True, exist_ok=True)
        prepared = directory / "prepared.json"
        if not prepared.exists():
            prepared.write_text(json.dumps(dict(core=dict(sha256=core), arm=dict(name=arm), phase_checkpoint=101520)))
        episodes = rows(successes, n=n)
        for r in episodes:
            r["layout_id"] = f"L{int(r['layout_id'][1:]) + shift:03d}"
        (directory / "comparison" / f"step_{checkpoint:06d}" / "result.json").write_text(json.dumps(dict(
            status=status, arm=arm, tier="comparison", checkpoint=checkpoint, core_sha256=core, planned_trials=n,
            dropped_layouts=list(dropped), episodes=episodes, missing_ordinals=[] if status == "complete" else [1])))

    def analyse(self, **kwargs):
        return analyse.analyse(self.root, verify=False, resamples=300, **kwargs)

    def test_the_whole_analysis_on_a_synthetic_campaign(self):
        got = self.analyse()
        self.assertEqual((got["floor"], got["ceiling"], got["a5"]), (30, 58, 45))
        self.assertAlmostEqual(got["recovery"]["R"], (45 - 30) / (58 - 30))
        self.assertAlmostEqual(got["recovery_matched_phase"]["R"], (45 - 28) / (58 - 28))
        self.assertEqual(got["a5_vs_a2"]["new_only"], 15)
        self.assertEqual(got["a5_vs_a3"]["control_only"], 13)
        self.assertEqual(sorted(got["corruptions"]), ["A6", "A7", "A8", "A9"])
        self.assertEqual(got["corruptions"]["A6"]["successes"], 55)
        self.assertEqual(got["corruptions"]["A9"]["vs_a3"]["control_only"], 38)
        self.assertEqual(sorted(got["holm"]["adjusted"]), ["A6", "A7", "A8", "A9"])
        self.assertIn("branch", got["decision"])
        self.assertEqual(got["core_sha256"], CORE)
        self.assertEqual(got["a9_reading"], analyse.a9_reading(20 / 60))

    def test_an_arm_that_was_not_run_is_absent_not_zero(self):
        (self.root / "arms/A9/comparison/step_101520/result.json").unlink()
        got = self.analyse()
        self.assertNotIn("A9", got["corruptions"])
        self.assertEqual(got["not_run"], ["A9"])
        self.assertEqual(got["holm"]["raw"]["A9"], 1.0)

    def test_it_refuses_without_a_complete_a5_a2_or_a3_stage(self):
        for arm, checkpoint in (("A5", 101520), ("A2", 88830), ("A3", 101520), ("A2", 101520)):
            self.write(arm, checkpoint, self.sets[(arm, checkpoint)], status="incomplete")
            with self.assertRaises(RuntimeError):
                self.analyse()
            self.write(arm, checkpoint, self.sets[(arm, checkpoint)])
        (self.root / "arms/A3/comparison/step_101520/result.json").unlink()
        with self.assertRaises(RuntimeError):
            self.analyse()

    def test_it_refuses_another_core_in_any_arm(self):
        # the prepared record alone, then the stage alone: each check must stand on its own
        (self.root / "arms/A7/prepared.json").write_text(json.dumps(dict(core=dict(sha256="0" * 64), arm=dict(name="A7"), phase_checkpoint=101520)))
        with self.assertRaisesRegex(ValueError, "core"):
            self.analyse()
        (self.root / "arms/A7/prepared.json").write_text(json.dumps(dict(core=dict(sha256=CORE), arm=dict(name="A7"), phase_checkpoint=101520)))
        self.write("A7", 101520, self.sets[("A7", 101520)], core="0" * 64)
        with self.assertRaisesRegex(ValueError, "core"):
            self.analyse()

    def test_it_refuses_other_layouts_or_other_dropped_layouts(self):
        self.write("A8", 101520, self.sets[("A8", 101520)], shift=1)
        with self.assertRaisesRegex(ValueError, "layouts"):
            self.analyse()
        self.write("A8", 101520, self.sets[("A8", 101520)])
        self.write("A6", 101520, self.sets[("A6", 101520)], dropped=("D1", "D2"))
        with self.assertRaisesRegex(ValueError, "dropped"):
            self.analyse()

    def test_it_refuses_stages_at_another_checkpoint(self):
        self.write("A5", 88830, self.sets[("A5", 101520)])
        (self.root / "arms/A5/comparison/step_101520/result.json").unlink()
        with self.assertRaises(RuntimeError):
            self.analyse()

    def test_verification_checks_the_core_the_manifests_and_the_arm_of_every_arm_it_loads(self):
        calls = []
        record = lambda kind: (lambda prepared, *args, **kwargs: calls.append((kind, prepared["arm"]["name"])))
        with patch.object(analyse.evaluate, "verify_core", side_effect=record("core")), \
                patch.object(analyse.evaluate, "verify_manifests", side_effect=lambda root, prepared: calls.append(("manifests", prepared["arm"]["name"]))), \
                patch.object(analyse.evaluate, "verify_arm", side_effect=record("arm")):
            analyse.analyse(self.root, verify=True, resamples=50)
        for kind in ("core", "manifests", "arm"):
            self.assertEqual(sorted({name for k, name in calls if k == kind}), ["A2", "A3", "A5", "A6", "A7", "A8", "A9"], kind)
        with patch.object(analyse.evaluate, "verify_core", side_effect=ValueError("the shared core changed")):
            with self.assertRaisesRegex(ValueError, "shared core changed"):
                analyse.analyse(self.root, verify=True, resamples=50)

    def test_the_command_line_writes_one_new_file_and_never_replaces_it(self):
        out = self.root / "m3b_analysis.json"
        self.assertEqual(analyse.main(["--root", str(self.root), "analyse", "--no-verify", "--resamples", "200"]), 0)
        written = json.loads(out.read_text())
        self.assertIn("recovery", written)
        with self.assertRaises(FileExistsError):
            analyse.main(["--root", str(self.root), "analyse", "--no-verify", "--resamples", "200"])
        self.assertEqual(json.loads(out.read_text()), written)

    def test_it_never_writes_inside_the_arm_directories_it_reads(self):
        before = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))
        self.analyse()
        analyse.main(["--root", str(self.root), "analyse", "--no-verify", "--resamples", "100"])
        after = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))
        self.assertEqual(sorted(set(after) - set(before)), ["m3b_analysis.json"])


if __name__ == "__main__":
    unittest.main()
