import math
import unittest
from collections import Counter

from experiments.props_p2_data import plan as p

PLAN = p.make_plan()


def _layouts(plan=PLAN):
    """Every drawn layout in the plan: collected episodes, reserved layouts and gate attempts."""
    return (list(plan["episodes"]) + list(plan["evaluation_holdout"].values()) + list(plan["gate_attempts"]))


def _poses(layout):
    return [(layout["object"], layout["start"])] + [(d["object"], d["pose"]) for d in layout["distractors"]]


class CountsTests(unittest.TestCase):
    def test_cells_and_counts(self):
        self.assertEqual(len(p.TRAINED_CELLS), 20)
        self.assertEqual(len(p.HELD_OUT_CELLS), 4)
        self.assertEqual((p.TRAIN_COUNT, p.VALIDATION_COUNT, p.ADMIT_COUNT), (480, 60, 540))
        self.assertEqual(len(PLAN["episodes"]), 540)
        self.assertEqual([e["episode_index"] for e in PLAN["episodes"]], list(range(540)))
        self.assertEqual(PLAN["train_ids"], list(range(480)))
        self.assertEqual(PLAN["validation_ids"], list(range(480, 540)))

    def test_every_trained_cell_has_12_one_and_12_two_distractor_training_episodes(self):
        train = Counter((tuple(e["cell"]), e["distractor_count"]) for e in PLAN["episodes"] if e["split"] == "train")
        self.assertEqual(train, {(c, n): 12 for c in p.TRAINED_CELLS for n in (1, 2)})
        validation = [e for e in PLAN["episodes"] if e["split"] == "validation"]
        self.assertEqual(Counter(tuple(e["cell"]) for e in validation), {c: 3 for c in p.TRAINED_CELLS})
        self.assertEqual(Counter(e["distractor_count"] for e in validation), {1: 30, 2: 30})

    def test_reserved_tiers(self):
        holdout = PLAN["evaluation_holdout"]
        self.assertEqual(len(holdout), 120)
        tiers = Counter(v["tier"] for v in holdout.values())
        self.assertEqual(tiers, {"screen": 20, "report": 60, "control": 20, "confirm": 20})
        for tier in ("screen", "report", "control", "confirm"):
            self.assertEqual(sorted(PLAN[f"evaluation_{tier}_ids"]),
                             sorted(k for k, v in holdout.items() if v["tier"] == tier))
        self.assertTrue(all(v["distractor_count"] == 0 for v in holdout.values() if v["tier"] == "control"))
        report = [v for v in holdout.values() if v["tier"] == "report"]
        trained = Counter(v["distractor_count"] for v in report if not v["held_out"])
        self.assertEqual(trained, {1: 20, 2: 20})
        self.assertEqual(sum(v["held_out"] for v in report), 20)
        for tier in ("screen", "confirm"):
            counts = Counter(v["distractor_count"] for v in holdout.values() if v["tier"] == tier)
            self.assertEqual(counts, {1: 10, 2: 10})

    def test_gate_attempts_cover_every_cell_and_count(self):
        gate = PLAN["gate_attempts"]
        self.assertEqual(len(gate), 288)
        self.assertEqual(Counter((tuple(g["cell"]), g["distractor_count"]) for g in gate),
                         {(c, n): 6 for c in p.CELLS for n in (1, 2)})


class ReviewFocusTests(unittest.TestCase):
    def test_held_out_pairs_are_never_collected_targets_but_are_reported_in_both_arrangements(self):
        held_out = {tuple(x) for x in p.HELD_OUT_PAIRS}
        self.assertFalse(any((e["object"], e["bin"]) in held_out for e in PLAN["episodes"]))
        report = {(v["object"], v["bin"], v["blue_slot"]) for v in PLAN["evaluation_holdout"].values()
                  if v["tier"] == "report"}
        for pair in held_out:
            for slot in p.SLOTS:
                self.assertIn((*pair, slot), report)

    def test_held_out_objects_still_appear_as_distractors_in_training(self):
        seen = Counter(d["object"] for e in PLAN["episodes"] if e["split"] == "train" for d in e["distractors"])
        self.assertEqual(set(seen), set(p.OBJECTS))

    def test_colour_predicts_slot_equally_often(self):
        train = [e for e in PLAN["episodes"] if e["split"] == "train"]
        for colour in p.BIN_COLOURS:
            by_slot = Counter(e["blue_slot"] for e in train if e["bin"] == colour)
            self.assertEqual(by_slot["left"], by_slot["right"])

    def test_distractors_are_distinct_other_objects_and_balanced(self):
        for layout in _layouts():
            names = [layout["object"]] + [d["object"] for d in layout["distractors"]]
            self.assertEqual(len(names), len(set(names)), layout["layout_id"])
            self.assertEqual(len(layout["distractors"]), layout["distractor_count"])
        seen = Counter(d["object"] for e in PLAN["episodes"] if e["split"] == "train" for d in e["distractors"])
        mean = sum(seen.values()) / len(seen)
        for name, n in seen.items():
            self.assertLessEqual(abs(n - mean) / mean, 0.15, (name, n, mean))

    def test_every_layout_is_inside_the_region_and_spaced(self):
        for layout in _layouts():
            poses = _poses(layout)
            for name, (x, y, yaw) in poses:
                self.assertTrue(p.START_X_M[0] <= x <= p.START_X_M[1], layout["layout_id"])
                self.assertTrue(p.START_Y_M[0] <= y <= p.START_Y_M[1], layout["layout_id"])
                self.assertGreaterEqual(math.hypot(x, y), p.START_RADIUS_MIN_M - 1e-12)
                self.assertLessEqual(abs(yaw), p.YAW_MAX_DEG_BY_OBJECT[name] + 1e-12)
            self.assertTrue(p.spaced([n for n, _ in poses], [q for _, q in poses]), layout["layout_id"])
            for i, (a, pa) in enumerate(poses):
                for b, pb in poses[i + 1:]:
                    self.assertGreaterEqual(math.hypot(pa[0] - pb[0], pa[1] - pb[1]),
                                            p.min_distance(a, b) - 1e-12, layout["layout_id"])
                    self.assertTrue(p.hand_clear(a, pa, b, pb) and p.hand_clear(b, pb, a, pa), layout["layout_id"])

    def test_hand_clearance_rejects_the_measured_gate_clash_geometry(self):
        # Gate 2026-10-13: a can target with milk beside it along the finger axis was clipped by the housing.
        self.assertFalse(p.hand_clear("can", [0.33, 0.0, 0.0], "milk", [0.33, 0.09, 0.0]))
        self.assertTrue(p.hand_clear("can", [0.33, 0.0, 0.0], "milk", [0.33 + 0.09, 0.0, 0.0]))
        # A neighbour lower than the housing's underside is never in the way.
        self.assertTrue(p.hand_clear("milk", [0.33, 0.0, 0.0], "bread", [0.33, 0.09, 0.0]))
        # The square milk is checked on both candidate axes.
        self.assertEqual(len(p.finger_axes_deg("milk", 10.0)), 2)
        self.assertEqual(p.finger_axes_deg("cereal", 10.0), [100.0])
        self.assertEqual(p.finger_axes_deg("can", 10.0), [90.0])

    def test_spaced_is_symmetric_in_object_order(self):
        for layout in _layouts()[:200]:
            poses = _poses(layout)
            names, qs = [n for n, _ in poses], [q for _, q in poses]
            self.assertEqual(p.spaced(names, qs), p.spaced(names[::-1], qs[::-1]))

    def test_spacing_rule_is_symmetric_so_it_cannot_mark_the_target(self):
        for a in p.OBJECTS:
            for b in p.OBJECTS:
                self.assertEqual(p.min_distance(a, b), p.min_distance(b, a))
                self.assertGreaterEqual(p.min_distance(a, b), p.FINGER_REACH_M + max(p.FOOTPRINT_RADIUS_M[a],
                                                                                    p.FOOTPRINT_RADIUS_M[b]))
                self.assertGreaterEqual(p.min_distance(a, b),
                                        p.FOOTPRINT_RADIUS_M[a] + p.FOOTPRINT_RADIUS_M[b] + p.PAIR_GAP_M)

    def test_resampled_layouts_keep_identities_and_stay_valid(self):
        for episode in PLAN["episodes"][:40]:
            for r in (1, 2, 3):
                resampled = p.resample_start(episode, r)
                self.assertEqual([d["object"] for d in resampled["distractors"]],
                                 [d["object"] for d in episode["distractors"]])
                self.assertNotEqual(resampled["start"], episode["start"])
                self.assertEqual(resampled["layout_id"], f"{episode['layout_id']}_r{r}")
                poses = _poses(resampled)
                self.assertTrue(p.spaced([n for n, _ in poses], [q for _, q in poses]))

    def test_no_reserved_layout_coincides_with_a_collected_one(self):
        collected = {tuple(e["start"]) for e in PLAN["episodes"]}
        self.assertFalse(collected & {tuple(v["start"]) for v in PLAN["evaluation_holdout"].values()})


class DeterminismShardsAndDrawsTests(unittest.TestCase):
    def test_plan_is_deterministic_and_validate_plan_rejects_a_tamper(self):
        self.assertEqual(p.digest(PLAN), p.digest(p.make_plan()))
        p.validate_plan(PLAN)
        with self.assertRaises(ValueError):
            p.validate_plan({**PLAN, "episodes": PLAN["episodes"][:-1]})

    def test_shards_partition_the_slots_into_contiguous_blocks(self):
        blocks = [p.shard_slots(k, 12) for k in range(12)]
        self.assertEqual([s for block in blocks for s in block], list(range(540)))
        self.assertTrue(all(len(block) == 45 for block in blocks))
        with self.assertRaises(ValueError):
            p.shard_slots(12, 12)

    def test_attempt_draw_and_resample_indices(self):
        episode = PLAN["episodes"][0]
        self.assertEqual(p.attempt_draw(episode, 0)["seed"], episode["seed"])
        self.assertNotEqual(p.attempt_draw(episode, 1)["seed"], episode["seed"])
        self.assertIs(p.resample_start(episode, 0), episode)
        with self.assertRaises(ValueError):
            p.attempt_draw(episode, -1)
        with self.assertRaises(ValueError):
            p.resample_start(episode, -1)

    def test_instruction_template(self):
        for e in PLAN["episodes"]:
            self.assertEqual(e["instruction"], f"put the {e['object']} in the {e['bin']} bin")


if __name__ == "__main__":
    unittest.main()
