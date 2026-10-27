import json
import math
import unittest
from collections import Counter

from experiments.props_p3_data import plan as p

PLAN = p.make_plan()

# Written out here, not read from the code under test: the spec's mapping and split (D1, D4).
EXPECTED_BIN = {"can": "orange", "milk": "orange", "bottle": "orange", "bread": "blue", "lemon": "blue", "cereal": "blue"}
TRAINED = ("can", "milk", "bread", "lemon")
HELD_OUT = ("bottle", "cereal")


def _layouts(plan=PLAN):
    """Every drawn layout in the plan: collected episodes, reserved layouts and gate attempts."""
    return list(plan["episodes"]) + list(plan["evaluation_holdout"].values()) + list(plan["gate_attempts"])


class CountsTests(unittest.TestCase):
    def test_cells_and_counts(self):
        self.assertEqual(len(p.CELLS), 12)
        self.assertEqual(len(p.TRAINED_CELLS), 8)
        self.assertEqual(len(p.HELD_OUT_CELLS), 4)
        self.assertEqual((p.TRAIN_COUNT, p.VALIDATION_COUNT, p.ADMIT_COUNT), (480, 64, 544))
        self.assertEqual([e["episode_index"] for e in PLAN["episodes"]], list(range(544)))
        self.assertEqual(PLAN["train_ids"], list(range(480)))
        self.assertEqual(PLAN["validation_ids"], list(range(480, 544)))
        self.assertEqual([e["split"] for e in PLAN["episodes"]], ["train"] * 480 + ["validation"] * 64)

    def test_every_trained_cell_has_60_training_and_8_validation_episodes(self):
        train = Counter(tuple(e["cell"]) for e in PLAN["episodes"] if e["split"] == "train")
        validation = Counter(tuple(e["cell"]) for e in PLAN["episodes"] if e["split"] == "validation")
        self.assertEqual(set(train), {(o, EXPECTED_BIN[o], s) for o in TRAINED for s in ("left", "right")})
        self.assertEqual(set(train.values()), {60})
        self.assertEqual(validation, {c: 8 for c in train})

    def test_reserved_tiers(self):
        holdout = PLAN["evaluation_holdout"]
        self.assertEqual(len(holdout), 288)
        self.assertEqual(Counter(v["tier"] for v in holdout.values()), {"screen": 24, "report": 240, "confirm": 24})
        for tier in ("screen", "report", "confirm"):
            self.assertEqual(sorted(PLAN[f"evaluation_{tier}_ids"]), sorted(k for k, v in holdout.items() if v["tier"] == tier))
        self.assertNotIn("evaluation_control_ids", PLAN)
        for tier in ("screen", "confirm"):
            cells = Counter(tuple(v["cell"]) for v in holdout.values() if v["tier"] == tier)
            self.assertEqual(cells, {c: 3 for c in p.TRAINED_CELLS})
        report = [v for v in holdout.values() if v["tier"] == "report"]
        self.assertEqual(Counter(tuple(v["cell"]) for v in report if not v["held_out"]), {c: 10 for c in p.TRAINED_CELLS})
        self.assertEqual(Counter(tuple(v["cell"]) for v in report if v["held_out"]), {c: 40 for c in p.HELD_OUT_CELLS})
        self.assertEqual(Counter(v["object"] for v in report if v["held_out"]), {"bottle": 80, "cereal": 80})
        self.assertEqual(sum(not v["held_out"] for v in report), 80)

    def test_gate_attempts_cover_every_object_and_arrangement_equally(self):
        gate = PLAN["gate_attempts"]
        self.assertEqual(len(gate), 288)
        self.assertEqual(Counter((g["object"], g["blue_slot"]) for g in gate),
                         {(o, s): 24 for o in p.OBJECTS for s in ("left", "right")})


class ReviewFocusTests(unittest.TestCase):
    def test_held_out_objects_are_never_a_collected_target(self):
        # Review Focus 1, against the real plan: no training or validation slot targets bottle or cereal.
        self.assertFalse({e["object"] for e in PLAN["episodes"]} & set(HELD_OUT))
        self.assertEqual({e["object"] for e in PLAN["episodes"]}, set(TRAINED))
        self.assertEqual({tuple(c) for c in PLAN["held_out_cells"]},
                         {(o, EXPECTED_BIN[o], s) for o in HELD_OUT for s in ("left", "right")})

    def test_the_bin_follows_from_the_object_exactly(self):
        # Review Focus 2, and the D1 mapping: drinks -> orange, food -> blue.
        for layout in _layouts():
            self.assertEqual(layout["bin"], EXPECTED_BIN[layout["object"]], layout["layout_id"])
        self.assertEqual({o: p.container_of(o) for o in p.OBJECTS}, EXPECTED_BIN)
        self.assertEqual({o: p.CATEGORY[o] for o in p.OBJECTS}["milk"], "drinks")

    def test_side_does_not_predict_the_container_or_the_object(self):
        for split_layouts in ([e for e in PLAN["episodes"] if e["split"] == "train"],
                              [e for e in PLAN["episodes"] if e["split"] == "validation"],
                              [v for v in PLAN["evaluation_holdout"].values() if v["tier"] == "report"]):
            by_object = Counter((l["object"], l["blue_slot"]) for l in split_layouts)
            for obj in {l["object"] for l in split_layouts}:
                self.assertEqual(by_object[(obj, "left")], by_object[(obj, "right")], obj)
            by_bin = Counter((l["bin"], l["blue_slot"]) for l in split_layouts)
            for colour in p.BIN_COLOURS:
                self.assertEqual(by_bin[(colour, "left")], by_bin[(colour, "right")], colour)

    def test_the_prompt_is_fixed_and_the_expert_wording_never_enters_the_plan(self):
        self.assertEqual(p.PROMPT, "sort the item")
        for layout in _layouts():
            self.assertEqual(layout["instruction"], p.PROMPT)
        self.assertNotIn("put the", json.dumps(PLAN))

    def test_there_is_no_distractor_machinery(self):
        for layout in _layouts():
            self.assertFalse({"distractors", "distractor_count"} & set(layout), layout["layout_id"])
        for name in ("DISTRACTOR_COUNTS", "FOOTPRINT_RADIUS_M", "hand_clear", "spaced", "min_distance"):
            self.assertFalse(hasattr(p, name), name)

    def test_every_layout_is_inside_the_start_region(self):
        for layout in _layouts():
            x, y, yaw = layout["start"]
            self.assertTrue(p.START_X_M[0] <= x <= p.START_X_M[1], layout["layout_id"])
            self.assertTrue(p.START_Y_M[0] <= y <= p.START_Y_M[1], layout["layout_id"])
            self.assertGreaterEqual(math.hypot(x, y), p.START_RADIUS_MIN_M - 1e-12)
            self.assertLessEqual(abs(yaw), p.YAW_MAX_DEG_BY_OBJECT[layout["object"]] + 1e-12)
        self.assertEqual(p.YAW_MAX_DEG_BY_OBJECT["lemon"], 75.0)

    def test_no_reserved_or_gate_layout_coincides_with_a_collected_one(self):
        collected = {tuple(e["start"]) for e in PLAN["episodes"]}
        others = [tuple(v["start"]) for v in PLAN["evaluation_holdout"].values()] + [tuple(g["start"]) for g in PLAN["gate_attempts"]]
        self.assertFalse(collected & set(others))
        self.assertEqual(len(set(others)), len(others))
        self.assertEqual(len(collected), 544)

    def test_resampled_layouts_keep_the_slot_and_stay_valid(self):
        for episode in PLAN["episodes"][:60]:
            for r in (1, 2, 3):
                resampled = p.resample_start(episode, r)
                for key in ("object", "bin", "blue_slot", "cell", "instruction", "episode_index"):
                    self.assertEqual(resampled[key], episode[key])
                self.assertNotEqual(resampled["start"], episode["start"])
                self.assertEqual(resampled["layout_id"], f"{episode['layout_id']}_r{r}")
                x, y, yaw = resampled["start"]
                self.assertGreaterEqual(math.hypot(x, y), p.START_RADIUS_MIN_M - 1e-12)
                self.assertLessEqual(abs(yaw), p.YAW_MAX_DEG_BY_OBJECT[episode["object"]] + 1e-12)


class DeterminismShardsAndDrawsTests(unittest.TestCase):
    def test_plan_is_deterministic_and_validate_plan_rejects_a_tamper(self):
        self.assertEqual(p.digest(PLAN), p.digest(p.make_plan()))
        p.validate_plan(PLAN)
        with self.assertRaises(ValueError):
            p.validate_plan({**PLAN, "episodes": PLAN["episodes"][:-1]})

    def test_shards_partition_the_slots_into_contiguous_blocks(self):
        blocks = [p.shard_slots(k, 16) for k in range(16)]
        self.assertEqual([s for block in blocks for s in block], list(range(544)))
        self.assertTrue(all(len(block) == 34 for block in blocks))
        with self.assertRaises(ValueError):
            p.shard_slots(16, 16)
        with self.assertRaises(ValueError):
            p.shard_slots(0, 12)

    def test_attempt_draw_and_resample_indices(self):
        episode = PLAN["episodes"][0]
        self.assertEqual(p.attempt_draw(episode, 0)["seed"], episode["seed"])
        self.assertNotEqual(p.attempt_draw(episode, 1)["seed"], episode["seed"])
        self.assertIs(p.resample_start(episode, 0), episode)
        with self.assertRaises(ValueError):
            p.attempt_draw(episode, -1)
        with self.assertRaises(ValueError):
            p.resample_start(episode, -1)


if __name__ == "__main__":
    unittest.main()
