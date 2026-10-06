import unittest
from collections import Counter

from experiments.props_p1_data import plan as p


class SizesAndCellsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = p.make_plan()

    def test_pairs_cells_and_held_out(self):
        self.assertEqual(len(p.PAIRS), 12)
        self.assertEqual(len(p.CELLS), 24)
        self.assertEqual(len(p.TRAINED_CELLS), 20)
        self.assertEqual(len(p.HELD_OUT_CELLS), 4)
        self.assertEqual(set(tuple(c) for c in p.HELD_OUT_CELLS),
                         {("milk", "blue", "left"), ("milk", "blue", "right"),
                          ("bread", "orange", "left"), ("bread", "orange", "right")})

    def test_counts(self):
        self.assertEqual(p.TRAIN_COUNT, 480)
        self.assertEqual(p.VALIDATION_COUNT, 60)
        self.assertEqual(p.ADMIT_COUNT, 540)
        self.assertEqual(p.EVAL_SCREEN_COUNT, 20)
        self.assertEqual(p.EVAL_REPORT_COUNT, 40)
        self.assertEqual(p.EVAL_CONFIRM_COUNT, 20)
        self.assertEqual(p.EVAL_COUNT, 80)

    def test_plan_episode_and_holdout_totals(self):
        plan = self.plan
        self.assertEqual(len(plan["episodes"]), 540)
        self.assertEqual(len(plan["train_ids"]), 480)
        self.assertEqual(len(plan["validation_ids"]), 60)
        self.assertEqual(len(plan["evaluation_holdout"]), 80)
        self.assertEqual(len(plan["evaluation_screen_ids"]), 20)
        self.assertEqual(len(plan["evaluation_report_ids"]), 40)
        self.assertEqual(len(plan["evaluation_confirm_ids"]), 20)

    def test_every_trained_cell_gets_the_same_count(self):
        plan = self.plan
        train = Counter(tuple(e["cell"]) for e in plan["episodes"] if e["split"] == "train")
        validation = Counter(tuple(e["cell"]) for e in plan["episodes"] if e["split"] == "validation")
        self.assertEqual(set(train), set(tuple(c) for c in p.TRAINED_CELLS))
        self.assertEqual(set(train.values()), {24})
        self.assertEqual(set(validation.values()), {3})

    def test_instructions_follow_the_put_the_x_in_the_y_bin_template(self):
        # The props_scene expert's exact wording is cross-checked in test_episode.py (robosuite image).
        for obj in p.OBJECTS:
            for colour in p.BIN_COLOURS:
                self.assertEqual(p.instruction(obj, colour), f"put the {obj} in the {colour} bin")


class ReviewFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = p.make_plan()

    def test_instruction_colour_matches_the_target_bin_in_both_arrangements(self):
        # Review Focus 1: the instruction names the bin colour, independent of which slot it sits in.
        for e in self.plan["episodes"]:
            self.assertEqual(e["instruction"], f"put the {e['object']} in the {e['bin']} bin")
        for v in self.plan["evaluation_holdout"].values():
            self.assertEqual(v["instruction"], f"put the {v['object']} in the {v['bin']} bin")

    def test_held_out_pairs_never_appear_in_training_or_validation(self):
        # Review Focus 2.
        held_out = {tuple(p) for p in self.plan["held_out_pairs"]}
        for e in self.plan["episodes"]:
            self.assertNotIn((e["object"], e["bin"]), held_out)
        report_pairs = {(v["object"], v["bin"]) for k, v in self.plan["evaluation_holdout"].items()
                        if v["tier"] == "report"}
        report_arrangements = {(v["object"], v["bin"], v["blue_slot"]) for k, v in self.plan["evaluation_holdout"].items()
                               if v["tier"] == "report"}
        for pair in held_out:
            self.assertIn(pair, report_pairs)
            for slot in ("left", "right"):
                self.assertIn((*pair, slot), report_arrangements)

    def test_colour_predicts_slot_equally_often_in_training(self):
        # Review Focus 3: a policy cannot use position as a substitute for the colour word.
        train = [e for e in self.plan["episodes"] if e["split"] == "train"]
        for colour in p.BIN_COLOURS:
            by_slot = Counter(e["blue_slot"] for e in train if e["bin"] == colour)
            self.assertEqual(by_slot["left"], by_slot["right"])

    def test_no_reserved_start_coincides_with_a_collected_one(self):
        # Review Focus 5 (part): plans are independent draws; nothing should collide exactly.
        collected = {(e["object"], e["bin"], e["blue_slot"], tuple(e["start"])) for e in self.plan["episodes"]}
        reserved = {(v["object"], v["bin"], v["blue_slot"], tuple(v["start"])) for v in self.plan["evaluation_holdout"].values()}
        self.assertEqual(collected & reserved, set())

    def test_every_start_is_inside_the_amended_region(self):
        # Review Focus 5 (part): radius floor and the per-object yaw cap.
        import math
        for e in list(self.plan["episodes"]) + list(self.plan["evaluation_holdout"].values()):
            x, y, yaw = e["start"]
            self.assertGreaterEqual(math.hypot(x, y), p.START_RADIUS_MIN_M - 1e-9, e)
            self.assertLessEqual(abs(yaw), p.YAW_MAX_DEG_BY_OBJECT[e["object"]] + 1e-9, e)
            self.assertEqual(len(e["home_pose_jitter_deg"]), 7)
            self.assertTrue(all(abs(j) <= p.HOME_JITTER_DEG_MAX for j in e["home_pose_jitter_deg"]))


class DeterminismAndAttemptsTests(unittest.TestCase):
    def test_make_plan_is_deterministic_and_validate_plan_rejects_a_tamper(self):
        a, b = p.make_plan(), p.make_plan()
        self.assertEqual(p.digest(a), p.digest(b))
        p.validate_plan(a)
        tampered = {**a, "episodes": list(a["episodes"][:-1])}
        with self.assertRaises(ValueError):
            p.validate_plan(tampered)

    def test_episode_indices_are_unique_and_ordered(self):
        plan = p.make_plan()
        self.assertEqual([e["episode_index"] for e in plan["episodes"]], list(range(540)))
        self.assertEqual(len({e["seed"] for e in plan["episodes"]}), 540)

    def test_attempt_draw_k0_is_the_planned_draw_and_k_gt_0_is_fresh(self):
        episode = p.make_plan()["episodes"][0]
        self.assertEqual(p.attempt_draw(episode, 0),
                         {"seed": episode["seed"], "home_pose_jitter_deg": list(episode["home_pose_jitter_deg"])})
        draw1 = p.attempt_draw(episode, 1)
        self.assertNotEqual(draw1["seed"], episode["seed"])
        self.assertEqual(draw1, p.attempt_draw(episode, 1))
        with self.assertRaises(ValueError):
            p.attempt_draw(episode, -1)

    def test_resample_start_r0_is_unchanged_and_r_gt_0_is_a_fresh_pose(self):
        episode = p.make_plan()["episodes"][0]
        self.assertIs(p.resample_start(episode, 0), episode)
        resampled = p.resample_start(episode, 1)
        self.assertNotEqual(resampled["start"], episode["start"])
        self.assertEqual(resampled["layout_id"], episode["layout_id"] + "_r1")
        self.assertEqual(resampled, p.resample_start(episode, 1))
        with self.assertRaises(ValueError):
            p.resample_start(episode, -1)


if __name__ == "__main__":
    unittest.main()
