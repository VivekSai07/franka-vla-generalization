"""Stage-one contracts; never execute a real expert trajectory in CPU tests."""
import copy
import unittest


class PairedPlanTests(unittest.TestCase):
    def test_schedule_counts_and_literal_identities(self):
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        self.assertEqual(len(plan["layouts"]), 18)
        self.assertEqual(len(plan["episodes"]), 36)
        self.assertEqual(plan["layouts"][0]["layout_id"], "train/red-green/0")
        self.assertEqual(plan["layouts"][-1]["layout_id"], "heldout/blue-yellow/0")
        self.assertEqual(plan["episodes"][0]["instruction"],
                         "pick up the red cube and place it on top of the green cube")
        self.assertEqual(plan["episodes"][1]["pick"], "green")
        self.assertEqual(plan["episodes"][-1]["place"], "blue")
        self.assertEqual(plan["subsets"]["paired"], list(range(24)))
        self.assertEqual(plan["subsets"]["heldout"], list(range(24, 36)))
        self.assertEqual(plan["subsets"]["control"], [0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23])

    def test_family_splits_and_control_cover_every_ordered_pair(self):
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        selected = [plan["episodes"][i] for i in plan["subsets"]["control"]]
        self.assertEqual(len({(e["pick"], e["place"]) for e in selected}), 12)
        for layout in plan["layouts"]:
            episodes = [e for e in plan["episodes"] if e["layout_id"] == layout["layout_id"]]
            self.assertEqual(len(episodes), 2)
            self.assertEqual({e["split"] for e in episodes}, {layout["split"]})
            self.assertEqual([e["direction"] for e in episodes], ["forward", "reverse"])
            self.assertEqual(episodes[0]["pick"], episodes[1]["place"])
            self.assertEqual(episodes[0]["place"], episodes[1]["pick"])

    def test_seed_streams_are_prerecorded_unique_and_repeatable(self):
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        self.assertEqual(plan, make_plan())
        seeds = [e["seed"] for e in plan["episodes"]] + [l["seed"] for l in plan["layouts"]]
        self.assertEqual(len(set(seeds)), 54)
        self.assertTrue(all(type(s) is int and 0 <= s < 2**32 for s in seeds))
        self.assertEqual(plan["master_seed"], 20260917)

    def test_mutated_schedule_or_types_cannot_be_approved(self):
        from experiments.paired_demonstrations.plan import make_plan, validate_plan
        validate_plan(make_plan())
        for mutation in ("seed", "direction", "split", "missing", "control", "bool"):
            plan = make_plan()
            if mutation == "seed": plan["episodes"][0]["seed"] += 1
            if mutation == "direction": plan["episodes"][0]["place"] = "blue"
            if mutation == "split": plan["layouts"][0]["split"] = "heldout"
            if mutation == "missing": plan["episodes"].pop()
            if mutation == "control": plan["subsets"]["control"][-1] = 22
            if mutation == "bool": plan["episodes"][0]["episode_index"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_plan(plan)

    def test_complete_chunks_never_pad_or_cross_episode_boundary(self):
        from experiments.paired_demonstrations.plan import complete_starts
        self.assertEqual(list(complete_starts(528)), list(range(479)))
        self.assertEqual(list(complete_starts(50)), [0])
        self.assertEqual(list(complete_starts(49)), [])
        for length in (-1, True, 1.2):
            with self.assertRaises(ValueError): complete_starts(length)


if __name__ == "__main__":
    unittest.main()