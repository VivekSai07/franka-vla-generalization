import json
import unittest
from collections import Counter
from pathlib import Path

from experiments.props_p2_data import plan as p2
from experiments.vla_inputs_eval import plan
from experiments.xvla_props_p2_retest import plan as retest

P2_ROOT = Path(__file__).resolve().parents[3] / "outputs/props_p2_data/props_p2_20261013"

TRAINED_PAIRS = set(p2.TRAINED_PAIRS)
HELD_OUT_PAIRS = set(p2.HELD_OUT_PAIRS)


def dumps(value):
    return json.dumps(value, sort_keys=True)


class Counts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan.make_plan()
        cls.layouts = cls.plan["layouts"]

    def tier(self, name):
        return [self.layouts[i] for i in self.plan["tiers"][name]]

    def test_tier_sizes(self):
        self.assertEqual({k: len(v) for k, v in self.plan["tiers"].items()}, {"smoke": 2, "selection": 80, "comparison": 576})
        self.assertEqual(len(self.layouts), 80 + 576)

    def test_selection_is_two_per_trained_cell_and_never_held_out(self):
        counts = Counter(tuple(l["cell"]) for l in self.tier("selection"))
        self.assertEqual(len(counts), 40)
        self.assertEqual(set(counts.values()), {2})
        self.assertFalse(any(l["held_out"] for l in self.tier("selection")))
        self.assertTrue(all((l["object"], l["bin"]) in TRAINED_PAIRS for l in self.tier("selection")))

    def test_comparison_is_twelve_per_cell_over_all_48_cells(self):
        counts = Counter(tuple(l["cell"]) for l in self.tier("comparison"))
        self.assertEqual(len(counts), 48)
        self.assertEqual(set(counts.values()), {12})
        held = [l for l in self.tier("comparison") if l["held_out"]]
        self.assertEqual(len(held), 96)
        self.assertTrue(all((l["object"], l["bin"]) in HELD_OUT_PAIRS for l in held))

    def test_a_cell_is_object_bin_side_and_distractor_count(self):
        for l in self.layouts.values():
            self.assertEqual(l["cell"], [l["object"], l["bin"], l["blue_slot"], l["distractor_count"]])
            self.assertIn(l["distractor_count"], (1, 2))
            self.assertEqual(len(l["distractors"]), l["distractor_count"])
            self.assertNotIn(l["object"], [d["object"] for d in l["distractors"]])

    def test_smoke_is_inside_selection(self):
        self.assertTrue(set(self.plan["tiers"]["smoke"]) <= set(self.plan["tiers"]["selection"]))

    def test_selection_and_comparison_share_no_layout(self):
        self.assertFalse(set(self.plan["tiers"]["selection"]) & set(self.plan["tiers"]["comparison"]))

    def test_identities_use_the_m0_namespace_and_tier(self):
        for tier, ids in self.plan["tiers"].items():
            if tier == "smoke":
                continue
            for i in ids:
                self.assertTrue(self.layouts[i]["identity"].startswith(f"vla_inputs:m0:{tier}:"), i)
                self.assertEqual(self.layouts[i]["tier"], tier)

    def test_every_seed_is_distinct(self):
        seeds = [l["seed"] for l in self.layouts.values()]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(len(set(self.plan["rng_seeds"].values())), len(self.plan["rng_seeds"]))

    def test_distractor_decks_show_every_option_before_any_repeat(self):
        for tier in ("selection", "comparison"):
            groups = {}
            for l in self.tier(tier):
                groups.setdefault(tuple(l["cell"]), []).append(sorted(d["object"] for d in l["distractors"]))
            for cell, sets in groups.items():
                options = len(p2._options(cell[0], cell[3]))
                first = sets[:options]
                if len(sets) >= options:
                    self.assertEqual(len({tuple(s) for s in first}), options, (tier, cell))
                else:
                    self.assertEqual(len({tuple(s) for s in sets}), len(sets), (tier, cell))

    def test_decks_are_shuffled_not_a_fixed_order(self):
        orders = set()
        for cell in plan.COMPARISON_CELLS:
            if cell[0] == "can" and cell[3] == 1:
                ids = [l for l in self.tier("comparison") if tuple(l["cell"]) == cell]
                orders.add(tuple(d["object"] for l in ids[:5] for d in l["distractors"]))
        self.assertGreater(len(orders), 1)
        sets = [tuple(sorted(d["object"] for d in l["distractors"])) for l in self.tier("comparison")
                if tuple(l["cell"]) == ("can", "blue", "left", 1)]
        self.assertNotEqual(sets[:5], sets[5:10])  # the second pass is a fresh shuffle

    def test_every_layout_satisfies_p2_spacing(self):
        for l in self.layouts.values():
            names = [l["object"], *(d["object"] for d in l["distractors"])]
            poses = [l["start"], *(d["pose"] for d in l["distractors"])]
            self.assertTrue(p2.spaced(names, poses), l["layout_id"])


class Deterministic(unittest.TestCase):
    def test_two_calls_are_byte_equal(self):
        self.assertEqual(dumps(plan.make_plan()), dumps(plan.make_plan()))


def p2_values():
    """Every seed and every pose in the sealed P2 plan and the P2 re-test plan."""
    sealed = json.loads((P2_ROOT / "plan.json").read_text())
    rows = [*sealed["episodes"], *sealed["evaluation_holdout"].values(), *sealed["gate_attempts"]]
    retests = list(retest.make_plan()["evaluation_holdout"].values())
    return sealed, rows + retests


class DisjointFromP2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sealed, cls.rows = p2_values()
        cls.plan = plan.make_plan()

    def test_the_regenerated_p2_plan_is_the_sealed_one(self):
        self.assertEqual(p2.digest(p2.make_plan()), json.loads((P2_ROOT / "seal.json").read_text())["plan_sha256"])

    def test_no_seed_coincides_with_a_p2_or_retest_seed(self):
        seeds = {r["seed"] for r in self.rows} | set(self.sealed["rng_seeds"].values()) | set(retest.make_plan()["seeds"].values())
        ours = {l["seed"] for l in self.plan["layouts"].values()} | set(self.plan["rng_seeds"].values())
        self.assertFalse(seeds & ours)

    def test_no_pose_coincides_with_a_p2_or_retest_pose(self):
        poses = set()
        for r in self.rows:
            poses.add(tuple(r["start"]))
            poses |= {tuple(d["pose"]) for d in r["distractors"]}
        ours = set()
        for l in self.plan["layouts"].values():
            ours.add(tuple(l["start"]))
            ours |= {tuple(d["pose"]) for d in l["distractors"]}
        self.assertFalse(poses & ours)

    def test_a_forced_collision_with_a_real_p2_layout_is_refused(self):
        forbidden = plan.forbidden_values()
        real = self.sealed["evaluation_holdout"]["report_000"]
        for field, value in (("seed", real["seed"]), ("start", real["start"])):
            layouts = json.loads(dumps(self.plan["layouts"]))
            victim = next(iter(layouts.values()))
            victim[field] = value
            with self.assertRaises(ValueError, msg=field):
                plan.check_disjoint(layouts, forbidden)
        layouts = json.loads(dumps(self.plan["layouts"]))
        victim = next(l for l in layouts.values() if l["distractors"])
        victim["distractors"][0]["pose"] = real["distractors"][0]["pose"]
        with self.assertRaises(ValueError):
            plan.check_disjoint(layouts, forbidden)

    def test_the_unmodified_plan_passes_the_check(self):
        plan.check_disjoint(self.plan["layouts"], plan.forbidden_values())

    def test_make_plan_refuses_when_its_own_draw_is_forbidden(self):
        own = plan.make_plan()
        first = next(iter(own["layouts"].values()))
        forbidden = plan.forbidden_values()
        forbidden["seeds"] = set(forbidden["seeds"]) | {first["seed"]}
        with self.assertRaises(ValueError):
            plan.make_plan(forbidden=forbidden)


class Write(unittest.TestCase):
    def test_writes_plan_and_digest_and_never_overwrites(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan"
            written = plan.write(out)
            self.assertEqual(written, p2.digest(json.loads((out / "plan.json").read_text())))
            self.assertEqual((out / "plan.sha256").read_text().strip(), written)
            with self.assertRaises(FileExistsError):
                plan.write(out)


if __name__ == "__main__":
    unittest.main()
