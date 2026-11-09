import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.gen_eval import plan as plan_module
from experiments.gen_scene import levels
from experiments.gen_scene.conditions import Condition
from experiments.props_p2_data import plan as p2
from experiments.vla_inputs_eval import plan as m0

_PLAN = {}


def the_plan():
    if not _PLAN:
        _PLAN["plan"] = plan_module.make_plan()
    return _PLAN["plan"]


class Sizes(unittest.TestCase):
    def test_tier_sizes_and_cells(self):
        plan = the_plan()
        self.assertEqual({t: len(ids) for t, ids in plan["tiers"].items()}, {"smoke": 2, "selection": 24, "benchmark": 144})
        self.assertEqual(Counter(tuple(plan["layouts"][i]["cell"]) for i in plan["tiers"]["benchmark"]), {tuple(c): 3 for c in m0.COMPARISON_CELLS})
        selection = [tuple(plan["layouts"][i]["cell"]) for i in plan["tiers"]["selection"]]
        self.assertEqual(selection, list(m0.COMPARISON_CELLS[::2]))
        self.assertEqual(len(set(selection)), 24)
        self.assertEqual(len(plan["layouts"]), 170)

    def test_smoke_layouts_are_two_and_one_distractor_trained_cells(self):
        plan = the_plan()
        smoke = [plan["layouts"][i] for i in plan["tiers"]["smoke"]]
        self.assertEqual([l["distractor_count"] for l in smoke], [2, 1])
        self.assertTrue(all(not l["held_out"] for l in smoke))

    def test_layouts_of_a_cell_are_consecutive_so_rotation_covers_each_cell(self):
        plan = the_plan()
        cells = [tuple(plan["layouts"][i]["cell"]) for i in plan["tiers"]["benchmark"]]
        self.assertEqual(cells, [tuple(c) for c in m0.COMPARISON_CELLS for _ in range(3)])


class Namespace(unittest.TestCase):
    def test_identities_seeds_and_poses_are_fresh(self):
        plan = the_plan()
        self.assertEqual(plan["namespace"], "gen:g0")
        for layout in plan["layouts"].values():
            self.assertTrue(layout["identity"].startswith("gen:g0:"))
            self.assertEqual(layout["tier"], layout["identity"].split(":")[2])
        forbidden = plan_module.forbidden_values()
        for layout in plan["layouts"].values():
            self.assertNotIn(layout["seed"], forbidden["seeds"])
            self.assertNotIn(tuple(layout["start"]), forbidden["poses"])
            self.assertFalse({tuple(d["pose"]) for d in layout["distractors"]} & forbidden["poses"])

    def test_the_forbidden_set_covers_p2_the_retest_m0_and_the_pilot(self):
        forbidden = plan_module.forbidden_values()
        m0_plan = m0.make_plan()
        some = next(iter(m0_plan["layouts"].values()))
        self.assertIn(some["seed"], forbidden["seeds"])
        self.assertIn(tuple(some["start"]), forbidden["poses"])
        from experiments.vla_inputs_pilot_plan import plan as pilot
        pilot_layout = next(iter(pilot.make_plan()["layouts"].values()))
        self.assertIn(pilot_layout["seed"], forbidden["seeds"])
        self.assertIn(tuple(pilot_layout["start"]), forbidden["poses"])
        self.assertIn(tuple(pilot_layout["distractors"][0]["pose"]), forbidden["poses"])
        sealed = json.loads((m0.P2_ROOT / "plan.json").read_text())["episodes"][0]
        self.assertIn(sealed["seed"], forbidden["seeds"])

    def test_no_layout_repeats_a_start_or_a_seed(self):
        plan = the_plan()
        starts = [tuple(l["start"]) for l in plan["layouts"].values()]
        seeds = [l["seed"] for l in plan["layouts"].values()]
        self.assertEqual((len(set(starts)), len(set(seeds))), (170, 170))

    def test_a_forbidden_draw_is_refused_not_redrawn(self):
        plan = the_plan()
        layout = next(iter(plan["layouts"].values()))
        forbidden = {"seeds": {layout["seed"]}, "poses": set()}
        with self.assertRaises(ValueError):
            plan_module.check_disjoint(plan["layouts"], forbidden)
        forbidden = {"seeds": set(), "poses": {tuple(layout["start"])}}
        with self.assertRaises(ValueError):
            plan_module.check_disjoint(plan["layouts"], forbidden)
        forbidden = {"seeds": set(), "poses": {tuple(layout["distractors"][0]["pose"])}}
        with self.assertRaises(ValueError):
            plan_module.check_disjoint(plan["layouts"], forbidden)


class Conditions(unittest.TestCase):
    def test_every_layout_has_one_valid_level_per_axis_by_the_pinned_rotation(self):
        plan = the_plan()
        for tier in ("smoke", "selection", "benchmark"):
            for n, lid in enumerate(plan["tiers"][tier]):
                axes = plan["layouts"][lid]["axes"]
                self.assertEqual(list(axes), list(levels.AXES))
                for axis, entry in axes.items():
                    condition = Condition.from_dict(entry["condition"]).check()
                    self.assertEqual((condition.axis, condition.level, entry["level"]), (axis, levels.level_of(axis, n), levels.level_of(axis, n)))
                    self.assertEqual(entry["condition_sha256"], condition.digest())

    def test_benchmark_level_sizes(self):
        plan = the_plan()
        for axis, per in (("lighting", 36), ("background", 48), ("distractors", 48), ("spatial", 36), ("disturbance", 72)):
            counts = Counter(plan["layouts"][i]["axes"][axis]["level"] for i in plan["tiers"]["benchmark"])
            self.assertEqual(set(counts.values()), {per}, axis)

    def test_selection_level_sizes(self):
        plan = the_plan()
        for axis, per in (("lighting", 6), ("background", 8), ("distractors", 8), ("spatial", 6), ("disturbance", 12)):
            counts = Counter(plan["layouts"][i]["axes"][axis]["level"] for i in plan["tiers"]["selection"])
            self.assertEqual(set(counts.values()), {per}, axis)

    def test_test_pool_only(self):
        plan = the_plan()
        for layout in plan["layouts"].values():
            for entry in layout["axes"].values():
                c = Condition.from_dict(entry["condition"])
                self.assertIn(c.pool, ("none", "test"))
                self.assertNotEqual(c.pool, "train")

    def test_a_layout_dict_is_not_changed_by_its_conditions(self):
        plan = the_plan()
        for layout in list(plan["layouts"].values())[:20]:
            self.assertEqual(len(layout["distractors"]), layout["distractor_count"])
            self.assertTrue(p2.START_X_M[0] <= layout["start"][0] <= p2.START_X_M[1])


class Determinism(unittest.TestCase):
    def test_same_plan_twice(self):
        self.assertEqual(p2.digest(plan_module.make_plan()), p2.digest(the_plan()))

    def test_write_refuses_an_existing_directory_and_records_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan"
            digest = plan_module.write(out)
            self.assertEqual((out / "plan.sha256").read_text().strip(), digest)
            self.assertEqual(p2.digest(json.loads((out / "plan.json").read_text())), digest)
            with self.assertRaises(FileExistsError):
                plan_module.write(out)


if __name__ == "__main__":
    unittest.main()
