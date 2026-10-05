import ast
import time
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.b3_robosuite_data import plan as plan_module
from experiments.b3_robosuite_data.plan import (
    CELLS, COLOURS, HELD_OUT_PAIRS, MIN_CLEARANCE_M, NEAR_MAX_M, FAR_CLEARANCE_M, TRAINED_PAIRS,
    attempt_draw, make_plan, resample_layout, validate_plan,
)

TOL = 1e-9
PLAN_PY = Path(plan_module.__file__)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def in_region(xy, side):
    return (0.20 - TOL <= xy[0] <= 0.35 + TOL and 0.20 - TOL <= abs(xy[1]) <= 0.45 + TOL
            and (xy[1] < 0) == (side == "A"))


def in_band(xy, side, edge):
    (x_low, x_high), (y_low, y_high) = plan_module.EDGE_BANDS[edge]
    return x_low - TOL <= xy[0] <= x_high + TOL and y_low - TOL <= abs(xy[1]) <= y_high + TOL \
        and (xy[1] < 0) == (side == "A")


def check_geometry(test, e, distractors, placement, edge, label):
    """Shared by training episodes and reserved layouts."""
    pick, place, side = e["pick"], e["place"], e["side"]
    other = "B" if side == "A" else "A"
    xy = e["cube_xy_m"]
    test.assertTrue(in_region(xy[pick], side), label)
    test.assertTrue(in_region(xy[place], other), label)
    if edge != "interior":
        test.assertTrue(in_band(xy[pick], side, edge), label)
    for n, colour in enumerate(distractors):
        d = xy[colour]
        test.assertTrue(0.20 - TOL <= d[0] <= 0.35 + TOL and abs(d[1]) <= 0.45 + TOL, label)
        near = n == 0 and placement in ("near_pick", "near_place")
        d_pick, d_place = dist(d, xy[pick]), dist(d, xy[place])
        if near:
            anchor, far = (d_pick, d_place) if placement == "near_pick" else (d_place, d_pick)
            test.assertTrue(MIN_CLEARANCE_M - TOL <= anchor <= NEAR_MAX_M + TOL, f"{label} {anchor}")
            test.assertGreaterEqual(far, MIN_CLEARANCE_M - TOL, label)
        else:
            test.assertGreaterEqual(min(d_pick, d_place), FAR_CLEARANCE_M - TOL, label)
    cubes = list(xy.values())
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            test.assertGreaterEqual(dist(cubes[i], cubes[j]), MIN_CLEARANCE_M - TOL, label)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = make_plan()
        cls.train = [e for e in cls.plan["episodes"] if e["split"] == "train"]
        cls.validation = [e for e in cls.plan["episodes"] if e["split"] == "validation"]

    def test_counts(self):
        self.assertEqual(len(self.train), 500)
        self.assertEqual(len(self.validation), 60)
        self.assertEqual(len(self.plan["episodes"]), 560)
        self.assertEqual([e["episode_index"] for e in self.plan["episodes"]], list(range(560)))
        self.assertEqual(len(CELLS), 20)
        self.assertEqual(Counter((e["pick"], e["place"], e["side"]) for e in self.train), {c: 25 for c in CELLS})
        self.assertEqual(Counter((e["pick"], e["place"], e["side"]) for e in self.validation), {c: 3 for c in CELLS})
        self.assertEqual(Counter(len(e["distractors"]) for e in self.train), {0: 100, 1: 200, 2: 200})
        self.assertEqual(Counter(len(e["distractors"]) for e in self.validation), {0: 20, 1: 20, 2: 20})
        self.assertEqual(Counter(e["edge"] for e in self.train), {"interior": 400, "inner": 50, "outer": 50})
        self.assertEqual(Counter(e["edge"] for e in self.validation), {"interior": 60})

    def test_per_cell_profile_and_edge_balance(self):
        for cell in CELLS:
            rows = [e for e in self.train if (e["pick"], e["place"], e["side"]) == cell]
            profiles = Counter(tuple(e["profile"]) for e in rows)
            self.assertEqual(sum(profiles.values()), 25)
            self.assertEqual(profiles[(0, "none")], 5)
            self.assertEqual(profiles[(1, "far")], 5)
            self.assertEqual(profiles[(2, "far")], 5)
            self.assertEqual(sum(v for (c, p), v in profiles.items() if c == 1 and p != "far"), 5)
            self.assertEqual(sum(v for (c, p), v in profiles.items() if c == 2 and p != "far"), 5)
            self.assertEqual(Counter(e["edge"] for e in rows)["interior"], 20)
            self.assertEqual(sorted(Counter(e["edge"] for e in rows if e["edge"] != "interior").values()), [2, 3])
            one = Counter(e["distractors"][0] for e in rows if len(e["distractors"]) == 1)
            self.assertEqual(sorted(one.values()), [5, 5])

    def test_held_out_pairs_never_collected(self):
        for e in self.plan["episodes"]:
            self.assertNotIn((e["pick"], e["place"]), HELD_OUT_PAIRS)
        self.assertEqual(len(TRAINED_PAIRS), 10)

    def test_colour_does_not_predict_region(self):
        for role in ("pick", "place"):
            for colour in COLOURS:
                sides = Counter(e["side"] for e in self.train if e[role] == colour)
                self.assertTrue(sides["A"] > 0)
                self.assertEqual(sides["A"], sides["B"], f"{role} {colour}")

    def test_every_colour_trained_in_both_roles(self):
        self.assertEqual({e["pick"] for e in self.train}, set(COLOURS))
        self.assertEqual({e["place"] for e in self.train}, set(COLOURS))
        picks = Counter(p for p, _ in TRAINED_PAIRS)
        places = Counter(q for _, q in TRAINED_PAIRS)
        self.assertEqual((picks["red"], picks["blue"], picks["green"], picks["yellow"]), (3, 3, 2, 2))
        self.assertEqual((places["red"], places["blue"], places["green"], places["yellow"]), (2, 2, 3, 3))

    def test_geometry(self):
        for e in self.plan["episodes"]:
            self.assertEqual(sorted(e["cube_xy_m"]), e["colours"])
            self.assertEqual(sorted({e["pick"], e["place"], *e["distractors"]}), e["colours"])
            check_geometry(self, e, e["distractors"], e["profile"][1], e["edge"], e["layout_id"])
        for layout_id, h in self.plan["evaluation_holdout"].items():
            for name, xy in h["conditions"].items():
                scene = {"pick": h["pick"], "place": h["place"], "side": h["side"],
                         "cube_xy_m": {**h["cube_xy_m"], **xy}}
                order = h["distractors"] if name == "distractor" else []
                check_geometry(self, scene, order, h["near"], h["edge"], f"{layout_id}:{name}")
                self.assertEqual(set(xy), set(order))

    def test_near_profiles_are_split_between_pick_and_place(self):
        near = Counter(e["profile"][1] for e in self.train if e["profile"][1].startswith("near"))
        self.assertEqual(near, {"near_pick": 100, "near_place": 100})

    def test_validation_profiles_are_one_each_per_cell(self):
        for cell in CELLS:
            rows = [e for e in self.validation if (e["pick"], e["place"], e["side"]) == cell]
            self.assertEqual(sorted(e["profile"] for e in rows), [[0, "none"], [1, "far"], [2, "near_pick"]])

    def test_reserved_layouts(self):
        hold = self.plan["evaluation_holdout"]
        self.assertEqual(list(hold), [f"evaluation_{n:03d}" for n in range(60)])
        self.assertEqual((len(self.plan["evaluation_screen_ids"]), len(self.plan["evaluation_report_ids"])), (20, 40))
        screen = [hold[i] for i in self.plan["evaluation_screen_ids"]]
        report = [hold[i] for i in self.plan["evaluation_report_ids"]]
        self.assertEqual(Counter((h["pick"], h["place"], h["side"]) for h in screen), {c: 1 for c in CELLS})
        self.assertEqual(sum((h["pick"], h["place"]) in HELD_OUT_PAIRS for h in report), 20)
        self.assertEqual(sum((h["pick"], h["place"]) in HELD_OUT_PAIRS for h in screen), 0)
        for pair in HELD_OUT_PAIRS:
            for side in ("A", "B"):
                self.assertEqual(sum((h["pick"], h["place"], h["side"]) == (*pair, side) for h in report), 5)
        self.assertEqual(sum(h["edge"] != "interior" for h in screen), 5)
        self.assertEqual(sum(h["edge"] != "interior" for h in report), 10)
        self.assertEqual({tuple(sorted(h["conditions"])) for h in screen}, {("distractor",)})
        self.assertEqual({tuple(sorted(h["conditions"])) for h in report}, {("control", "distractor")})
        self.assertEqual(Counter(h["near"] for h in hold.values()), {"near_pick": 30, "near_place": 30})
        used = {tuple(e["cube_xy_m"][e["pick"]]) for e in self.plan["episodes"]}
        self.assertFalse(used & {tuple(h["cube_xy_m"][h["pick"]]) for h in hold.values()})
        self.assertEqual(len({tuple(h["cube_xy_m"][h["pick"]]) for h in hold.values()}), 60)

    def test_deterministic_and_canonical(self):
        self.assertEqual(make_plan(), make_plan())
        validate_plan(make_plan())
        bad = make_plan()
        bad["episodes"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            validate_plan(bad)
        bad = make_plan()
        bad["min_clearance_m"] = 0.05
        with self.assertRaises(ValueError):
            validate_plan(bad)

    def test_riskiest_profiles_come_first(self):
        e = self.plan["episodes"]
        self.assertEqual((e[0]["profile"], e[0]["edge"]), ([2, "near_pick"], "outer"))
        self.assertEqual(e[1]["profile"], [2, "near_place"])
        self.assertEqual((e[2]["profile"], e[2]["edge"]), ([1, "near_pick"], "inner"))
        self.assertEqual(e[3]["profile"], [0, "none"])

    def test_two_distractor_near_profiles_put_the_near_cube_first(self):
        for e in self.train:
            if e["profile"][0] == 2 and e["profile"][1].startswith("near"):
                target = e["pick"] if e["profile"][1] == "near_pick" else e["place"]
                near, far = e["distractors"]
                self.assertLessEqual(dist(e["cube_xy_m"][near], e["cube_xy_m"][target]), NEAR_MAX_M + TOL)
                self.assertGreaterEqual(dist(e["cube_xy_m"][far], e["cube_xy_m"][target]), FAR_CLEARANCE_M - TOL)

    def test_episode_fields(self):
        instructions = self.plan["instructions"]
        self.assertEqual(instructions, sorted(instructions))
        self.assertEqual(len(instructions), 10)
        for e in self.plan["episodes"]:
            self.assertEqual(e["instruction"], instructions[e["task_index"]])
            self.assertEqual(len(e["home_pose_jitter_deg"]), 7)
            self.assertTrue(all(type(v) is float and abs(v) <= 1.0 for v in e["home_pose_jitter_deg"]))
            self.assertEqual(set(e["hover_z_offset"]), {"pre_grasp", "pre_place"})
            self.assertTrue(all(type(v) is float and abs(v) <= 0.03 for v in e["hover_z_offset"].values()))

    def test_seeds_are_unique(self):
        seeds = self.plan["rng_seeds"]
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertEqual(len({e["seed"] for e in self.plan["episodes"]}), 560)

    def test_attempt_draw(self):
        e = self.plan["episodes"][7]
        self.assertEqual(attempt_draw(e, 0), {"seed": e["seed"], "home_pose_jitter_deg": e["home_pose_jitter_deg"]})
        first, second = attempt_draw(e, 1), attempt_draw(e, 2)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first["seed"], e["seed"])
        self.assertEqual(first, attempt_draw(e, 1))
        for draw in (first, second):
            self.assertEqual(len(draw["home_pose_jitter_deg"]), 7)
            self.assertTrue(all(type(v) is float and abs(v) <= 1.0 for v in draw["home_pose_jitter_deg"]))

    def test_resample_layout(self):
        for index in (0, 1, 2, 3, 40, 300, 520):
            e = self.plan["episodes"][index]
            self.assertIs(resample_layout(e, 0), e)
            seen = {tuple(map(tuple, e["cube_xy_m"].values()))}
            for r in (1, 2, 3):
                s = resample_layout(e, r)
                for key in ("cell", "profile", "edge", "colours", "instruction", "pick", "place", "side",
                            "distractors", "seed", "task_index", "split", "episode_index"):
                    self.assertEqual(s[key], e[key], key)
                self.assertEqual(s["layout_id"], f"{e['layout_id']}_r{r}")
                check_geometry(self, s, s["distractors"], s["profile"][1], s["edge"], s["layout_id"])
                self.assertEqual(s, resample_layout(e, r))
                seen.add(tuple(map(tuple, s["cube_xy_m"].values())))
            self.assertEqual(len(seen), 4)

    def test_plan_is_json_canonical(self):
        import json
        round_trip = json.loads(json.dumps(self.plan))
        validate_plan(round_trip)

    def test_runs_quickly(self):
        started = time.monotonic()
        make_plan()
        self.assertLess(time.monotonic() - started, 10)


class PurityTests(unittest.TestCase):
    def test_no_simulator_imports(self):
        banned = {"robosuite", "mujoco", "torch", "lerobot"}
        for name in ("plan.py", "contract.py"):
            path = PLAN_PY.with_name(name)
            if not path.exists():
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                for module in modules:
                    self.assertNotIn(module.split(".")[0], banned, f"{name} imports {module}")


if __name__ == "__main__":
    unittest.main()
