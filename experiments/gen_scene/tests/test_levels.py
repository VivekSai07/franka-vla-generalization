import math
import random
import unittest
from collections import Counter

from experiments.gen_scene import levels, pools
from experiments.gen_scene.conditions import Condition, resolve
from helpers import layouts, make_layout
from experiments.props_p2_data import plan as p2


class PinnedNumbers(unittest.TestCase):
    """Every number the spec fixes, literal and independent of levels.py's own names."""

    def test_levels_and_offsets(self):
        self.assertEqual(levels.AXES, ("lighting", "background", "distractors", "spatial", "disturbance"))
        self.assertEqual(levels.LEVELS, {"lighting": ("dark", "bright", "tinted", "low_angle"), "background": ("table", "room", "both"),
                                         "distractors": ("three", "four", "five"),
                                         "spatial": ("table_up5", "table_down5", "table_up10", "outer_start"),
                                         "disturbance": ("move", "drop")})
        self.assertEqual(levels.OFFSETS, {"lighting": 0, "background": 1, "distractors": 2, "spatial": 1, "disturbance": 1})

    def test_lighting(self):
        dark, bright = levels.lighting("dark", 0), levels.lighting("bright", 0)
        self.assertEqual((dark.light_diffuse_scale, dark.ambient_scale, dark.light_specular_scale, dark.tint), (0.3, 0.3, 1.0, (1.0, 1.0, 1.0)))
        self.assertEqual((bright.light_diffuse_scale, bright.light_specular_scale, bright.ambient_scale), (2.5, 2.5, 1.0))
        self.assertEqual(levels.lighting("tinted", 0).tint, (1.0, 0.75, 0.5))
        self.assertEqual(levels.lighting("tinted", 1).tint, (0.6, 0.75, 1.0))
        self.assertEqual(levels.lighting("tinted", 2).tint, (1.0, 0.75, 0.5))
        low = levels.lighting("low_angle", 0)
        self.assertTrue(low.light_shadows)
        self.assertEqual(low.headlight_scale, 0.3)
        self.assertAlmostEqual(low.light_dir[2], -math.sin(math.radians(20)))
        self.assertAlmostEqual(low.light_dir[1], -math.cos(math.radians(20)))
        self.assertEqual(low.light_dir[0], 0.0)
        self.assertAlmostEqual(sum(v * v for v in low.light_dir), 1.0)
        for level in levels.LEVELS["lighting"]:
            self.assertEqual((levels.lighting(level, 0).pool, levels.lighting(level, 0).table_offset_m), ("none", 0.0))

    def test_spatial_table_levels(self):
        for level, offset in (("table_up5", 0.05), ("table_down5", -0.05), ("table_up10", 0.10)):
            c = levels.spatial(level, make_layout(0), "x", 0)
            self.assertEqual((c.table_offset_m, c.bin_shift_m, c.start), (offset, 0.0, None), level)

    def test_regions_and_distances(self):
        self.assertEqual((levels.OUTER_START_X_M, levels.OUTER_START_Y_M, levels.OUTER_BIN_SHIFT_M), ((0.42, 0.50), (-0.12, 0.12), 0.04))
        self.assertEqual((levels.DISTRACTOR_REGION_X_M, levels.DISTRACTOR_REGION_Y_M), ((0.25, 0.52), (-0.20, 0.20)))
        self.assertEqual((levels.MOVE_DISTANCE_M, levels.MOVE_TRIGGER_M), (0.08, 0.06))
        self.assertEqual((levels.MOVE_REGION_X_M, levels.MOVE_REGION_Y_M), ((0.25, 0.46), (-0.16, 0.16)))
        self.assertEqual((levels.DROP_LIFT_M, levels.DROP_OPEN_TICKS, levels.LIFT_SUCCESS_M), (0.05, 10, 0.05))
        self.assertEqual((levels.DISTRACTOR_PROPS, levels.PRIMITIVES), (3, {"three": 0, "four": 1, "five": 2}))
        self.assertEqual((levels.REGION_RADIUS_MIN_M, levels.BIN_CLEARANCE_M), (0.285, 0.02))

    def test_tick_caps(self):
        self.assertEqual((levels.NOMINAL_TICKS, levels.DISTURBANCE_TICKS), (900, 1200))
        for level in ("move", "drop"):
            c = levels.disturbance(level, make_layout(0), "x")
            self.assertEqual(levels.ticks_for(c), 1200)
        for c in (levels.lighting("dark", 0), levels.spatial("table_up5", make_layout(0), "x", 0)):
            self.assertEqual(levels.ticks_for(c), 900)
        self.assertEqual(levels.ticks_for(Condition()), 900)


class Rotation(unittest.TestCase):
    def counts(self, n):
        return {axis: Counter(levels.level_of(axis, i) for i in range(n)) for axis in levels.AXES}

    def test_benchmark_tier_level_sizes(self):
        got = self.counts(144)
        for axis, per in (("lighting", 36), ("background", 48), ("distractors", 48), ("spatial", 36), ("disturbance", 72)):
            self.assertEqual(set(got[axis].values()), {per}, axis)
            self.assertEqual(set(got[axis]), set(levels.LEVELS[axis]))

    def test_selection_tier_level_sizes(self):
        got = self.counts(24)
        for axis, per in (("lighting", 6), ("background", 8), ("distractors", 8), ("spatial", 6), ("disturbance", 12)):
            self.assertEqual(set(got[axis].values()), {per}, axis)

    def test_each_cell_of_three_meets_every_three_level_axis_once(self):
        for axis in ("background", "distractors"):
            for cell in range(48):
                self.assertEqual(sorted(levels.level_of(axis, 3 * cell + i) for i in range(3)), sorted(levels.LEVELS[axis]))

    def test_first_layouts_pin_the_offsets(self):
        self.assertEqual([levels.level_of("lighting", n) for n in range(5)], ["dark", "bright", "tinted", "low_angle", "dark"])
        self.assertEqual([levels.level_of("background", n) for n in range(3)], ["room", "both", "table"])
        self.assertEqual([levels.level_of("distractors", n) for n in range(3)], ["five", "three", "four"])
        self.assertEqual([levels.level_of("spatial", n) for n in range(4)], ["table_down5", "table_up10", "outer_start", "table_up5"])
        self.assertEqual([levels.level_of("disturbance", n) for n in range(3)], ["drop", "move", "drop"])


class Background(unittest.TestCase):
    def test_test_pool_only_and_even_use(self):
        both = pools.pools()
        seen_table, seen_floor, seen_wall = Counter(), Counter(), Counter()
        for j in range(48):
            c = levels.background("both", j)
            self.assertIn(c.table_texture, both["table"]["test"])
            self.assertIn(c.floor_texture, both["room"]["test"])
            self.assertIn(c.wall_texture, both["room"]["test"])
            self.assertNotEqual(c.floor_texture, c.wall_texture)
            seen_table[c.table_texture] += 1
            seen_floor[c.floor_texture] += 1
            seen_wall[c.wall_texture] += 1
        for seen in (seen_table, seen_floor, seen_wall):
            self.assertEqual((len(seen), set(seen.values())), (12, {4}))

    def test_levels_touch_only_their_surfaces(self):
        t, r, b = levels.background("table", 0), levels.background("room", 0), levels.background("both", 0)
        self.assertEqual((t.floor_texture, t.wall_texture), (None, None))
        self.assertIsNone(r.table_texture)
        self.assertTrue(r.floor_texture and r.wall_texture)
        self.assertTrue(b.table_texture and b.floor_texture and b.wall_texture)
        self.assertTrue(all(c.pool == "test" for c in (t, r, b)))


class Geometry(unittest.TestCase):
    def test_hand_clear_equals_p2s_for_props(self):
        rng = random.Random(1)
        for _ in range(400):
            a, b = rng.sample(list(p2.OBJECTS), 2)
            pa = (rng.uniform(0.27, 0.4), rng.uniform(-0.12, 0.12), rng.uniform(-90, 90))
            pb = (pa[0] + rng.uniform(-0.2, 0.2), pa[1] + rng.uniform(-0.2, 0.2), rng.uniform(-90, 90))
            self.assertEqual(levels.hand_clear(a, pa, b, pb), p2.hand_clear(a, pa, b, pb))
            self.assertEqual(levels.min_distance(a, b), p2.min_distance(a, b))

    def test_a_tall_primitive_in_the_hand_strip_blocks_and_a_flat_one_does_not(self):
        self.assertFalse(levels.hand_clear("can", (0.33, 0.0, 0.0), "tube_pink", (0.33, 0.12, 0.0)))
        self.assertTrue(levels.hand_clear("can", (0.33, 0.0, 0.0), "disc_pink", (0.33, 0.12, 0.0)))
        self.assertFalse(levels.spaced([("can", (0.33, 0.0, 0.0)), ("tube_pink", (0.33, 0.12, 0.0))]))

    def test_bins_clear(self):
        self.assertTrue(levels.bins_clear("can", (0.33, 0.0, 0.0)))
        self.assertFalse(levels.bins_clear("can", (0.33, 0.17, 0.0)))     # against the left bin's inner wall (edge at y = 0.21)
        self.assertTrue(levels.bins_clear("can", (0.33, 0.17, 0.0), bin_shift_m=0.04))
        self.assertFalse(levels.bins_clear("can", (0.33, 0.13, 0.0), bin_shift_m=-0.04))

    def test_primitive_extents(self):
        self.assertAlmostEqual(levels.extent("ball_pink")[0], 0.030)
        self.assertAlmostEqual(levels.extent("ball_pink")[1], 0.060)
        self.assertAlmostEqual(levels.extent("pill_lime")[0], 0.050)     # lying capsule: half-length plus radius
        self.assertAlmostEqual(levels.extent("pill_lime")[1], 0.040)
        self.assertEqual(levels.extent("can"), (p2.FOOTPRINT_RADIUS_M["can"], p2.HEIGHT_M["can"]))


class Distractors(unittest.TestCase):
    def setUp(self):
        self.layouts = layouts(12)

    def test_three_four_five(self):
        for layout in self.layouts:
            base = [d["object"] for d in layout["distractors"]]
            for level, n_prim in (("three", 0), ("four", 1), ("five", 2)):
                c = levels.distractors(level, layout, layout["identity"])
                c.check()
                added = [p[0] for p in c.extra_props]
                self.assertEqual(len(base) + len(added), 3, (level, base, added))
                self.assertEqual(len(c.primitives), n_prim)
                self.assertFalse(({layout["object"], *base} & set(added)) or len(set(added)) != len(added))
                self.assertTrue(all(s in pools.PRIMITIVE_SHAPES["B"] and col in pools.PRIMITIVE_COLOURS["B"] for s, col, *_ in c.primitives))
                resolved = resolve(layout, c)
                items = [(n, resolved["poses"][n]) for n in resolved["names"]]
                self.assertTrue(levels.spaced(items))
                self.assertTrue(all(0.25 <= x <= 0.52 and -0.20 <= y <= 0.20 for _, x, y, _ in c.extra_props))

    def test_deterministic_and_identity_dependent(self):
        a = levels.distractors("five", self.layouts[0], "id-a")
        self.assertEqual(a, levels.distractors("five", self.layouts[0], "id-a"))
        self.assertNotEqual(a, levels.distractors("five", self.layouts[0], "id-b"))


class OuterStart(unittest.TestCase):
    def test_start_is_outside_p2s_region_and_spaced(self):
        for index, layout in enumerate(layouts(12)):
            c = levels.spatial("outer_start", layout, layout["identity"], index)
            x, y, yaw = c.start
            self.assertTrue(0.42 <= x <= 0.50 and -0.12 <= y <= 0.12, c.start)
            self.assertGreater(x, p2.START_X_M[1])
            self.assertLessEqual(abs(yaw), p2.YAW_MAX_DEG_BY_OBJECT[layout["object"]])
            self.assertIn(abs(c.bin_shift_m), (0.04,))
            others = [(d["object"], tuple(d["pose"])) for d in layout["distractors"]]
            self.assertTrue(levels.spaced([(layout["object"], c.start), *others], c.bin_shift_m))

    def test_bin_shift_sign_follows_parity_unless_it_would_touch(self):
        layout = layouts(1)[0]
        even = levels.spatial("outer_start", layout, layout["identity"], 0)
        odd = levels.spatial("outer_start", layout, layout["identity"], 1)
        self.assertEqual(even.bin_shift_m, 0.04)
        self.assertIn(odd.bin_shift_m, (-0.04, 0.04))
        signs = {levels.spatial("outer_start", l, l["identity"], 1).bin_shift_m for l in layouts(24)}
        self.assertEqual(signs, {-0.04, 0.04})   # together is used when it is clear, apart otherwise


class Move(unittest.TestCase):
    def test_destination_is_eight_cm_away_free_and_in_the_region(self):
        for layout in layouts(24):
            c = levels.disturbance("move", layout, layout["identity"])
            c.check()
            x, y = c.move_to
            self.assertAlmostEqual(math.hypot(x - layout["start"][0], y - layout["start"][1]), 0.08, places=12)
            self.assertTrue(0.25 <= x <= 0.46 and -0.16 <= y <= 0.16 and math.hypot(x, y) >= 0.285)
            others = [(d["object"], tuple(d["pose"])) for d in layout["distractors"]]
            self.assertTrue(levels.physical([(layout["object"], (x, y, layout["start"][2])), *others]))
            self.assertEqual(c, levels.disturbance("move", layout, layout["identity"]))

    def test_full_spacing_is_preferred_when_a_spot_has_it(self):
        strict = [levels.disturbance("move", l, l["identity"]) for l in layouts(24)]
        self.assertGreaterEqual(sum(levels.spaced([(l["object"], (*c.move_to, l["start"][2])),
                                                   *((d["object"], tuple(d["pose"])) for d in l["distractors"])])
                                    for l, c in zip(layouts(24), strict)), 12)

    def test_the_rules_run_from_strict_to_lenient(self):
        self.assertEqual(levels.MOVE_RULES, (levels.spaced, levels.free, levels.physical))
        self.assertEqual(levels.MOVE_GAP_M, 0.01)

    def test_a_layout_with_no_spaced_or_free_spot_falls_back_to_non_overlap(self):
        from experiments.gen_scene.tools import expert_probe
        layout = next(l for l in expert_probe.layouts() if l["layout_id"] == "gen_probe_can-blue-left-d2_0")
        c = levels.disturbance("move", layout, layout["identity"])
        items = [(layout["object"], (*c.move_to, layout["start"][2])), *((d["object"], tuple(d["pose"])) for d in layout["distractors"])]
        self.assertFalse(levels.free(items))          # no spot 8 cm away keeps P2's distance from both neighbours ...
        self.assertTrue(levels.physical(items))       # ... but one exists that overlaps nothing
        self.assertAlmostEqual(math.hypot(c.move_to[0] - layout["start"][0], c.move_to[1] - layout["start"][1]), 0.08, places=12)

    def test_physical_rule(self):
        self.assertTrue(levels.physical([("can", (0.30, 0.0, 0.0)), ("bread", (0.30 + 0.036 + 0.032 + 0.0101, 0.0, 0.0))]))
        self.assertFalse(levels.physical([("can", (0.30, 0.0, 0.0)), ("bread", (0.30 + 0.036 + 0.032 + 0.0099, 0.0, 0.0))]))
        self.assertFalse(levels.physical([("can", (0.33, 0.17, 0.0))]))


class Assign(unittest.TestCase):
    def test_every_layout_gets_one_valid_condition_per_axis(self):
        ls = layouts(24)
        out = levels.assign(ls)
        self.assertEqual(list(out), [l["layout_id"] for l in ls])
        per_level = Counter()
        for n, layout in enumerate(ls):
            conditions = out[layout["layout_id"]]
            self.assertEqual(list(conditions), list(levels.AXES))
            for axis, c in conditions.items():
                self.assertEqual((c.axis, c.level), (axis, levels.level_of(axis, n)))
                per_level[(axis, c.level)] += 1
        self.assertEqual({k: v for k, v in per_level.items() if k[0] == "lighting"}, {("lighting", l): 6 for l in levels.LEVELS["lighting"]})
        self.assertEqual({k: v for k, v in per_level.items() if k[0] == "disturbance"}, {("disturbance", "move"): 12, ("disturbance", "drop"): 12})

    def test_tinted_alternates_within_the_level(self):
        out = levels.assign(layouts(24))
        tints = [out[l["layout_id"]]["lighting"].tint for n, l in enumerate(layouts(24)) if levels.level_of("lighting", n) == "tinted"]
        self.assertEqual(tints, [levels.WARM, levels.COOL] * 3)

    def test_deterministic(self):
        self.assertEqual({k: {a: c.digest() for a, c in v.items()} for k, v in levels.assign(layouts(12)).items()},
                         {k: {a: c.digest() for a, c in v.items()} for k, v in levels.assign(layouts(12)).items()})


if __name__ == "__main__":
    unittest.main()
