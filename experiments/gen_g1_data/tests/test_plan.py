import ast
import copy
import math
import unittest
from collections import Counter
from pathlib import Path

from experiments.gen_g1_data import plan as p
from experiments.gen_scene import levels, pools
from experiments.props_scene import constants as psc

PLAN = p.make_plan()
EPISODES = PLAN["episodes"]
PACKAGE = Path(p.__file__).parent
TRAIN_POOLS = pools.pools()


def _items(e):
    """(name, pose) of the target, the prop distractors and the primitives of one slot."""
    return ([(e["object"], tuple(e["start"]))] + [(d["object"], tuple(d["pose"])) for d in e["distractors"]]
            + [(f"{s}_{c}", (x, y, yaw)) for s, c, x, y, yaw in p.condition_of(e).primitives])


def _split(split):
    return [e for e in EPISODES if e["split"] == split]


class CountsTests(unittest.TestCase):
    def test_sizes_and_indices(self):
        self.assertEqual((p.TRAIN_COUNT, p.VALIDATION_COUNT, p.ADMIT_COUNT, p.GATE_ATTEMPTS), (2700, 300, 3000, 96))
        self.assertEqual([e["episode_index"] for e in EPISODES], list(range(3000)))
        self.assertEqual(PLAN["train_ids"], list(range(2700)))
        self.assertEqual(PLAN["validation_ids"], list(range(2700, 3000)))
        self.assertEqual(Counter(e["split"] for e in EPISODES), {"train": 2700, "validation": 300})
        self.assertEqual([e["split"] for e in EPISODES[:2700]], ["train"] * 2700)

    def test_every_trained_cell_has_135_training_and_15_validation_episodes(self):
        self.assertEqual(Counter(tuple(e["cell"]) for e in _split("train")), {c: 135 for c in p.TRAINED_CELLS})
        self.assertEqual(Counter(tuple(e["cell"]) for e in _split("validation")), {c: 15 for c in p.TRAINED_CELLS})
        self.assertEqual(len(p.TRAINED_CELLS), 20)

    def test_held_out_pairs_are_never_collected(self):
        held = {tuple(x) for x in p.HELD_OUT_PAIRS}
        self.assertEqual(held, {("milk", "blue"), ("bread", "orange")})
        self.assertFalse(any((e["object"], e["bin"]) in held for e in EPISODES + PLAN["gate_attempts"]))

    def test_colour_does_not_predict_slot(self):
        for colour in ("blue", "orange"):
            by_slot = Counter(e["blue_slot"] for e in _split("train") if e["bin"] == colour)
            self.assertEqual(by_slot["left"], by_slot["right"])

    def test_distractor_counts_follow_the_revision_exactly(self):
        self.assertEqual(p.count_deck(3000), {0: 150, 1: 450, 2: 750, 3: 750, 4: 600, 5: 300})
        self.assertEqual(Counter(e["distractor_count"] for e in _split("train")), {0: 135, 1: 405, 2: 675, 3: 675, 4: 540, 5: 270})
        self.assertEqual(Counter(e["distractor_count"] for e in _split("validation")), {0: 15, 1: 45, 2: 75, 3: 75, 4: 60, 5: 30})
        with self.assertRaises(ValueError):
            p.count_deck(1001)

    def test_counts_are_what_the_slot_contains(self):
        for e in EPISODES:
            self.assertEqual(e["distractor_count"], len(e["distractors"]) + len(p.condition_of(e).primitives))
            self.assertEqual((e["prop_distractors"], e["primitive_count"]), (len(e["distractors"]), len(p.condition_of(e).primitives)))

    def test_strata_are_balanced_by_construction(self):
        for split, n in (("train", 2700), ("validation", 300)):
            rows = _split(split)
            self.assertEqual(Counter(e["quintile"] for e in rows), {q: n // 5 for q in range(5)})
            self.assertEqual(Counter(e["region"] for e in rows), {"outer": round(n * 0.3), "p2": n - round(n * 0.3)})

    def test_quintile_label_matches_the_table_height(self):
        for e in EPISODES + PLAN["gate_attempts"]:
            self.assertEqual(p.quintile_of(p.condition_of(e).table_offset_m), e["quintile"])
        self.assertEqual([p.quintile_of(v) for v in (-0.07, 0.0, 0.0309, 0.12)], [0, 1, 2, 4])

    def test_instruction_is_the_slots_object_and_bin(self):
        for e in EPISODES:
            self.assertEqual(e["instruction"], f"put the {e['object']} in the {e['bin']} bin")

    def test_condition_digest_is_stored_and_matches(self):
        for e in EPISODES:
            self.assertEqual(p.condition_of(e).digest(), e["condition_sha256"])


class RangeTests(unittest.TestCase):
    def test_every_continuous_value_is_inside_its_range(self):
        for e in EPISODES + PLAN["gate_attempts"]:
            c = p.condition_of(e)
            for value, (low, high) in ((c.light_diffuse_scale, p.DIFFUSE_SCALE), (c.light_specular_scale, p.SPECULAR_SCALE),
                                       (c.ambient_scale, p.AMBIENT_SCALE), (c.headlight_scale, p.HEADLIGHT_SCALE),
                                       (c.table_offset_m, p.TABLE_OFFSET_M), (c.bin_shift_m, p.BIN_SHIFT_M),
                                       *((t, p.TINT) for t in c.tint)):
                self.assertTrue(low <= value <= high, (e["layout_id"], value, low, high))
            self.assertEqual(len(c.tint), 3)

    def test_the_ranges_are_the_revisions(self):
        self.assertEqual((p.DIFFUSE_SCALE, p.SPECULAR_SCALE, p.AMBIENT_SCALE, p.HEADLIGHT_SCALE),
                         ((0.25, 2.8), (0.5, 2.8), (0.2, 1.5), (0.3, 1.5)))
        self.assertEqual((p.TINT, p.ELEVATION_DEG, p.TABLE_OFFSET_M, p.BIN_SHIFT_M, p.HOME_JITTER_DEG_MAX),
                         ((0.55, 1.0), (15.0, 80.0), (-0.07, 0.12), (-0.05, 0.05), 10.0))
        self.assertEqual((p.TEXTURE_SHARE, p.LIGHT_DIR_SHARE, p.SHADOW_SHARE, p.OUTER_SHARE), (0.85, 0.5, 0.5, 0.3))

    def test_the_whole_range_is_used_not_a_corner_of_it(self):
        conditions = [p.condition_of(e) for e in EPISODES]
        for field, (low, high) in (("light_diffuse_scale", p.DIFFUSE_SCALE), ("light_specular_scale", p.SPECULAR_SCALE),
                                   ("ambient_scale", p.AMBIENT_SCALE), ("headlight_scale", p.HEADLIGHT_SCALE),
                                   ("table_offset_m", p.TABLE_OFFSET_M), ("bin_shift_m", p.BIN_SHIFT_M)):
            values = [getattr(c, field) for c in conditions]
            self.assertLess(min(values), low + 0.05 * (high - low), field)
            self.assertGreater(max(values), high - 0.05 * (high - low), field)

    def test_light_direction_is_a_unit_vector_between_the_elevations(self):
        directed = [p.condition_of(e).light_dir for e in EPISODES if p.condition_of(e).light_dir is not None]
        self.assertTrue(0.45 < len(directed) / len(EPISODES) < 0.55)
        for d in directed:
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in d)), 1.0, places=9)
            elevation = math.degrees(math.asin(-d[2]))
            self.assertTrue(p.ELEVATION_DEG[0] - 1e-9 <= elevation <= p.ELEVATION_DEG[1] + 1e-9)
        azimuths = [math.degrees(math.atan2(d[1], d[0])) % 360 for d in directed]
        self.assertLess(min(azimuths), 20)
        self.assertGreater(max(azimuths), 340)

    def test_shares_of_shadows_and_textures(self):
        conditions = [p.condition_of(e) for e in EPISODES]
        self.assertTrue(0.45 < sum(c.light_shadows for c in conditions) / 3000 < 0.55)
        for field in ("table_texture", "floor_texture", "wall_texture"):
            self.assertTrue(0.80 < sum(getattr(c, field) is not None for c in conditions) / 3000 < 0.90, field)

    def test_home_jitter_is_ten_degrees_not_one(self):
        jitters = [j for e in EPISODES + PLAN["gate_attempts"] for j in e["home_pose_jitter_deg"]]
        self.assertTrue(all(abs(j) <= 10.0 for j in jitters))
        self.assertGreater(max(abs(j) for j in jitters), 9.0)
        self.assertTrue(all(len(e["home_pose_jitter_deg"]) == 7 for e in EPISODES))

    def test_start_is_in_its_region_and_the_outer_ring_is_the_outer_ring(self):
        for e in EPISODES:
            x, y, yaw = e["start"]
            low, high = psc.START_X_M if e["region"] == "p2" else levels.OUTER_START_X_M
            self.assertTrue(low <= x <= high and psc.START_Y_M[0] <= y <= psc.START_Y_M[1], e["layout_id"])
            self.assertGreaterEqual(math.hypot(x, y), psc.START_RADIUS_MIN_M)
            self.assertLessEqual(abs(yaw), psc.YAW_MAX_DEG_BY_OBJECT[e["object"]])
        self.assertGreater(levels.OUTER_START_X_M[0], psc.START_X_M[1])

    def test_condition_start_is_unused_and_the_layout_holds_the_start(self):
        self.assertTrue(all(p.condition_of(e).start is None and not p.condition_of(e).extra_props for e in EPISODES))
        self.assertTrue(all(p.condition_of(e).disturbance == "none" and p.condition_of(e).move_to is None for e in EPISODES))


class SceneTests(unittest.TestCase):
    def test_every_slot_is_spaced_and_clear_of_both_shifted_bins(self):
        for e in EPISODES + PLAN["gate_attempts"]:
            self.assertTrue(levels.spaced(_items(e), p.condition_of(e).bin_shift_m), e["layout_id"])

    def test_distractor_poses_are_in_the_wide_region(self):
        for e in EPISODES:
            for name, (x, y, _) in _items(e)[1:]:
                self.assertTrue(p.DISTRACTOR_REGION_X_M[0] <= x <= p.DISTRACTOR_REGION_X_M[1], (e["layout_id"], name))
                self.assertTrue(p.DISTRACTOR_REGION_Y_M[0] <= y <= p.DISTRACTOR_REGION_Y_M[1], (e["layout_id"], name))
                self.assertGreaterEqual(math.hypot(x, y), psc.START_RADIUS_MIN_M)

    def test_distractors_are_distinct_other_objects(self):
        for e in EPISODES:
            names = [d["object"] for d in e["distractors"]]
            self.assertEqual(len(set(names)), len(names))
            self.assertNotIn(e["object"], names)
            primitives = [(s, c) for s, c, *_ in p.condition_of(e).primitives]
            self.assertEqual(len(set(primitives)), len(primitives))

    def test_props_and_primitives_both_occur_and_every_object_is_a_distractor(self):
        train = _split("train")
        self.assertEqual({d["object"] for e in train for d in e["distractors"]}, set(p.OBJECTS))
        self.assertGreater(sum(e["primitive_count"] > 0 for e in train), 1500)
        self.assertGreater(sum(e["prop_distractors"] > 0 for e in train), 2000)
        self.assertEqual(max(e["distractor_count"] for e in EPISODES), 5)
        self.assertEqual(Counter(e["distractor_count"] for e in EPISODES if e["distractor_count"] == 0), {0: 150})

    def test_a_bin_shift_of_both_signs_and_every_height_stratum_occurs_with_every_count(self):
        shifts = [p.condition_of(e).bin_shift_m for e in EPISODES]
        self.assertTrue(sum(s < 0 for s in shifts) > 1000 and sum(s > 0 for s in shifts) > 1000)
        self.assertEqual({(e["quintile"], e["distractor_count"]) for e in EPISODES}, {(q, n) for q in range(5) for n in range(6)})


class TrainPoolOnlyTests(unittest.TestCase):
    def _conditions(self):
        out = [p.condition_of(e) for e in EPISODES + PLAN["gate_attempts"]]
        for r in (1, 2):
            out += [p.condition_of(p.resample_start(e, r)) for e in EPISODES[::25]]
        return out

    def test_no_test_pool_texture_or_pool_b_primitive_is_ever_drawn(self):
        test_textures = set(TRAIN_POOLS["table"]["test"]) | set(TRAIN_POOLS["room"]["test"])
        train_textures = {"table": set(TRAIN_POOLS["table"]["train"]), "room": set(TRAIN_POOLS["room"]["train"])}
        drawn = set()
        for c in self._conditions():
            self.assertEqual(c.pool, "train")
            c.check()
            for field, surface in (("table_texture", "table"), ("floor_texture", "room"), ("wall_texture", "room")):
                name = getattr(c, field)
                if name is not None:
                    self.assertIn(name, train_textures[surface])
                    drawn.add(name)
            for shape, colour, *_ in c.primitives:
                self.assertIn(shape, pools.PRIMITIVE_SHAPES["A"])
                self.assertIn(colour, pools.PRIMITIVE_COLOURS["A"])
                self.assertNotIn(shape, pools.PRIMITIVE_SHAPES["B"])
                self.assertNotIn(colour, pools.PRIMITIVE_COLOURS["B"])
        self.assertFalse(drawn & test_textures)
        self.assertGreater(len(drawn), 80)   # most of the 96 train textures are used

    def test_every_train_texture_and_primitive_is_used(self):
        used_primitives = {(s, c) for cond in self._conditions() for s, c, *_ in cond.primitives}
        self.assertEqual(len(used_primitives), 36)

    def test_the_package_never_names_a_test_pool(self):
        for path in sorted(PACKAGE.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and node.value in ("test", "B"):
                    self.fail(f"{path.name}:{node.lineno} names the test pool or primitive pool B")
                if isinstance(node, ast.Attribute) and node.attr in ("TEST_EVERY", "TEST_REMAINDER"):
                    self.fail(f"{path.name}:{node.lineno} touches the split rule")


class ReproducibilityTests(unittest.TestCase):
    def test_the_plan_is_byte_deterministic_and_pinned(self):
        self.assertEqual(p.digest(p.make_plan()), p.digest(PLAN))
        self.assertEqual(p.digest(PLAN), p.PLAN_SHA256)
        p.validate_plan(PLAN)

    def test_a_noncanonical_plan_is_refused(self):
        changed = copy.deepcopy(PLAN)
        changed["episodes"][0]["start"][0] += 1e-9
        with self.assertRaises(ValueError):
            p.validate_plan(changed)
        changed = copy.deepcopy(PLAN)
        changed["ranges"]["table_offset_m"] = [-0.07, 0.13]
        with self.assertRaises(ValueError):
            p.validate_plan(changed)

    def test_seeds_are_unique_and_the_namespace_is_g1(self):
        seeds = PLAN["rng_seeds"]
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertTrue(all(e["identity"].startswith("gen:g1:") for e in EPISODES + PLAN["gate_attempts"]))
        self.assertEqual(len({e["seed"] for e in EPISODES + PLAN["gate_attempts"]}), 3096)
        self.assertEqual(p.NAMESPACE, "gen:g1")
        self.assertNotEqual(p.NAMESPACE, "gen:g0")

    def test_the_plan_binds_the_texture_manifest(self):
        self.assertEqual(PLAN["ranges"]["texture_manifest_sha256"], pools.manifest_digest())


class DisjointnessTests(unittest.TestCase):
    """Needs the sealed P2 data mounted at outputs/props_p2_data (M0's forbidden set reads it); it fails, never skips, without."""

    def test_no_slot_or_gate_attempt_shares_a_seed_or_pose_with_an_earlier_plan_or_g0(self):
        forbidden = p.earlier_values()
        self.assertGreater(len(forbidden["seeds"]), 4000)
        p.check_disjoint_from_earlier(PLAN, forbidden)

    def test_the_check_notices_a_shared_pose_a_shared_seed_and_a_repeat(self):
        forbidden = p.earlier_values()
        for mutate in (lambda q: q["episodes"][5].__setitem__("seed", next(iter(forbidden["seeds"]))),
                       lambda q: q["episodes"][5].__setitem__("start", list(next(iter(forbidden["poses"])))),
                       lambda q: q["gate_attempts"][1]["distractors"].__setitem__(0, {"object": "can", "pose": list(next(iter(forbidden["poses"])))}),
                       lambda q: q["episodes"][7].__setitem__("start", list(q["episodes"][6]["start"]))):
            changed = copy.deepcopy(PLAN)
            mutate(changed)
            with self.assertRaises(ValueError):
                p.check_disjoint_from_earlier(changed, forbidden)

    def test_g0s_own_poses_are_in_the_forbidden_set(self):
        from experiments.gen_eval import plan as g0
        layout = next(iter(g0.make_plan()["layouts"].values()))
        self.assertIn(tuple(layout["start"]), p.earlier_values()["poses"])


class SubstitutionTests(unittest.TestCase):
    def test_resample_keeps_the_stratum_and_changes_the_scene(self):
        for e in EPISODES[::97]:
            for r in (1, 2, 3):
                q = p.resample_start(e, r)
                self.assertEqual(q, p.resample_start(e, r))
                for key in ("episode_index", "split", "object", "bin", "blue_slot", "cell", "distractor_count", "quintile",
                            "region", "instruction", "seed"):
                    self.assertEqual(q[key], e[key], key)
                self.assertEqual(q["layout_id"], f"{e['layout_id']}_r{r}")
                self.assertNotEqual(q["start"], e["start"])
                self.assertEqual(p.quintile_of(p.condition_of(q).table_offset_m), e["quintile"])
                self.assertEqual(q["distractor_count"], len(q["distractors"]) + len(p.condition_of(q).primitives))
                self.assertTrue(levels.spaced(_items(q), p.condition_of(q).bin_shift_m))
                self.assertTrue(all(abs(j) <= 10.0 for j in q["home_pose_jitter_deg"]))
        self.assertIs(p.resample_start(EPISODES[0], 0), EPISODES[0])
        for bad in (-1, 1.0, True):
            with self.assertRaises(ValueError):
                p.resample_start(EPISODES[0], bad)

    def test_attempt_draws_are_deterministic_distinct_and_ten_degrees(self):
        e = EPISODES[3]
        self.assertEqual(p.attempt_draw(e, 0), {"seed": e["seed"], "home_pose_jitter_deg": e["home_pose_jitter_deg"]})
        draws = [p.attempt_draw(e, k) for k in range(3)]
        self.assertEqual(len({d["seed"] for d in draws}), 3)
        self.assertEqual(p.attempt_draw(e, 2), draws[2])
        self.assertGreater(max(abs(j) for k in range(1, 40) for j in p.attempt_draw(e, k)["home_pose_jitter_deg"]), 5.0)
        self.assertTrue(all(abs(j) <= 10.0 for k in range(1, 40) for j in p.attempt_draw(e, k)["home_pose_jitter_deg"]))
        with self.assertRaises(ValueError):
            p.attempt_draw(e, -1)

    def test_a_gate_attempt_and_a_slot_with_the_same_index_draw_differently(self):
        gate = PLAN["gate_attempts"][3]
        self.assertNotEqual(p.attempt_draw(gate, 1)["seed"], p.attempt_draw(EPISODES[3], 1)["seed"])


class ShardTests(unittest.TestCase):
    def test_sixteen_shards_partition_the_slots_contiguously(self):
        blocks = [p.shard_slots(k, 16) for k in range(16)]
        self.assertEqual([len(b) for b in blocks], [188] * 8 + [187] * 8)
        self.assertEqual([s for b in blocks for s in b], list(range(3000)))
        for b in blocks:
            self.assertEqual(b, list(range(b[0], b[0] + len(b))))

    def test_other_shard_counts_and_bad_arguments(self):
        self.assertEqual([s for k in range(12) for s in p.shard_slots(k, 12)], list(range(3000)))
        self.assertEqual(p.shard_slots(0, 1), list(range(3000)))
        for k, n in ((16, 16), (-1, 16), (0, 0)):
            with self.assertRaises(ValueError):
                p.shard_slots(k, n)


class GateAttemptTests(unittest.TestCase):
    GATE = PLAN["gate_attempts"]

    def test_the_gate_is_balanced_on_what_it_reports(self):
        self.assertEqual(len(self.GATE), 96)
        self.assertEqual(Counter(g["object"] for g in self.GATE), {o: 16 for o in p.OBJECTS})
        self.assertEqual(Counter(g["distractor_count"] for g in self.GATE), {n: 16 for n in range(6)})
        self.assertEqual(sorted(Counter(g["quintile"] for g in self.GATE).values()), [19, 19, 19, 19, 20])
        self.assertEqual(Counter(g["region"] for g in self.GATE), {"outer": 29, "p2": 67})
        self.assertEqual([g["episode_index"] for g in self.GATE], list(range(96)))
        self.assertTrue(all(g["split"] == "gate" for g in self.GATE))

    def test_gate_cells_are_trained_cells(self):
        self.assertTrue(all(tuple(g["cell"]) in p.TRAINED_CELLS for g in self.GATE))

    def test_gate_attempts_are_not_collected_slots(self):
        self.assertFalse({g["identity"] for g in self.GATE} & {e["identity"] for e in EPISODES})
        self.assertFalse({g["seed"] for g in self.GATE} & {e["seed"] for e in EPISODES})
        self.assertFalse({tuple(g["start"]) for g in self.GATE} & {tuple(e["start"]) for e in EPISODES})


if __name__ == "__main__":
    unittest.main()
