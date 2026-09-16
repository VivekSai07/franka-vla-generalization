"""Pure-stdlib contract tests for the balanced pilot manifest."""
import copy
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


COLORS = ("red", "green", "blue", "yellow")
PAIRS = tuple((pick, place) for pick in COLORS for place in COLORS if pick != place)
ROOT = Path(__file__).resolve().parents[1]


class PilotPlanTests(unittest.TestCase):
    def setUp(self):
        # Report a missing implementation as the intended initial RED failure.
        try:
            self.api = importlib.import_module("data_collection.pilot_plan")
        except ModuleNotFoundError as exc:
            if exc.name != "data_collection.pilot_plan":
                raise
            self.fail("data_collection.pilot_plan is not implemented")

    def assert_slot_schema(self, slot, retries):
        self.assertEqual(set(slot), {
            "slot_id", "split", "pair_id", "pick", "place", "colors", "retry_seeds",
        })
        self.assertIs(type(slot["slot_id"]), str)
        self.assertTrue(slot["slot_id"])
        self.assertIs(type(slot["pair_id"]), int)
        self.assertIn(slot["pair_id"], range(12))
        self.assertEqual((slot["pick"], slot["place"]), PAIRS[slot["pair_id"]])
        self.assertIs(type(slot["colors"]), list)
        self.assertEqual(slot["colors"][:2], [slot["pick"], slot["place"]])
        self.assertIn(len(slot["colors"]), (2, 3, 4))
        self.assertEqual(len(set(slot["colors"])), len(slot["colors"]))
        self.assertTrue(set(slot["colors"]).issubset(COLORS))
        self.assertIs(type(slot["retry_seeds"]), list)
        self.assertEqual(len(slot["retry_seeds"]), retries)
        self.assertEqual(len(set(slot["retry_seeds"])), retries)
        for seed in slot["retry_seeds"]:
            self.assertIs(type(seed), int)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**32)

    def assert_quotas(self, slots, per_pair, totals, rotated):
        self.assertEqual(len(slots), 12 * per_pair)
        self.assertEqual(Counter(s["pair_id"] for s in slots),
                         Counter({p: per_pair for p in range(12)}))
        self.assertEqual(Counter(len(s["colors"]) - 2 for s in slots),
                         Counter(dict(enumerate(totals))))
        for pair_id, pair in enumerate(PAIRS):
            group = [s for s in slots if s["pair_id"] == pair_id]
            self.assertEqual(Counter(len(s["colors"]) - 2 for s in group),
                             Counter(dict(enumerate(rotated[pair_id % 3]))))
            singles = [s["colors"][2] for s in group if len(s["colors"]) == 3]
            eligible = set(COLORS) - set(pair)
            self.assertEqual(set(singles), eligible)
            counts = [singles.count(color) for color in eligible]
            self.assertLessEqual(abs(counts[0] - counts[1]), 1)
            self.assertTrue(all(a != b for a, b in zip(singles, singles[1:])))

    def test_pilot_exact_quotas_and_training_prefix(self):
        plan = self.api.make_plan("pilot", 20260915)
        self.assertEqual(set(plan), {
            "version", "mode", "seed", "max_attempts_per_slot", "slots", "splits",
            "scene_slots",
        })
        self.assertTrue(plan["version"])
        self.assertEqual(plan["mode"], "pilot")
        self.assertEqual(plan["seed"], 20260915)
        self.assertEqual(plan["max_attempts_per_slot"], 10)
        self.assertEqual(len(plan["slots"]), 420)
        self.assertEqual([s["split"] for s in plan["slots"]],
                         ["train"] * 300 + ["validation"] * 120)
        for slot in plan["slots"]:
            self.assert_slot_schema(slot, 10)
        self.assert_quotas(plan["slots"][:300], 25, (100, 100, 100),
                           ((9, 8, 8), (8, 9, 8), (8, 8, 9)))
        self.assert_quotas(plan["slots"][300:], 10, (40, 40, 40),
                           ((4, 3, 3), (3, 4, 3), (3, 3, 4)))

    def test_preparation_one_per_pair_and_fixed_splits(self):
        plan = self.api.make_plan("preparation", 0)
        self.assertEqual(len(plan["slots"]), 12)
        self.assertEqual([s["pair_id"] for s in plan["slots"]], list(range(12)))
        self.assertEqual([len(s["colors"]) - 2 for s in plan["slots"]],
                         [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual([s["split"] for s in plan["slots"]],
                         ["train"] * 9 + ["validation"] * 3)
        self.assertEqual(Counter(len(s["colors"]) - 2 for s in plan["slots"]),
                         {0: 4, 1: 4, 2: 4})
        self.assertEqual(plan["max_attempts_per_slot"], 10)
        for slot in plan["slots"]:
            self.assert_slot_schema(slot, 10)

    def test_scene_banks_have_exact_unfiltered_quotas_and_one_seed(self):
        for mode in ("pilot", "preparation"):
            with self.subTest(mode=mode):
                scenes = self.api.make_plan(mode, 123)["scene_slots"]
                self.assertEqual([s["split"] for s in scenes],
                                 ["tuning"] * 120 + ["final"] * 240)
                for scene in scenes:
                    self.assert_slot_schema(scene, 1)
                self.assert_quotas(scenes[:120], 10, (40, 40, 40),
                                   ((4, 3, 3), (3, 4, 3), (3, 3, 4)))
                self.assert_quotas(scenes[120:], 20, (80, 80, 80),
                                   ((7, 7, 6), (6, 7, 7), (7, 6, 7)))

    def test_splits_are_complete_ordered_disjoint_slot_ids(self):
        for mode in ("pilot", "preparation"):
            plan = self.api.make_plan(mode, 9)
            self.assertEqual(set(plan["splits"]), {"train", "validation"})
            for split in ("train", "validation"):
                self.assertEqual(plan["splits"][split],
                                 [s["slot_id"] for s in plan["slots"] if s["split"] == split])
            self.assertFalse(set(plan["splits"]["train"]) & set(plan["splits"]["validation"]))
            ids = [s["slot_id"] for s in plan["slots"] + plan["scene_slots"]]
            self.assertEqual(len(set(ids)), len(ids))

    def test_all_attempt_and_scene_seeds_are_uint32_and_disjoint(self):
        for seed in (0, 1, 42, 20260915, 2**32 - 1):
            with self.subTest(seed=seed):
                combined = []
                for mode, expected in (("pilot", 4560), ("preparation", 480)):
                    plan = self.api.make_plan(mode, seed)
                    streams = {}
                    for slot in plan["slots"] + plan["scene_slots"]:
                        self.assert_slot_schema(slot, 10 if slot["split"] in ("train", "validation") else 1)
                        streams.setdefault(slot["split"], []).extend(slot["retry_seeds"])
                    values = [s for stream in streams.values() for s in stream]
                    self.assertEqual(len(values), expected)
                    self.assertEqual(len(set(values)), expected)
                    combined.extend(values)
                # Preparation is a separate namespace, never pilot attempts.
                self.assertEqual(len(set(combined)), len(combined))

    def test_determinism_json_roundtrip_and_no_shared_mutable_state(self):
        for mode in ("pilot", "preparation"):
            original = self.api.make_plan(mode, 17)
            before = copy.deepcopy(original)
            self.assertEqual(original, self.api.make_plan(mode, 17))
            self.assertNotEqual(original, self.api.make_plan(mode, 18))
            decoded = json.loads(json.dumps(original))
            self.assertIsNone(self.api.validate_plan(decoded))
            self.assertEqual(original, before)
            original["slots"][0]["colors"].append("invalid")
            original["splits"]["train"].clear()
            self.assertEqual(self.api.make_plan(mode, 17), before)

    def test_single_distractor_order_is_seeded(self):
        signatures = set()
        for seed in range(8):
            slots = self.api.make_plan("pilot", seed)["slots"]
            signatures.add(tuple(s["colors"][2] for s in slots if len(s["colors"]) == 3))
        self.assertGreater(len(signatures), 1)

    def test_master_seed_requires_exact_uint32_int(self):
        class IntSubclass(int):
            pass

        for bad in (True, False, -1, 2**32, 1.0, "1", None, IntSubclass(1)):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                self.api.make_plan("pilot", bad)
        for good in (0, 2**32 - 1):
            self.assertEqual(self.api.make_plan("preparation", good)["seed"], good)

    def test_invalid_modes_are_rejected(self):
        for bad in ("train", "Pilot", "", None, True, 1, []):
            with self.subTest(mode=bad), self.assertRaises(ValueError):
                self.api.make_plan(bad, 0)

    def test_retry_lookup_preserves_condition_and_stops_at_ten(self):
        slot = self.api.make_plan("pilot", 41)["slots"][9]
        before = copy.deepcopy(slot)
        attempts = [self.api.slot_seed(slot, retry) for retry in range(10)]
        self.assertEqual(attempts, slot["retry_seeds"])
        self.assertEqual(len(set(attempts)), 10)
        self.assertEqual(slot, before)
        for bad in (-1, 10, 11, 2**32, True, False, 0.0, "0", None):
            with self.subTest(retry=bad), self.assertRaises(ValueError):
                self.api.slot_seed(slot, bad)

    def test_scene_seed_has_no_success_retry(self):
        for scene in self.api.make_plan("pilot", 41)["scene_slots"]:
            self.assertEqual(self.api.slot_seed(scene, 0), scene["retry_seeds"][0])
            for retry in (1, 9, 10, -1):
                with self.assertRaises(ValueError):
                    self.api.slot_seed(scene, retry)

    def test_retry_lookup_rejects_malformed_seed_lists(self):
        slot = self.api.make_plan("preparation", 1)["slots"][0]
        for bad in ([], [1], list(range(9)), list(range(11)), [1] * 10,
                    tuple(range(10)), [True] + list(range(1, 10)),
                    [-1] + list(range(1, 10)), [2**32] + list(range(1, 10))):
            altered = dict(slot, retry_seeds=bad)
            with self.subTest(seeds=bad), self.assertRaises(ValueError):
                self.api.slot_seed(altered, 0)

    def test_validate_rejects_every_top_level_missing_or_extra_field(self):
        plan = self.api.make_plan("pilot", 123)
        for key in plan:
            changed = copy.deepcopy(plan)
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.api.validate_plan(changed)
        with self.assertRaises(ValueError):
            self.api.validate_plan(dict(plan, unexpected=True))
        for bad in (None, [], "{}", 1):
            with self.assertRaises(ValueError):
                self.api.validate_plan(bad)

    def test_validate_rejects_mutations_even_when_totals_still_match(self):
        mutations = [
            (("version",), "future"),
            (("mode",), "preparation"),
            (("seed",), 124),
            (("max_attempts_per_slot",), 11),
            (("slots", 0, "slot_id"), "replacement"),
            (("slots", 0, "split"), "validation"),
            (("slots", 0, "pair_id"), 1),
            (("slots", 0, "pick"), "blue"),
            (("slots", 0, "place"), "yellow"),
            (("slots", 0, "colors", 0), "blue"),
            (("slots", 0, "retry_seeds", 0), 0),
            (("scene_slots", 0, "split"), "final"),
            (("scene_slots", 0, "retry_seeds"), [1, 2]),
        ]
        baseline = self.api.make_plan("pilot", 123)
        for path, replacement in mutations:
            changed = copy.deepcopy(baseline)
            target = changed
            for key in path[:-1]:
                target = target[key]
            self.assertNotEqual(target[path[-1]], replacement)
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.api.validate_plan(changed)
        for field in ("slots", "scene_slots"):
            for operation in ("reverse", "duplicate", "drop", "seed_swap", "extra_key"):
                changed = copy.deepcopy(baseline)
                values = changed[field]
                if operation == "reverse":
                    values.reverse()
                elif operation == "duplicate":
                    values[1] = copy.deepcopy(values[0])
                elif operation == "drop":
                    values.pop()
                elif operation == "seed_swap":
                    values[0]["retry_seeds"], values[1]["retry_seeds"] = (
                        values[1]["retry_seeds"], values[0]["retry_seeds"])
                else:
                    values[0]["unexpected"] = True
                with self.subTest(field=field, operation=operation), self.assertRaises(ValueError):
                    self.api.validate_plan(changed)
        changed = copy.deepcopy(baseline)
        changed["slots"][0], changed["slots"][1] = changed["slots"][1], changed["slots"][0]
        changed["splits"]["train"] = [s["slot_id"] for s in changed["slots"][:300]]
        with self.assertRaises(ValueError):
            self.api.validate_plan(changed)
        # A different eligible identity preserves pair/count quotas but is not canonical.
        changed = copy.deepcopy(baseline)
        single = next(s for s in changed["slots"] if len(s["colors"]) == 3)
        single["colors"][2] = next(c for c in COLORS if c not in single["colors"])
        with self.assertRaises(ValueError):
            self.api.validate_plan(changed)

    def test_validate_rejects_split_reassignment_reordering_and_duplicates(self):
        for operation in ("swap", "reverse", "duplicate", "drop", "extra"):
            changed = self.api.make_plan("pilot", 2)
            splits = changed["splits"]
            if operation == "swap":
                splits["train"][0], splits["validation"][0] = splits["validation"][0], splits["train"][0]
            elif operation == "reverse":
                splits["train"].reverse()
            elif operation == "duplicate":
                splits["train"][1] = splits["train"][0]
            elif operation == "drop":
                splits["validation"].pop()
            else:
                splits["tuning"] = []
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.api.validate_plan(changed)

    def test_validate_rejects_python_equal_noncanonical_types(self):
        baseline = self.api.make_plan("pilot", 1)
        mutations = [
            (("seed",), True), (("seed",), 1.0),
            (("max_attempts_per_slot",), 10.0),
            (("slots", 0, "pair_id"), False),
            (("slots", 0, "pair_id"), 0.0),
            (("slots", 0, "retry_seeds", 0), float(baseline["slots"][0]["retry_seeds"][0])),
            (("slots", 0, "colors"), tuple(baseline["slots"][0]["colors"])),
            (("slots",), tuple(baseline["slots"])),
            (("scene_slots", 0, "pair_id"), False),
        ]
        for path, replacement in mutations:
            changed = copy.deepcopy(baseline)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            with self.subTest(path=path, value_type=type(replacement)), self.assertRaises(ValueError):
                self.api.validate_plan(changed)

    def test_digest_is_sha256_of_canonical_json_and_ignores_dict_key_order(self):
        plan = self.api.make_plan("pilot", 42)
        canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(self.api.plan_digest(plan), expected)
        self.assertRegex(expected, r"^[0-9a-f]{64}$")

        def reverse_keys(value):
            if isinstance(value, dict):
                return {key: reverse_keys(item) for key, item in reversed(list(value.items()))}
            if isinstance(value, list):
                return [reverse_keys(item) for item in value]
            return value

        reordered = reverse_keys(plan)
        self.assertIsNone(self.api.validate_plan(reordered))
        self.assertEqual(self.api.plan_digest(reordered), expected)
        self.assertNotEqual(self.api.plan_digest(self.api.make_plan("pilot", 43)), expected)
        self.assertNotEqual(self.api.plan_digest(self.api.make_plan("preparation", 42)), expected)
        plan["slots"][0]["pick"] = "invalid"
        with self.assertRaises(ValueError):
            self.api.plan_digest(plan)

    def run_cli(self, *args, script=False):
        entry = [str(ROOT / "data_collection/pilot_plan.py")] if script else [
            "-m", "data_collection.pilot_plan",
        ]
        return subprocess.run([sys.executable, "-B", "-S", *entry, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=20)

    def test_cli_writes_valid_json_and_refuses_overwrite(self):
        for mode in ("pilot", "preparation"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "plan.json"
                args = ("--mode", mode, "--seed", "4294967295", "--output", str(output))
                result = self.run_cli(*args, script=mode == "preparation")
                self.assertEqual(result.returncode, 0, result.stderr)
                saved = output.read_bytes()
                decoded = json.loads(saved)
                self.assertEqual(decoded, self.api.make_plan(mode, 2**32 - 1))
                self.assertIsNone(self.api.validate_plan(decoded))
                retry = self.run_cli(*args)
                self.assertNotEqual(retry.returncode, 0)
                self.assertEqual(output.read_bytes(), saved)

    def test_cli_rejects_invalid_or_missing_arguments_without_creating_output(self):
        cases = [
            ("--mode", "invalid", "--seed", "1"),
            ("--mode", "pilot", "--seed", "-1"),
            ("--mode", "pilot", "--seed", "4294967296"),
            ("--mode", "pilot", "--seed", "True"),
            ("--mode", "pilot", "--seed", "1.0"),
            ("--mode", "pilot"), ("--seed", "1"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            for args in cases:
                with self.subTest(args=args):
                    result = self.run_cli(*args, "--output", str(output))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())
        self.assertNotEqual(self.run_cli("--mode", "pilot", "--seed", "1").returncode, 0)


if __name__ == "__main__":
    unittest.main()