"""Task 1 canonical plan tests; no simulator or policy imports."""
import copy
import hashlib
import importlib
import json
import unittest


class ContractTests(unittest.TestCase):
    def setUp(self):
        try:
            self.c = importlib.import_module("experiments.fixed_layout.contract")
        except ModuleNotFoundError as exc:
            self.fail(f"Task 1 contract is not implemented: {exc}")

    def test_exact_schedule_and_same_geometry(self):
        plan = self.c.make_plan()
        self.assertEqual(len(plan["episodes"]), 24)
        self.assertEqual(plan["train_ids"], list(range(20)))
        self.assertEqual(plan["validation_ids"], list(range(20, 24)))
        self.assertEqual(plan["preparation_ids"], [0, 1, 2])
        self.assertIs(plan["validation_is_spatial_holdout"], False)
        for i, episode in enumerate(plan["episodes"]):
            self.assertEqual(episode["episode_index"], i)
            self.assertEqual(episode["split"], "train" if i < 20 else "validation")
            self.assertEqual((episode["pick"], episode["place"]), ("red", "green"))
            self.assertEqual(episode["colors"], ["red", "green"])
            self.assertEqual(episode["source_episode_index"], 7)
            self.assertEqual(episode["scene_hash"],
                             "88060815ffd36e578284aeda158cbaac38e0149db398f09c4cf22bad48150efd")
            self.assertEqual(episode["instruction"],
                             "pick up the red cube and place it on top of the green cube")
        self.assertIsNone(self.c.validate_plan(plan))

    def test_seed_formula_is_independently_reconstructed(self):
        seeds = [e["seed"] for e in self.c.make_plan()["episodes"]]
        expected = [int(hashlib.sha256(
            f"fixed_layout_baseline_v1:20260918:expert:{i}".encode("ascii")
        ).hexdigest()[:8], 16) for i in range(24)]
        self.assertEqual(seeds, expected)
        self.assertEqual(len(set(seeds)), 24)

    def test_fixed_source_schema_and_budgets(self):
        plan = self.c.make_plan()
        self.assertEqual(plan["source"]["sha256"],
                         "ca5c19eda5d089a4a33a838a3fc3e55b44895f77aa40e8d92d9490953290221b")
        self.assertEqual(plan["source"]["path"],
                         "datasets/franka_pilot_20260915/replay/episode_000007.npz")
        self.assertEqual(plan["expected_episode_frames"], 528)
        self.assertEqual((plan["fps"], plan["physics_hz"], plan["substeps_per_control"]), (30, 600, 20))
        self.assertEqual((plan["state_dim"], plan["action_dim"]), (9, 8))
        self.assertEqual(plan["camera_size"], [512, 512])
        self.assertEqual(plan["encoding"]["vcodec"], "libsvtav1")
        self.assertEqual(plan["encoding"]["pix_fmt"], "yuv420p")
        self.assertEqual(plan["expert_attempts"], 24)
        self.assertEqual(plan["replay_modes"], ["reference", "persisted_float32"])
        self.assertEqual(plan["failure_policy"], "first_failure_stop_no_retry_no_resume")
        self.assertEqual(plan["min_free_bytes"], 30 * 1024**3)
        self.assertEqual(plan["max_seconds"], 3 * 60 * 60)
        self.assertEqual(self.c.MIN_FREE_BYTES, plan["min_free_bytes"])
        self.assertEqual(self.c.MAX_SECONDS, plan["max_seconds"])
        self.assertEqual(self.c.OUTPUT, self.c.ROOT / "outputs/fixed_layout/fixed_layout_20260918")

    def test_returned_plans_share_no_mutable_state(self):
        first = self.c.make_plan()
        first["episodes"][0]["colors"].append("blue")
        first["source"]["sha256"] = "0" * 64
        self.c.validate_plan(self.c.make_plan())
        self.c.validate_plan(json.loads(json.dumps(self.c.make_plan())))

    def test_roles_split_order_seed_and_source_mutations_rejected(self):
        mutations = [
            lambda p: p["episodes"][0].update(pick="green"),
            lambda p: p["episodes"][0].update(split="validation"),
            lambda p: p["episodes"].reverse(),
            lambda p: p["episodes"][0].update(seed=42),
            lambda p: p["train_ids"].pop(),
            lambda p: p["validation_ids"].reverse(),
            lambda p: p["preparation_ids"].append(3),
            lambda p: p["source"].update(sha256="0" * 64),
            lambda p: p["episodes"][0].update(scene_hash="0" * 64),
            lambda p: p.update(validation_is_spatial_holdout=True),
            lambda p: p.update(min_free_bytes=0),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(self.c.make_plan())
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.c.validate_plan(changed)

    def test_missing_and_unknown_fields_rejected_at_all_levels(self):
        for keys in ((), ("episodes", 0), ("source",), ("encoding",)):
            for missing in (True, False):
                changed = self.c.make_plan()
                node = changed
                for key in keys:
                    node = node[key]
                if missing:
                    node.pop(next(iter(node)))
                else:
                    node["unapproved"] = True
                with self.subTest(keys=keys, missing=missing), self.assertRaises(ValueError):
                    self.c.validate_plan(changed)

    def test_bool_float_tuple_and_nonfinite_type_spoofing_rejected(self):
        for key, value in (("episode_index", False), ("episode_index", 0.0),
                           ("seed", True), ("seed", float("nan")),
                           ("colors", ("red", "green"))):
            changed = self.c.make_plan()
            changed["episodes"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.c.validate_plan(changed)
        for value in (None, [], True):
            with self.assertRaises(ValueError):
                self.c.validate_plan(value)


if __name__ == "__main__":
    unittest.main()