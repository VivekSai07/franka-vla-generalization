"""Pilot layouts: 192 comparison (48 cells x 4), 2 smoke, an empty selection tier; disjoint from P2, the P2 re-test and M0."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.props_p2_data import plan as p2
from experiments.vla_inputs_eval import contract, evaluate, feasibility, plan as m0
from experiments.vla_inputs_pilot_plan import plan
from experiments.xvla_props_p2_retest import plan as retest

ROOT = Path(__file__).resolve().parents[3]
HAVE_P2 = (m0.P2_ROOT / "plan.json").is_file()
A0_PREPARED = ROOT / "outputs/vla_inputs_eval/m0_20261030/arms/A0/prepared.json"
CORE_TESTS = ROOT / "experiments/vla_inputs_eval/tests/test_evaluate.py"


@unittest.skipUnless(HAVE_P2, "sealed P2 root not mounted")
class Shape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan.make_plan()
        cls.layouts = cls.plan["layouts"]

    def tier(self, name):
        return [self.layouts[i] for i in self.plan["tiers"][name]]

    def test_tier_sizes(self):
        self.assertEqual({k: len(v) for k, v in self.plan["tiers"].items()}, {"smoke": 2, "selection": 0, "comparison": 192})
        self.assertEqual(len(self.layouts), 194)
        self.assertEqual(self.plan["tiers"]["selection"], [])

    def test_comparison_is_four_per_cell_over_all_48_cells(self):
        counts = Counter(tuple(l["cell"]) for l in self.tier("comparison"))
        self.assertEqual(len(counts), 48)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(sorted(counts), sorted(m0.COMPARISON_CELLS))
        held = [l for l in self.tier("comparison") if l["held_out"]]
        self.assertEqual(len(held), 32)
        self.assertTrue(all((l["object"], l["bin"]) in set(p2.HELD_OUT_PAIRS) for l in held))
        self.assertEqual(Counter(l["distractor_count"] for l in self.tier("comparison")), {1: 96, 2: 96})

    def test_smoke_is_two_extra_trained_layouts_with_two_and_one_distractors(self):
        smoke = self.tier("smoke")
        self.assertEqual([l["distractor_count"] for l in smoke], [2, 1])
        self.assertTrue(all(l["tier"] == "smoke" and not l["held_out"] for l in smoke))
        self.assertFalse(set(self.plan["tiers"]["smoke"]) & set(self.plan["tiers"]["comparison"]))

    def test_layouts_carry_the_schema_m0_layouts_carry(self):
        reference = next(iter(m0.make_plan()["layouts"].values()))
        for layout in self.layouts.values():
            self.assertEqual(sorted(layout), sorted(reference))
            self.assertEqual(layout["cell"], [layout["object"], layout["bin"], layout["blue_slot"], layout["distractor_count"]])
            self.assertEqual(len(layout["distractors"]), layout["distractor_count"])
            self.assertNotIn(layout["object"], [d["object"] for d in layout["distractors"]])
            self.assertEqual(layout["instruction"], p2.instruction(layout["object"], layout["bin"]))

    def test_the_namespace_is_the_pilots_own(self):
        self.assertEqual(plan.NAMESPACE, "vla_inputs:pilot")
        self.assertNotEqual(plan.NAMESPACE, m0.NAMESPACE)
        for layout in self.layouts.values():
            self.assertTrue(layout["identity"].startswith("vla_inputs:pilot:"))
            self.assertEqual(layout["layout_id"], layout["identity"].replace(":", "_"))
        self.assertEqual(self.plan["namespace"], "vla_inputs:pilot")
        self.assertEqual(len(self.layouts), len({l["identity"] for l in self.layouts.values()}))

    def test_distractor_decks_cover_every_option_before_repeating(self):
        for cell in m0.COMPARISON_CELLS:
            obj, _, _, count = cell
            options = p2._options(obj, count)
            sets = [tuple(sorted(d["object"] for d in l["distractors"])) for l in self.tier("comparison") if tuple(l["cell"]) == cell]
            self.assertEqual(len(sets), 4)
            if len(options) >= 4:
                self.assertEqual(len(set(sets)), 4, cell)

    def test_deterministic_and_written_with_a_matching_digest(self):
        again = plan.make_plan()
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.plan, sort_keys=True))
        with tempfile.TemporaryDirectory() as tmp:
            value = plan.write(Path(tmp) / "plan")
            self.assertEqual(value, p2.digest(self.plan))
            self.assertEqual((Path(tmp) / "plan/plan.sha256").read_text().strip(), value)
            loaded = feasibility.load_plan(Path(tmp) / "plan")
            self.assertEqual(loaded, json.loads(json.dumps(self.plan)))
            with self.assertRaises(FileExistsError):
                plan.write(Path(tmp) / "plan")

    def test_the_cores_two_digest_functions_agree_on_this_plan(self):
        self.assertEqual(contract.digest(self.plan), p2.digest(self.plan))


@unittest.skipUnless(HAVE_P2, "sealed P2 root not mounted")
class Disjoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = plan.make_plan()
        cls.m0_plan = m0.make_plan()

    @staticmethod
    def values(layouts):
        seeds = {l["seed"] for l in layouts}
        poses = set()
        for l in layouts:
            poses.add(tuple(l["start"]))
            poses |= {tuple(d["pose"]) for d in l["distractors"]}
        return seeds, poses

    def test_no_seed_start_or_distractor_pose_is_shared_with_p2_the_retest_or_either_m0_tier(self):
        seeds, poses = self.values(self.pilot["layouts"].values())
        sealed = json.loads((m0.P2_ROOT / "plan.json").read_text())
        rows = [*sealed["episodes"], *sealed["evaluation_holdout"].values(), *sealed["gate_attempts"],
                *retest.make_plan()["evaluation_holdout"].values()]
        other_seeds = set(sealed["rng_seeds"].values()) | set(retest.make_plan()["seeds"].values()) | {r["seed"] for r in rows}
        other_poses = set()
        for r in rows:
            other_poses.add(tuple(r["start"]))
            other_poses |= {tuple(d["pose"]) for d in r["distractors"]}
        m0_seeds, m0_poses = self.values(self.m0_plan["layouts"].values())
        for name, (s, p) in {"p2+retest": (other_seeds, other_poses), "m0": (m0_seeds, m0_poses)}.items():
            self.assertFalse(seeds & s, f"seeds shared with {name}")
            self.assertFalse(poses & p, f"poses shared with {name}")
        self.assertFalse(set(self.pilot["rng_seeds"].values()) & set(self.m0_plan["rng_seeds"].values()))

    def test_layout_ids_are_disjoint_from_m0(self):
        self.assertFalse(set(self.pilot["layouts"]) & set(self.m0_plan["layouts"]))

    def test_seeds_and_start_poses_are_unique_inside_the_pilot_plan(self):
        layouts = list(self.pilot["layouts"].values())
        self.assertEqual(len({l["seed"] for l in layouts}), len(layouts))
        self.assertEqual(len({tuple(l["start"]) for l in layouts}), len(layouts))

    def test_the_forbidden_set_covers_m0s_plan_as_well_as_p2_and_the_retest(self):
        forbidden = plan.forbidden_values()
        for layout in self.m0_plan["layouts"].values():
            self.assertIn(layout["seed"], forbidden["seeds"])
            self.assertIn(tuple(layout["start"]), forbidden["poses"])
        base = m0.forbidden_values()
        self.assertTrue(base["seeds"] < forbidden["seeds"] and base["poses"] < forbidden["poses"])

    def test_the_disjointness_filter_refuses_every_kind_of_overlap(self):
        layouts = self.pilot["layouts"]
        sample = next(iter(layouts.values()))
        empty = {"seeds": set(), "poses": set()}
        plan.check_disjoint(layouts, empty)
        for label, forbidden in {
                "seed": {"seeds": {sample["seed"]}, "poses": set()},
                "start": {"seeds": set(), "poses": {tuple(sample["start"])}},
                "distractor": {"seeds": set(), "poses": {tuple(next(l for l in layouts.values() if l["distractors"])["distractors"][0]["pose"])}},
        }.items():
            with self.subTest(label), self.assertRaises(ValueError):
                plan.check_disjoint(layouts, forbidden)
        with self.assertRaises(ValueError):
            plan.make_plan(forbidden={"seeds": {sample["seed"]}, "poses": set()})
        collided = {k: dict(v) for k, v in layouts.items()}
        first, second = list(collided.values())[:2]
        second["seed"] = first["seed"]
        with self.assertRaises(ValueError):
            plan.check_unique(collided)

    def test_m0_files_are_not_edited_by_importing_them(self):
        self.assertEqual(m0.NAMESPACE, "vla_inputs:m0")


@unittest.skipUnless(HAVE_P2, "sealed P2 root not mounted")
class CoreServesThisPlan(unittest.TestCase):
    """The frozen core runs this plan in its own campaign root (plan Task 1, now on the real pilot plan)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("m0_core_tests_evaluate", CORE_TESTS)
        cls.core_tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core_tests)
        T = cls.core_tests
        root = Path(tempfile.mkdtemp(prefix="pilot_template_"))
        pilot = plan.make_plan()
        plan.write(root / "campaign/plan")
        (root / "campaign/feasibility").mkdir()
        (root / "campaign/feasibility/feasibility.json").write_text(json.dumps(
            dict(plan_sha256=contract.digest(pilot), feasible=sorted(pilot["layouts"]), dropped=[], dropped_by_cell={}, reasons={})))
        for rel in contract.core_provenance():
            target = root / "repo" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contract.ROOT / rel, target)
        cls.template = root
        T._TEMPLATE["root"] = root

        class Flow(T.Campaign):
            def test_prepare_stage_empty_selection_and_compare(self):
                a0, a1 = self.prepare("A0"), self.prepare("A1")
                self.assertEqual(a0["core"]["sha256"], a1["core"]["sha256"])
                self.stage("A0", "smoke")
                self.stage("A0", "comparison", rule=lambda t: t["ordinal"] % 3 != 0)
                self.stage("A1", "comparison", rule=lambda t: t["ordinal"] % 2 == 0)
                self.assertEqual(self.run_worker("A1", "selection"), [])
                stage = evaluate.collect(self.root, "A1", "selection", 1000)
                self.assertEqual((stage["status"], stage["planned_trials"], stage["episodes"]), ("complete", 0, []))
                decision = evaluate.select(self.root, "A1")
                self.assertEqual((decision["winner"], decision["comparison_checkpoints"]), (1000, [1000]))
                result = evaluate.compare(self.root, "A1", 1000, repo=self.repo)
                self.assertEqual((result["layouts"], result["role"], result["dropped_layouts"]), (192, "primary and matched-phase", []))

            def test_a_dropped_layout_is_dropped_for_every_arm_and_counted_by_cell(self):
                feasible = json.loads((self.root / "feasibility/feasibility.json").read_text())
                lid = self.core_plan["tiers"]["comparison"][0]
                feasible["feasible"].remove(lid)
                feasible["dropped"] = [lid]
                feasible["dropped_by_cell"] = {"-".join(map(str, self.core_plan["layouts"][lid]["cell"])): 1}
                (self.root / "feasibility/feasibility.json").write_text(json.dumps(feasible))
                self.prepare("A0")
                self.prepare("A1")
                self.stage("A0", "comparison")
                self.stage("A1", "comparison", rule=lambda t: t["ordinal"] % 2 == 0)
                self.run_worker("A1", "selection")
                evaluate.collect(self.root, "A1", "selection", 1000)
                evaluate.select(self.root, "A1")
                result = evaluate.compare(self.root, "A1", 1000, repo=self.repo)
                self.assertEqual((result["layouts"], result["dropped_layouts"]), (191, [lid]))

            @property
            def core_plan(self):
                return json.loads((self.root / "plan/plan.json").read_text())

        cls.Flow = Flow

    @classmethod
    def tearDownClass(cls):
        cls.core_tests._TEMPLATE.clear()
        shutil.rmtree(cls.template, ignore_errors=True)

    def test_flow(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.Flow)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual([(str(t), tb) for t, tb in result.failures + result.errors], [])

    @unittest.skipUnless(A0_PREPARED.is_file(), "A0's prepared.json is not mounted")
    def test_the_core_hash_is_the_one_a0_was_prepared_with(self):
        recorded = json.loads(A0_PREPARED.read_text())["core"]["sha256"]
        self.assertEqual(contract.core_sha256(), recorded)
        self.assertTrue(recorded.startswith("c4f6c303f2ae"))


if __name__ == "__main__":
    unittest.main()
