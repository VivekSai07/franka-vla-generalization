"""The arm closure M0 records for A6-A9 is complete, and these arms leave M0's core and A2's and A3's frozen files untouched.

Robosuite image, CPU. The layout tests run the real scene builder: by default one layout per cell of the comparison tier, and
every layout of the tier (shardable) when `M3B_FULL_LAYOUTS=1` (`M3B_SHARD`, `M3B_SHARDS`).
"""
import json
import os
import unittest

from experiments.vla_inputs_eval import adapters, contract, evaluate
from experiments.vla_inputs_arm_xvla_force_box_noisy import corruption

PREFIX = "experiments.vla_inputs_arm_xvla_force_box_noisy.adapter:"
CLASSES = ("XVLAJitterSmall", "XVLAJitterLarge", "XVLADropout", "XVLASwap", "XVLANoCorruption")
ROOT = contract.ROOT
CAMPAIGN = ROOT / "outputs/vla_inputs_eval/m0_20261030"
PACKAGE = "experiments/vla_inputs_arm_xvla_force_box_noisy"
CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"


class ArmClosureTests(unittest.TestCase):
    def test_m0_accepts_every_identifier_and_the_module_exposes_the_classes_and_the_inputs(self):
        module = evaluate._arm_module(PREFIX + "XVLASwap")
        for name in CLASSES:
            self.assertTrue(adapters._IDENTIFIER.match(PREFIX + name), name)
            self.assertTrue(callable(getattr(module, name)), name)
        self.assertEqual(module.INPUT_DIRS, (".cache/xvla_trial/bundle", "outputs/vla_inputs_signals/p2_signals_20261030"))

    def test_the_closure_names_this_package_a3s_arm_and_what_they_load_and_nothing_from_the_tests(self):
        files = evaluate.arm_files(PREFIX + "XVLASwap")
        for rel in (f"{PACKAGE}/adapter.py", f"{PACKAGE}/corruption.py", f"{PACKAGE}/README.md",
                    "experiments/vla_inputs_arm_xvla_force_box/adapter.py",
                    "experiments/xvla_p2_force_box/eval_adapter.py", "experiments/xvla_p2_force_box/overlay.py",
                    "experiments/xvla_p2_force_box/contract.py", "experiments/xvla_p2_force/eval_adapter.py",
                    "experiments/vla_inputs_arm_xvla/adapter.py"):
            self.assertIn(rel, files)
        for rel in files:
            self.assertNotIn("/tests/", rel)
            self.assertNotIn("eval_tests", rel)
        self.assertNotIn("experiments/xvla_p2_force_box/runner.py", files)
        self.assertNotIn("experiments/xvla_p2_force_box/eval_compare.py", files)

    def test_the_closure_does_not_repeat_what_the_core_binds(self):
        self.assertEqual(sorted(set(evaluate.arm_files(PREFIX + "XVLAJitterSmall")) & set(contract.core_provenance())), [])

    def test_this_package_is_not_in_the_shared_core(self):
        for rel in contract.core_provenance():
            self.assertNotIn("vla_inputs_arm_xvla_force_box_noisy", rel)

    def test_the_arm_directory_holds_only_what_the_closure_hashes(self):
        """Every top-level file of the package is bound by `prepare`; a stray helper would silently be bound too."""
        names = sorted(p.name for p in (ROOT / PACKAGE).iterdir() if p.is_file())
        self.assertEqual(names, ["README.md", "__init__.py", "adapter.py", "corruption.py"])

    def test_the_package_readme_states_the_constants_and_the_privilege(self):
        text = (ROOT / PACKAGE / "README.md").read_text().lower()
        self.assertGreater(len(text), 1500)
        for needle in ("jitter_small", "jitter_large", "dropout", "swap", "6 px", "18 px", "0.5", "force", "a6", "a9", "mean iou"):
            self.assertIn(needle, text)

    def test_the_shared_core_hash_equals_the_one_a0_was_prepared_with(self):
        prepared_path = CAMPAIGN / "arms/A0/prepared.json"
        if not prepared_path.is_file():
            self.skipTest("A0's prepared.json is not mounted")
        prepared = json.loads(prepared_path.read_text())
        self.assertEqual(contract.core_sha256(), prepared["core"]["sha256"])
        self.assertEqual(prepared["core"]["sha256"], CORE_SHA256)

    def test_a2s_and_a3s_recorded_closures_are_unchanged_by_this_arm(self):
        for arm in ("A2", "A3"):
            path = CAMPAIGN / f"arms/{arm}/prepared.json"
            if not path.is_file():
                self.skipTest(f"{arm}'s prepared.json is not mounted")
            for rel, digest in json.loads(path.read_text())["arm"]["files"].items():
                self.assertEqual(contract.file_hash(ROOT / rel), digest, (arm, rel))


class EveryLayoutHasADistractorForTheSwap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = CAMPAIGN / "plan/plan.json"
        if not path.is_file():
            raise unittest.SkipTest("the M0 plan is not mounted")
        from experiments.vla_inputs_eval import feasibility
        cls.plan = feasibility.load_plan(CAMPAIGN / "plan")

    def comparison(self):
        return [self.plan["layouts"][i] for i in self.plan["tiers"]["comparison"]]

    def test_the_plan_gives_every_comparison_layout_one_or_two_distinct_distractors_other_than_the_target(self):
        layouts = self.comparison()
        self.assertEqual(len(layouts), 576)
        for layout in layouts:
            names = [d["object"] for d in layout["distractors"]]
            self.assertIn(len(names), (1, 2), layout["layout_id"])
            self.assertEqual(len(set(names)), len(names))
            self.assertNotIn(layout["object"], names)

    def layouts_to_build(self):
        layouts = self.comparison()
        if os.environ.get("M3B_FULL_LAYOUTS") == "1":
            shard, shards = int(os.environ.get("M3B_SHARD", "0")), int(os.environ.get("M3B_SHARDS", "1"))
            return layouts[shard::shards]
        cells = {}
        for layout in layouts:
            cells.setdefault(tuple(layout["cell"]), layout)
        return list(cells.values())

    def test_the_scene_restores_and_the_swap_picks_one_of_the_planned_distractors(self):
        from experiments.vla_inputs_arm_xvla_force_box_noisy.adapter import prop_positions
        from experiments.vla_inputs_eval import scenes
        built = 0
        for layout in self.layouts_to_build():
            model, data, names = scenes.restore(layout)
            positions = prop_positions(model, data)
            self.assertEqual(sorted(positions), sorted(names), layout["layout_id"])
            chosen = corruption.nearest_distractor(layout["object"], positions)
            self.assertIn(chosen, [d["object"] for d in layout["distractors"]], layout["layout_id"])
            self.assertNotEqual(chosen, layout["object"])
            built += 1
        print(f"SWAP_LAYOUTS_CHECKED {built}", flush=True)


if __name__ == "__main__":
    unittest.main()
