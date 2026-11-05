"""The A5 arm closure M0 records is complete, and the arm leaves M0's core and the recorded closures of A2, A3 and A6-A9 untouched. Robosuite image, CPU."""
import json
import unittest

from experiments.vla_inputs_eval import adapters, contract, evaluate

IDENTIFIER = "experiments.vla_inputs_arm_xvla_force_box_pred.adapter:XVLAPredictedBoxAdapter"
ROOT = contract.ROOT
CAMPAIGN = ROOT / "outputs/vla_inputs_eval/m0_20261030"
PACKAGE = "experiments/vla_inputs_arm_xvla_force_box_pred"
CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"


class Closure(unittest.TestCase):
    def test_m0_accepts_the_identifier_and_the_module_exposes_the_adapter_and_the_inputs(self):
        self.assertTrue(adapters._IDENTIFIER.match(IDENTIFIER))
        module = evaluate._arm_module(IDENTIFIER)
        self.assertTrue(callable(module.XVLAPredictedBoxAdapter))
        self.assertEqual(module.INPUT_DIRS, (".cache/xvla_trial/bundle", "outputs/vla_inputs_signals/p2_signals_20261030", "outputs/vla_inputs_box_detector/run_v1/final"))

    def test_the_closure_binds_the_adapter_a3s_arm_and_the_detector_modules_it_loads_but_no_test_or_trainer(self):
        files = evaluate.arm_files(IDENTIFIER)
        for rel in (f"{PACKAGE}/adapter.py", f"{PACKAGE}/README.md", "experiments/vla_inputs_arm_xvla_force_box/adapter.py",
                    "experiments/xvla_p2_force_box/eval_adapter.py", "experiments/vla_inputs_box_detector/predictor.py",
                    "experiments/vla_inputs_box_detector/model.py", "experiments/vla_inputs_box_detector/inputs.py",
                    "experiments/vla_inputs_box_detector/constants.py"):
            self.assertIn(rel, files)
        for rel in files:
            self.assertNotIn("/tests/", rel)
        for rel in ("runner.py", "data.py", "gate.py", "contract.py"):
            self.assertNotIn(f"experiments/vla_inputs_box_detector/{rel}", files)  # the trainer, the cache and the gate are not part of a rollout

    def test_the_closure_does_not_repeat_what_the_core_binds_and_the_package_is_not_in_the_core(self):
        self.assertEqual(sorted(set(evaluate.arm_files(IDENTIFIER)) & set(contract.core_provenance())), [])
        for rel in contract.core_provenance():
            self.assertNotIn("vla_inputs_arm_xvla_force_box_pred", rel)
            self.assertNotIn("vla_inputs_box_detector", rel)

    def test_the_package_holds_only_what_the_closure_hashes(self):
        names = sorted(p.name for p in (ROOT / PACKAGE).iterdir() if p.is_file())
        self.assertEqual(names, ["README.md", "__init__.py", "adapter.py"])

    def test_the_readme_states_the_privilege_and_the_confound(self):
        text = (ROOT / PACKAGE / "README.md").read_text().lower()
        for needle in ("force", "oracle_box", "raw 512", "confound", "never saw noise", "predicted_box_logs"):
            self.assertIn(needle, text)

    def test_the_shared_core_hash_is_the_one_a0_was_prepared_with_and_recorded_closures_are_unchanged(self):
        path = CAMPAIGN / "arms/A0/prepared.json"
        if not path.is_file():
            self.skipTest("the campaign is not mounted")
        self.assertEqual(contract.core_sha256(), json.loads(path.read_text())["core"]["sha256"])
        self.assertEqual(contract.core_sha256(), CORE_SHA256)
        for arm in ("A2", "A3", "A6", "A7", "A8", "A9"):
            prepared = CAMPAIGN / f"arms/{arm}/prepared.json"
            if prepared.is_file():
                for rel, digest in json.loads(prepared.read_text())["arm"]["files"].items():
                    self.assertEqual(contract.file_hash(ROOT / rel), digest, (arm, rel))


if __name__ == "__main__":
    unittest.main()
