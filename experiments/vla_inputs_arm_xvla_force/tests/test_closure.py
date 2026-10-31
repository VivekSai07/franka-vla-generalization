"""The A2 arm closure M0 records is complete, and this arm leaves M0's shared core untouched. Robosuite image, CPU."""
import json
import unittest
from pathlib import Path

from experiments.vla_inputs_eval import adapters, contract, evaluate

IDENTIFIER = "experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter"
ROOT = contract.ROOT
A0_PREPARED = ROOT / "outputs/vla_inputs_eval/m0_20261030/arms/A0/prepared.json"


class ArmClosureTests(unittest.TestCase):
    def test_m0_accepts_the_identifier_and_the_module_exposes_the_adapter_and_its_inputs(self):
        self.assertTrue(adapters._IDENTIFIER.match(IDENTIFIER))
        module = evaluate._arm_module(IDENTIFIER)
        self.assertTrue(callable(module.XVLAForceAdapter))
        self.assertTrue(callable(module.state_force))
        self.assertEqual(module.INPUT_DIRS, (".cache/xvla_trial/bundle", "outputs/vla_inputs_signals/p2_signals_20261030"))

    def test_the_closure_names_the_adapter_everything_it_loads_and_nothing_from_the_tests(self):
        files = evaluate.arm_files(IDENTIFIER)
        for rel in ("experiments/vla_inputs_arm_xvla_force/adapter.py", "experiments/vla_inputs_arm_xvla_force/README.md",
                    "experiments/xvla_p2_force/eval_adapter.py", "experiments/xvla_p2_force/contract.py",
                    "experiments/xvla_p2_force/forces.py", "experiments/vla_inputs_arm_xvla/adapter.py",
                    "experiments/xvla_props_p2_eval/protocol.py", "experiments/xvla_props_p2_eval/evaluate.py",
                    "experiments/xvla_single_task/contract.py"):
            self.assertIn(rel, files)
        for rel in files:
            self.assertNotIn("/tests/", rel)
            self.assertNotIn("eval_tests", rel)
        self.assertNotIn("experiments/xvla_p2_force/runner.py", files)  # the training loop is not part of a rollout

    def test_the_closure_does_not_repeat_what_the_core_binds(self):
        core = set(contract.core_provenance())
        self.assertEqual(sorted(set(evaluate.arm_files(IDENTIFIER)) & core), [])

    def test_the_arms_input_directories_exist_and_are_not_empty(self):
        for rel in evaluate._arm_module(IDENTIFIER).INPUT_DIRS:
            directory = ROOT / rel
            if not directory.is_dir():
                self.skipTest(f"{rel} is not mounted")
            self.assertTrue(any(p.is_file() for p in directory.rglob("*")), rel)

    def test_neither_package_of_this_arm_is_in_the_shared_core(self):
        for rel in contract.core_provenance():
            self.assertNotIn("xvla_p2_force", rel)
            self.assertNotIn("vla_inputs_arm_xvla_force", rel)

    def test_the_shared_core_hash_equals_the_one_a0_was_prepared_with(self):
        if not A0_PREPARED.is_file():
            self.skipTest("A0's prepared.json is not mounted")
        prepared = json.loads(A0_PREPARED.read_text())
        self.assertEqual(contract.core_sha256(), prepared["core"]["sha256"])
        self.assertEqual(contract.core_provenance(), prepared["core"]["provenance"])


if __name__ == "__main__":
    unittest.main()
