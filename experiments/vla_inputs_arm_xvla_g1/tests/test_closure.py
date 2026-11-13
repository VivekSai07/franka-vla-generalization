"""The G1 arm closure the G0 evaluator and M0 record is complete, and this arm leaves both shared cores untouched. Robosuite image, CPU."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.gen_eval import contract as gen_contract
from experiments.gen_eval import evaluate as gen_evaluate
from experiments.vla_inputs_eval import adapters, contract as m0_contract, evaluate as m0_evaluate

IDENTIFIER = "experiments.vla_inputs_arm_xvla_g1.adapter:XVLAG1Adapter"
ROOT = gen_contract.ROOT
G0_ROOT = Path(os.environ.get("G0_ROOT", ROOT / "outputs/gen_g0/g0_20261109"))
M0_ROOT = ROOT / "outputs/vla_inputs_eval/m0_20261030"
G0_CORE_PREFIX, M0_CORE_PREFIX = "512806282b1f", "c4f6c303f2ae"


def closure(evaluator="gen_eval"):
    """The arm's source closure as `prepare` computes it: in a fresh process, so no other test's imports leak into it."""
    code = (f"import json; from experiments.{evaluator} import evaluate; "
            f"print(json.dumps(sorted(evaluate.arm_files({IDENTIFIER!r}))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class ArmClosureTests(unittest.TestCase):
    def test_both_evaluators_accept_the_identifier_and_the_module_exposes_the_adapter_and_its_inputs(self):
        self.assertTrue(adapters._IDENTIFIER.match(IDENTIFIER))
        for evaluate in (gen_evaluate, m0_evaluate):
            module = evaluate._arm_module(IDENTIFIER)
            self.assertTrue(callable(module.XVLAG1Adapter))
            self.assertTrue(callable(module.state_g1))
        self.assertEqual(module.INPUT_DIRS, (".cache/xvla_trial/bundle", "outputs/gen_g1_data/g1_20261110/signals"))
        self.assertEqual(module.EXTRA_SOURCE, ("outputs/gen_g1_data/g1_20261110/seal.json",))

    def test_the_closure_names_the_adapter_g1s_forces_and_the_seal_and_nothing_from_the_tests_or_the_training_loop(self):
        files = closure()
        for rel in ("experiments/vla_inputs_arm_xvla_g1/adapter.py", "experiments/vla_inputs_arm_xvla_g1/README.md",
                    "experiments/xvla_g1_rand/contract.py", "experiments/xvla_g1_rand/forces.py",
                    "experiments/vla_inputs_arm_xvla/adapter.py", "experiments/xvla_props_p2_eval/protocol.py",
                    "experiments/xvla_props_p2_eval/evaluate.py", "experiments/xvla_single_task/contract.py",
                    "outputs/gen_g1_data/g1_20261110/seal.json"):
            self.assertIn(rel, files)
        for rel in files:
            self.assertNotIn("/tests/", rel)
        self.assertNotIn("experiments/xvla_g1_rand/runner.py", files)

    def test_a2s_forces_and_constants_are_not_in_the_closure(self):
        for evaluator in ("gen_eval", "vla_inputs_eval"):
            for rel in closure(evaluator):
                self.assertNotIn("xvla_p2_force", rel)

    def test_importing_the_arm_does_not_load_a2s_force_module(self):
        code = ("import sys; import experiments.vla_inputs_arm_xvla_g1.adapter; "
                "print([m for m in sys.modules if m.startswith('experiments.xvla_p2_force')])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")

    def test_the_closure_does_not_repeat_what_either_core_binds(self):
        self.assertEqual(sorted(set(closure("gen_eval")) & set(gen_contract.core_provenance())), [])
        self.assertEqual(sorted(set(closure("vla_inputs_eval")) & set(m0_contract.core_provenance())), [])

    def test_the_arms_input_directories_exist_and_are_not_empty(self):
        for rel in gen_evaluate._arm_module(IDENTIFIER).INPUT_DIRS:
            directory = ROOT / rel
            if not directory.is_dir():
                self.skipTest(f"{rel} is not mounted")
            self.assertTrue(any(p.is_file() for p in directory.rglob("*")), rel)

    def test_neither_package_of_this_arm_is_in_a_shared_core(self):
        for core in (gen_contract.core_provenance(), m0_contract.core_provenance()):
            for rel in core:
                self.assertNotIn("xvla_g1_rand", rel)
                self.assertNotIn("vla_inputs_arm_xvla_g1", rel)

    def test_the_g0_core_hash_equals_the_one_a2_was_measured_with(self):
        prepared = G0_ROOT / "arms/A2/prepared.json"
        if not prepared.is_file():
            self.skipTest("G0's campaign is not mounted")
        recorded = json.loads(prepared.read_text())
        self.assertTrue(recorded["core"]["sha256"].startswith(G0_CORE_PREFIX))
        self.assertEqual(gen_contract.core_sha256(), recorded["core"]["sha256"])
        self.assertEqual(gen_contract.core_provenance(), recorded["core"]["provenance"])

    def test_the_m0_core_hash_equals_the_one_a0_was_prepared_with(self):
        prepared = M0_ROOT / "arms/A0/prepared.json"
        if not prepared.is_file():
            self.skipTest("M0's campaign is not mounted")
        recorded = json.loads(prepared.read_text())
        self.assertTrue(recorded["core"]["sha256"].startswith(M0_CORE_PREFIX))
        self.assertEqual(m0_contract.core_sha256(), recorded["core"]["sha256"])


if __name__ == "__main__":
    unittest.main()
