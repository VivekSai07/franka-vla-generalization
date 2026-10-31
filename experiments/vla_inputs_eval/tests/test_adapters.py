import subprocess
import sys
import types
import unittest

import numpy as np

from experiments.vla_inputs_eval import adapters


class FakeAdapter:
    name = "fake"

    def __init__(self, scale=1.0):
        self.scale = scale
        self.calls = 0

    def reset(self):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return np.full(8, self.scale)


def observation(**changes):
    obs = dict(images=[np.zeros((512, 512, 3), np.uint8), np.zeros((512, 512, 3), np.uint8)], state=np.zeros(9),
               instruction="put the can in the blue bin")
    obs.update(changes)
    return obs


class Loader(unittest.TestCase):
    def test_loads_an_adapter_by_explicit_identifier_and_passes_arguments(self):
        module = types.ModuleType("experiments.vla_inputs_arm_fake")
        module.FakeAdapter = FakeAdapter
        sys.modules["experiments.vla_inputs_arm_fake"] = module
        try:
            adapter = adapters.load_adapter("experiments.vla_inputs_arm_fake:FakeAdapter", scale=3.0)
        finally:
            del sys.modules["experiments.vla_inputs_arm_fake"]
        self.assertEqual(adapter.act(observation()).tolist(), [3.0] * 8)

    def test_refuses_an_identifier_outside_the_arm_namespace(self):
        for bad in ("os:system", "experiments.vla_inputs_eval.plan:make_plan", "experiments.vla_inputs_arm_x", "nonsense",
                    "experiments.vla_inputs_arm_x:Y:Z", "experiments.vla_inputs_arm_x/../..:Y"):
            with self.assertRaises(ValueError, msg=bad):
                adapters.load_adapter(bad)

    def test_an_object_that_is_not_an_adapter_is_refused(self):
        class NotAnAdapter:
            name = "x"

            def act(self, obs):
                return np.zeros(8)

        module = types.ModuleType("experiments.vla_inputs_arm_bad")
        module.Bad = NotAnAdapter
        sys.modules["experiments.vla_inputs_arm_bad"] = module
        try:
            with self.assertRaises(TypeError):
                adapters.load_adapter("experiments.vla_inputs_arm_bad:Bad")
        finally:
            del sys.modules["experiments.vla_inputs_arm_bad"]

    def test_importing_the_core_does_not_import_a_model_stack(self):
        code = ("import sys; import experiments.vla_inputs_eval.adapters; "
                "print(sorted(m for m in ('torch', 'lerobot', 'transformers', 'robosuite', 'mujoco') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr[-300:])


class Validation(unittest.TestCase):
    def test_a_good_observation_passes(self):
        adapters.check_observation(observation())

    def test_each_defect_is_refused(self):
        bad = [observation(images=[np.zeros((512, 512, 3), np.uint8)]),
               observation(images=[np.zeros((512, 512, 3), np.uint8)] * 3),
               observation(images=[np.zeros((512, 512, 3), np.float32)] * 2),
               observation(images=[np.zeros((256, 256, 3), np.uint8)] * 2),
               observation(state=[0.0] * 9), observation(instruction=3), observation(state=np.zeros((9, 1)))]
        for obs in bad:
            with self.assertRaises(ValueError):
                adapters.check_observation(obs)
        with self.assertRaises(ValueError):
            adapters.check_observation({k: v for k, v in observation().items() if k != "instruction"})

    def test_action_shape_is_eight(self):
        adapters.check_action(np.zeros(8))
        for bad in (np.zeros(7), np.zeros((1, 8)), [0.0] * 8):
            with self.assertRaises(ValueError):
                adapters.check_action(bad)

    def test_nonfinite_actions_are_reported_not_hidden(self):
        self.assertFalse(adapters.is_finite_action(np.array([np.nan] + [0.0] * 7)))
        self.assertTrue(adapters.is_finite_action(np.zeros(8)))

    def test_protocol_check(self):
        adapters.check_adapter(FakeAdapter())
        for bad in (object(), types.SimpleNamespace(name="x", reset=lambda: None)):
            with self.assertRaises(TypeError):
                adapters.check_adapter(bad)


class StateP2(unittest.TestCase):
    def test_is_the_first_nine_qpos_as_a_copy(self):
        data = types.SimpleNamespace(qpos=np.arange(30, dtype=float))
        state = adapters.state_p2(None, data, "can")
        self.assertEqual(state.tolist(), list(range(9)))
        state[0] = 99.0
        self.assertEqual(data.qpos[0], 0.0)
        self.assertEqual(state.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
