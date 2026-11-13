"""`XVLAG1Adapter` with stub model parts and G1's state builder: the interface, the 17-value state, which constants it uses.

Loading a real checkpoint needs the GPU and is not done here. Robosuite image, CPU.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.vla_inputs_eval import adapters


def images():
    return [np.zeros((512, 512, 3), np.uint8), np.full((512, 512, 3), 255, np.uint8)]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        from experiments.vla_inputs_arm_xvla_g1 import adapter
        from experiments.xvla_g1_rand import contract
        self.contract, self.adapter_module = contract, adapter
        self.instruction = contract.INSTRUCTIONS[0]

    def build(self, cls=None, **kwargs):
        seen = {'reset': 0}

        class Policy:
            config = SimpleNamespace()

            def reset(self):
                seen['reset'] += 1

            def select_action(self, processed):
                seen['processed'] = processed
                return torch.arange(8, dtype=torch.float32).reshape(1, 8)

        def pre(observation):
            seen['pre'] = observation
            return observation

        with patch('experiments.vla_inputs_arm_xvla.adapter.load_policy', return_value=Policy()), \
                patch('lerobot.policies.factory.make_pre_post_processors', return_value=(pre, lambda action: action)):
            adapter = (cls or self.adapter_module.XVLAG1Adapter)(Path('/nonexistent/checkpoint'), **kwargs)
        return adapter, seen

    def obs(self, state):
        return dict(images=images(), state=state, instruction=self.instruction)

    def test_it_implements_the_evaluator_interface_and_defaults_to_g1s_state_builder(self):
        adapter, seen = self.build()
        adapters.check_adapter(adapter)
        self.assertEqual(adapter.name, 'xvla_g1')
        self.assertIs(adapter.state_builder, self.adapter_module.state_g1)
        adapter.reset()
        self.assertEqual(seen['reset'], 1)

    def test_act_hands_the_processors_a_float32_17_value_state_and_both_images_in_order(self):
        adapter, seen = self.build()
        state = np.linspace(-1., 1., 17)
        action = adapter.act(self.obs(state))
        np.testing.assert_array_equal(action, np.arange(8, dtype=np.float32))
        given = seen['pre']
        self.assertEqual((given['observation.state'].dtype, tuple(given['observation.state'].shape)), (torch.float32, (17,)))
        np.testing.assert_array_equal(given['observation.state'].numpy(), state.astype(np.float32))
        self.assertEqual(given['task'], self.instruction)
        keys = [k for k in given if k.startswith('observation.images')]
        self.assertEqual(keys, list(self.contract.CAMERAS))
        self.assertEqual(given[keys[0]].max().item(), 0.)
        self.assertEqual(given[keys[1]].min().item(), 1.)

    def test_a_state_of_the_wrong_width_or_with_nonfinite_values_is_refused(self):
        adapter, _ = self.build()
        for state in (np.zeros(9), np.zeros(16), np.zeros(18), np.where(np.arange(17) == 12, np.nan, 0.)):
            with self.subTest(width=len(state)), self.assertRaises(ValueError):
                adapter.act(self.obs(state))

    def test_the_a0_adapter_cannot_take_17_values_which_is_why_act_is_overridden(self):
        from experiments.vla_inputs_arm_xvla.adapter import XVLAAdapter
        adapter, _ = self.build(cls=XVLAAdapter)
        with self.assertRaises(ValueError):
            adapter.act(self.obs(np.zeros(17)))

    def test_held_out_instructions_are_accepted_by_the_evaluator_and_others_are_not(self):
        from experiments.props_p2_data.plan import instruction
        adapter, _ = self.build()
        held_out = instruction('milk', 'blue')
        self.assertNotIn(held_out, self.contract.INSTRUCTIONS)
        adapter.act(dict(images=images(), state=np.zeros(17), instruction=held_out))
        with self.assertRaises(ValueError):
            adapter.act(dict(images=images(), state=np.zeros(17), instruction='put the cube on the table'))

    def test_the_shared_instruction_global_is_restored_after_each_call(self):
        from experiments.xvla_single_task import contract as single_task
        adapter, _ = self.build()
        before = single_task.TASK
        adapter.act(self.obs(np.zeros(17)))
        self.assertEqual(single_task.TASK, before)
        with self.assertRaises(ValueError):
            adapter.act(self.obs(np.zeros(9)))
        self.assertEqual(single_task.TASK, before)

    def test_a_state_builder_can_be_injected(self):
        builder = lambda model, data, target: np.zeros(17)
        adapter, _ = self.build(state_builder=builder)
        self.assertIs(adapter.state_builder, builder)


class StateBuilderTests(unittest.TestCase):
    """`state_g1` without a simulator: the signal functions are replaced, so what is tested is the assembly and the constants."""

    RAW = np.array([30., -12., 250., 3.1, -2.2, 0.9, 4., 7.5])  # wrench (6) then finger forces (2); some values beyond G1's clip range

    def state(self, raw=None):
        from experiments.vla_inputs_arm_xvla_g1 import adapter
        from experiments.vla_inputs_signals import signals
        raw = self.RAW if raw is None else raw
        data = SimpleNamespace(qpos=np.arange(44, dtype=np.float64) / 10)
        self.qpos_before = data.qpos.copy()
        with patch.object(signals, 'wrist_wrench', return_value=raw[:6]), patch.object(signals, 'finger_force', return_value=raw[6:]):
            return adapter.state_g1(None, data, 'milk'), data

    def test_the_state_is_nine_joint_values_then_the_extras_standardised_with_g1s_constants(self):
        from experiments.xvla_g1_rand import forces
        state, data = self.state()
        self.assertEqual((state.shape, state.dtype), ((17,), np.float64))
        np.testing.assert_array_equal(state[:9], data.qpos[:9])
        np.testing.assert_array_equal(state[9:].astype(np.float32), forces.standardise(self.RAW, forces.constants()))

    def test_it_does_not_use_a2s_constants(self):
        from experiments.xvla_g1_rand import forces as g1
        from experiments.xvla_p2_force import forces as a2
        self.assertNotEqual(g1.constants_digest(g1.constants()), a2.constants_digest(a2.constants()))
        state, _ = self.state()
        self.assertNotEqual(state[9:].astype(np.float32).tobytes(), a2.standardise(self.RAW, a2.constants()).tobytes())

    def test_the_clip_is_applied_before_standardising(self):
        from experiments.xvla_g1_rand import forces
        k = forces.constants()
        huge, _ = self.state(np.full(8, 1e9))
        np.testing.assert_array_equal(huge[9:].astype(np.float32), ((k['high'] - k['mean']) / k['std']).astype(np.float32))

    def test_the_wrench_comes_before_the_finger_forces(self):
        from experiments.xvla_g1_rand import forces
        k = forces.constants()
        swapped = np.concatenate([self.RAW[6:], self.RAW[:6]])
        state, _ = self.state()
        other, _ = self.state(swapped)
        self.assertNotEqual(state[9:].tobytes(), other[9:].tobytes())
        np.testing.assert_array_equal(state[9:].astype(np.float32), forces.standardise(self.RAW, k))

    def test_the_state_is_a_copy_and_the_simulation_is_not_written(self):
        state, data = self.state()
        np.testing.assert_array_equal(data.qpos, self.qpos_before)
        before = data.qpos.copy()
        state[:9] = -99.
        np.testing.assert_array_equal(data.qpos, before)


if __name__ == '__main__':
    unittest.main()
