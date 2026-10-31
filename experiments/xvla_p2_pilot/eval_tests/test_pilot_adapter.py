"""`XVLAPilotAdapter` with stub model parts: the variant it takes from the checkpoint, the state it validates, what it hands
the processors.

Loading a real checkpoint needs the GPU and is not done here. Robosuite image, CPU.
"""
import json
import tempfile
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
        from experiments.xvla_p2_pilot import contract, eval_adapter
        self.contract, self.eval_adapter = contract, eval_adapter
        self.instruction = contract.INSTRUCTIONS[0]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def checkpoint(self, trained='force', **marker):
        """A checkpoint directory holding only the marker the adapter reads, as trained with the variant `trained`."""
        path = Path(self.tmp.name)/f'ckpt_{len(list(Path(self.tmp.name).iterdir()))}'
        path.mkdir()
        loaded = self.contract.load_variant(trained)
        body = dict(status='complete', update=10_000, variant=trained, variant_sha256=loaded['sha256'])
        body.update(marker)
        (path/'checkpoint_result.json').write_text(json.dumps(body))
        return path

    def build(self, checkpoint=None, **kwargs):
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
            adapter = self.eval_adapter.XVLAPilotAdapter(checkpoint or self.checkpoint(), **kwargs)
        return adapter, seen

    def obs(self, state):
        return dict(images=images(), state=state, instruction=self.instruction)

    def test_it_implements_the_m0_interface_and_takes_its_variant_from_the_checkpoint_marker(self):
        for variant in ('control', 'force', 'box'):
            adapter, seen = self.build(self.checkpoint(variant))
            adapters.check_adapter(adapter)
            self.assertEqual((adapter.name, adapter.variant), ('xvla_pilot', variant))
            self.assertIs(adapter.state_builder, self.eval_adapter.LIVE[self.contract.BUILDERS[variant].name])
            adapter.reset()
            self.assertEqual(seen['reset'], 1)

    def test_every_training_builder_has_a_live_builder_and_no_other_exists(self):
        self.assertEqual(set(self.eval_adapter.LIVE), set(self.contract.BUILDERS))

    def test_a_checkpoint_of_an_unknown_changed_or_missing_variant_is_refused_before_anything_loads(self):
        for label, marker in {
            'unknown variant': dict(variant='nonesuch'),
            'variant file changed': dict(variant_sha256='0'*64),
            'no variant': dict(variant=None),
            'a path': dict(variant='../control'),
            'no hash': dict(variant_sha256=None),
        }.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.build(self.checkpoint('force', **marker))

    def test_a_checkpoint_without_a_marker_is_refused(self):
        empty = Path(self.tmp.name)/'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            self.build(empty)

    def test_act_hands_the_processors_a_float32_19_value_state_and_both_images_in_order(self):
        adapter, seen = self.build()
        state = np.linspace(-1., 1., 19)
        action = adapter.act(self.obs(state))
        np.testing.assert_array_equal(action, np.arange(8, dtype=np.float32))
        given = seen['pre']
        self.assertEqual((given['observation.state'].dtype, tuple(given['observation.state'].shape)), (torch.float32, (19,)))
        np.testing.assert_array_equal(given['observation.state'].numpy(), state.astype(np.float32))
        self.assertEqual(given['task'], self.instruction)
        keys = [k for k in given if k.startswith('observation.images')]
        self.assertEqual(keys, list(self.contract.CAMERAS))
        self.assertEqual(given[keys[0]].max().item(), 0.)
        self.assertEqual(given[keys[1]].min().item(), 1.)

    def test_a_state_of_the_wrong_width_or_with_nonfinite_values_is_refused(self):
        adapter, _ = self.build()
        for state in (np.zeros(9), np.zeros(17), np.zeros(18), np.zeros(20), np.where(np.arange(19) == 12, np.nan, 0.)):
            with self.subTest(width=len(state)), self.assertRaises(ValueError):
                adapter.act(self.obs(state))

    def test_the_a0_adapter_cannot_take_19_values_which_is_why_act_is_overridden(self):
        from experiments.vla_inputs_arm_xvla.adapter import XVLAAdapter
        with patch('experiments.vla_inputs_arm_xvla.adapter.load_policy', return_value=SimpleNamespace(config=SimpleNamespace())), \
                patch('lerobot.policies.factory.make_pre_post_processors', return_value=(lambda o: o, lambda a: a)):
            adapter = XVLAAdapter(Path('/nonexistent/checkpoint'))
        with self.assertRaises(ValueError):
            adapter.act(self.obs(np.zeros(19)))

    def test_held_out_instructions_are_accepted_by_the_evaluator_and_others_are_not(self):
        from experiments.props_p2_data.plan import instruction
        adapter, _ = self.build()
        held_out = instruction('milk', 'blue')
        self.assertNotIn(held_out, self.contract.INSTRUCTIONS)
        adapter.act(dict(images=images(), state=np.zeros(19), instruction=held_out))
        with self.assertRaises(ValueError):
            adapter.act(dict(images=images(), state=np.zeros(19), instruction='put the cube on the table'))

    def test_the_shared_instruction_global_is_restored_after_each_call(self):
        from experiments.xvla_single_task import contract as single_task
        adapter, _ = self.build()
        before = single_task.TASK
        adapter.act(self.obs(np.zeros(19)))
        self.assertEqual(single_task.TASK, before)
        with self.assertRaises(ValueError):
            adapter.act(self.obs(np.zeros(9)))
        self.assertEqual(single_task.TASK, before)

    def test_a_state_builder_table_can_be_injected(self):
        builder = lambda model, data, target: np.zeros(19)
        adapter, _ = self.build(state_builders={'force': builder})
        self.assertIs(adapter.state_builder, builder)

    def test_the_adapter_reads_no_training_runner(self):
        import sys
        self.assertNotIn('experiments.xvla_p2_pilot.runner', sys.modules)


if __name__ == '__main__':
    unittest.main()
