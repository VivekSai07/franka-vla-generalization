"""`XVLAForceBoxAdapter` with stub model parts: the interface, the box it stores at the state builder and draws in `act`, and what
it hands the processors. Robosuite image, CPU; loading a real checkpoint needs the GPU and is not done here.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.vla_inputs_eval import adapters

FRONT_BOX = np.array([0.25, 0.125, 0.5, 0.375, 1.0])
GRIPPER_BOX = np.array([0.5, 0.5, 0.75, 0.75, 1.0])
BOX = np.stack([FRONT_BOX, GRIPPER_BOX])


def images(front=0, gripper=255):
    return [np.full((512, 512, 3), front, np.uint8), np.full((512, 512, 3), gripper, np.uint8)]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        from experiments.xvla_p2_force import eval_adapter as m2_adapter
        from experiments.xvla_p2_force_box import contract, eval_adapter, overlay
        self.m2_adapter, self.contract, self.eval_adapter, self.overlay = m2_adapter, contract, eval_adapter, overlay
        self.instruction = contract.INSTRUCTIONS[0]

    def build(self, cls=None, real_inner=False, **kwargs):
        seen = {'reset': 0, 'pre': []}
        if cls is None and not real_inner:
            kwargs.setdefault('state_builder', lambda model, data, target: np.linspace(-1., 1., 17))

        class Policy:
            config = SimpleNamespace()

            def reset(self):
                seen['reset'] += 1

            def select_action(self, processed):
                seen['processed'] = processed
                return torch.arange(8, dtype=torch.float32).reshape(1, 8)

        def pre(observation):
            seen['pre'].append(observation)
            return observation

        with patch('experiments.vla_inputs_arm_xvla.adapter.load_policy', return_value=Policy()), \
                patch('lerobot.policies.factory.make_pre_post_processors', return_value=(pre, lambda action: action)):
            adapter = (cls or self.eval_adapter.XVLAForceBoxAdapter)(Path('/nonexistent/checkpoint'), **kwargs)
        return adapter, seen

    def obs(self, state=None, **kwargs):
        return dict(images=images(**kwargs), state=np.linspace(-1., 1., 17) if state is None else state, instruction=self.instruction)

    def tick(self, adapter, box=BOX, state=None, **kwargs):
        """What M0's rollout does at one tick: the state builder, then act."""
        with patch.object(self.eval_adapter.signals, 'target_box', return_value=box.copy()):
            built = adapter.state_builder('model', 'data', 'milk')
        return built, adapter.act(self.obs(state=built if state is None else state, **kwargs))

    def test_it_implements_the_m0_interface_and_is_loadable_by_the_arm_identifier(self):
        adapter, seen = self.build()
        adapters.check_adapter(adapter)
        self.assertEqual(adapter.name, 'xvla_force_box')
        self.assertTrue(callable(adapter.state_builder))
        adapter.reset()
        self.assertEqual(seen['reset'], 1)
        self.assertTrue(adapters._IDENTIFIER.match('experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter'))

    def test_the_state_builder_returns_m2s_state_and_stores_this_ticks_live_box(self):
        state = np.linspace(-2., 2., 17)
        inner_calls = []

        def inner(model, data, target):
            inner_calls.append((model, data, target))
            return state

        adapter, _ = self.build(state_builder=inner)
        with patch.object(self.eval_adapter.signals, 'target_box', return_value=BOX.copy()) as live:
            built = adapter.state_builder('model', 'data', 'milk')
        np.testing.assert_array_equal(built, state)
        self.assertEqual(inner_calls, [('model', 'data', 'milk')])
        live.assert_called_once_with('model', 'data', 'milk')

    def test_by_default_the_inner_state_builder_is_m2s_state_force(self):
        self.assertIs(self.eval_adapter.state_force, self.m2_adapter.state_force)
        adapter, _ = self.build(real_inner=True)
        self.assertIs(adapter.inner_state_builder, self.m2_adapter.state_force)

    def test_act_draws_the_front_box_on_the_first_camera_and_the_gripper_box_on_the_second(self):
        adapter, seen = self.build()
        self.tick(adapter)
        given = seen['pre'][-1]
        keys = [k for k in given if k.startswith('observation.images')]
        self.assertEqual(keys, list(self.contract.CAMERAS))
        undrawn = [torch.from_numpy(i).permute(2, 0, 1).float()/255 for i in images()]
        for key, undrawn_frame, box in zip(keys, undrawn, (FRONT_BOX, GRIPPER_BOX)):
            self.assertTrue(torch.equal(given[key], self.overlay.draw(undrawn_frame, box)), key)
            band = torch.from_numpy(self.overlay.band(box))
            self.assertTrue(torch.equal(given[key][0][band], torch.ones(int(band.sum()))))
            self.assertTrue(torch.equal(given[key][1][band], torch.zeros(int(band.sum()))))
            self.assertTrue(torch.equal(given[key][:, ~band], undrawn_frame[:, ~band]))
        self.assertFalse(torch.equal(given[keys[0]], self.overlay.draw(undrawn[0], GRIPPER_BOX)))

    def test_the_band_is_drawn_after_the_division_by_255_so_it_is_exactly_the_training_colour(self):
        adapter, seen = self.build()
        self.tick(adapter, front=77, gripper=200)
        given = seen['pre'][-1]
        band = torch.from_numpy(self.overlay.band(FRONT_BOX))
        magenta = (given[self.contract.CAMERAS[0]][:, band] == torch.tensor(self.overlay.COLOUR).reshape(3, 1)).all()
        self.assertTrue(bool(magenta))
        self.assertTrue(torch.equal(given[self.contract.CAMERAS[0]][:, 0, 0], torch.full((3,), 77.0)/255))

    def test_a_box_that_is_not_in_frame_draws_nothing_and_the_call_equals_a2s(self):
        box_out = np.stack([np.zeros(5), np.zeros(5)])
        adapter, seen = self.build()
        state = np.linspace(-1., 1., 17)
        self.tick(adapter, box=box_out, state=state)
        a2, seen_a2 = self.build(cls=self.m2_adapter.XVLAForceAdapter)
        a2.act(self.obs(state=state))
        mine, theirs = seen['pre'][-1], seen_a2['pre'][-1]
        self.assertEqual(sorted(mine), sorted(theirs))
        for key in mine:
            same = torch.equal(mine[key], theirs[key]) if torch.is_tensor(mine[key]) else mine[key] == theirs[key]
            self.assertTrue(same, key)

    def test_act_hands_the_processors_a_float32_17_value_state(self):
        adapter, seen = self.build()
        state = np.linspace(-1., 1., 17)
        self.tick(adapter, state=state)
        given = seen['pre'][-1]
        self.assertEqual((given['observation.state'].dtype, tuple(given['observation.state'].shape)), (torch.float32, (17,)))
        np.testing.assert_array_equal(given['observation.state'].numpy(), state.astype(np.float32))
        self.assertEqual(given['task'], self.instruction)

    def test_act_without_a_box_from_this_tick_raises_and_a_box_is_used_once(self):
        adapter, _ = self.build()
        with self.assertRaises(ValueError):
            adapter.act(self.obs())  # no state builder call yet
        self.tick(adapter)
        with self.assertRaises(ValueError):
            adapter.act(self.obs())  # the box of the previous tick must never be drawn again
        self.tick(adapter)  # a fresh tick works

    def test_reset_discards_a_box_left_over_from_the_previous_trial(self):
        adapter, _ = self.build()
        with patch.object(self.eval_adapter.signals, 'target_box', return_value=BOX.copy()):
            adapter.state_builder('model', 'data', 'milk')
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.act(self.obs())

    def test_the_newest_box_wins_if_the_builder_runs_twice_before_act(self):
        adapter, seen = self.build()
        other = np.stack([np.array([0.5, 0.5, 0.625, 0.625, 1.0]), GRIPPER_BOX])
        with patch.object(self.eval_adapter.signals, 'target_box', side_effect=[BOX.copy(), other.copy()]):
            adapter.state_builder('m', 'd', 't')
            built = adapter.state_builder('m', 'd', 't')
        adapter.act(self.obs(state=built))
        band = torch.from_numpy(self.overlay.band(other[0]))
        self.assertTrue(bool((seen['pre'][-1][self.contract.CAMERAS[0]][:, band] == torch.tensor(self.overlay.COLOUR).reshape(3, 1)).all()))

    def test_a_state_of_the_wrong_width_or_with_nonfinite_values_is_refused(self):
        adapter, _ = self.build()
        for state in (np.zeros(9), np.zeros(16), np.zeros(18), np.where(np.arange(17) == 12, np.nan, 0.)):
            with self.subTest(width=len(state)):
                with patch.object(self.eval_adapter.signals, 'target_box', return_value=BOX.copy()):
                    adapter.state_builder('m', 'd', 't')
                with self.assertRaises(ValueError):
                    adapter.act(self.obs(state=state))

    def test_the_shared_instruction_global_is_restored_after_each_call(self):
        from experiments.xvla_single_task import contract as single_task
        adapter, _ = self.build()
        before = single_task.TASK
        self.tick(adapter)
        self.assertEqual(single_task.TASK, before)
        with self.assertRaises(ValueError):
            self.tick(adapter, state=np.zeros(9))
        self.assertEqual(single_task.TASK, before)

    def test_an_instruction_outside_the_approved_set_is_refused(self):
        adapter, _ = self.build()
        with patch.object(self.eval_adapter.signals, 'target_box', return_value=BOX.copy()):
            adapter.state_builder('m', 'd', 't')
        with self.assertRaises(ValueError):
            adapter.act(dict(images=images(), state=np.zeros(17), instruction='put the cube on the table'))

    def test_the_input_images_are_not_modified(self):
        adapter, _ = self.build()
        given = images()
        before = [i.copy() for i in given]
        with patch.object(self.eval_adapter.signals, 'target_box', return_value=BOX.copy()):
            built = adapter.state_builder('m', 'd', 't')
        adapter.act(dict(images=given, state=built, instruction=self.instruction))
        for a, b in zip(given, before):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
