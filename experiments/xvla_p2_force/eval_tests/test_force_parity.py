"""Train/rollout parity of the 17-value state, on replayed sealed episodes. Robosuite image, CPU; no rendering.

The sealed episodes are replayed exactly (S0's `replay.frames`, which raises on any difference from the sealed `qpos` and
prop poses), and at every frame the rollout's state builder runs at the same instant the M0 rollout calls it. Its output
must equal, byte for byte, the row the training loader builds for that frame.
"""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

EPISODES = (0, 18, 271, 539)  # first, one with 45 clipped frames (the clip must be exercised), a middle one, the last
FRAMES = 452


def modules():
    from experiments.vla_inputs_signals import replay
    return (replay, importlib.import_module('experiments.xvla_p2_force.contract'),
            importlib.import_module('experiments.xvla_p2_force.eval_adapter'),
            importlib.import_module('experiments.xvla_p2_force.forces'))


def mounted():
    return all(path.exists() for path in (modules()[3].SIDECAR/'seal.json', modules()[1].PARENT/'seal.json'))


def live_states(index, wanted=None):
    """{frame: (17,) state the rollout's builder returns}, and the sealed episode, from one exact replay."""
    replay, c, adapter, f = modules()
    sealed = replay.load_episode(index)
    target, states = sealed['episode']['object'], {}

    def on_frame(tick, model, data):
        if wanted is None or tick in wanted:
            states[tick] = adapter.state_force(model, data, target)

    replay.frames(index, on_frame)
    return states, sealed


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest('P2 parent or S0 sidecar not mounted')
        cls.replay, cls.c, cls.adapter, cls.f = modules()
        cls.cache = {index: live_states(index) for index in EPISODES}

    def test_the_rollout_state_equals_the_training_row_byte_for_byte_at_every_frame(self):
        c = self.c
        for index, (states, sealed) in self.cache.items():
            self.assertEqual(sorted(states), list(range(FRAMES)))
            for tick in range(FRAMES):
                row = c.joined_state(torch.from_numpy(sealed['state'][tick]), index, tick).numpy()
                live = np.asarray(states[tick], dtype=np.float32)
                self.assertEqual(live.shape, (17,))
                self.assertEqual(live.tobytes(), row.tobytes(), (index, tick))

    def test_the_nine_joint_values_are_the_sealed_state_and_the_rest_come_from_the_sidecar(self):
        raw = self.f.load_raw(verify=False)['values']
        k = self.f.constants()
        for index, (states, sealed) in self.cache.items():
            live = np.stack([states[t] for t in range(FRAMES)])
            self.assertEqual(live.dtype, np.float64)
            np.testing.assert_array_equal(live[:, :9].astype(np.float32), sealed['state'])
            # An oracle that shares nothing with `forces.standardise`: the sealed raw rows, clipped, then standardised.
            expected = ((np.clip(raw[index], k['low'], k['high'])-k['mean'])/k['std']).astype(np.float32)
            self.assertEqual(live[:, 9:].astype(np.float32).tobytes(), expected.tobytes(), index)

    def test_the_live_signals_are_the_sealed_sidecar_values(self):
        """The builder's inputs, before the clip: wrench then finger force, equal to what S0 wrote."""
        replay, c, adapter, f = modules()
        from experiments.vla_inputs_signals import signals
        raw = f.load_raw(verify=False)['values']
        index = EPISODES[0]
        seen = {}

        def on_frame(tick, model, data):
            seen[tick] = np.concatenate([signals.wrist_wrench(model, data), signals.finger_force(model, data)])

        replay.frames(index, on_frame)
        np.testing.assert_array_equal(np.stack([seen[t] for t in range(FRAMES)]), raw[index])

    def test_some_frames_are_clipped_so_the_clip_is_part_of_what_is_proven(self):
        raw = self.f.load_raw(verify=False)['values']
        k = self.f.constants()
        clipped = sum(int(((raw[i] < k['low']) | (raw[i] > k['high'])).any(axis=1).sum()) for i in EPISODES)
        self.assertGreater(clipped, 0)

    def test_building_the_state_does_not_write_the_simulation(self):
        replay, c, adapter, f = modules()
        index, before, after = EPISODES[1], {}, {}
        target = replay.load_episode(index)['episode']['object']
        names = ('qpos', 'qvel', 'qacc', 'ctrl', 'act', 'xpos', 'cfrc_int', 'qfrc_constraint', 'efc_force')

        def on_frame(tick, model, data):
            if tick in (0, 150, 300):
                before[tick] = {n: getattr(data, n).copy() for n in names}
                adapter.state_force(model, data, target)
                after[tick] = {n: getattr(data, n).copy() for n in names}

        replay.frames(index, on_frame)
        self.assertEqual(sorted(before), [0, 150, 300])
        for tick in before:
            for name in names:
                np.testing.assert_array_equal(before[tick][name], after[tick][name], (tick, name))


class PolicyInputParityTests(unittest.TestCase):
    """The normalised, padded tensor the policy receives, not only the raw concatenation."""

    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest('P2 parent or S0 sidecar not mounted')
        cls.replay, cls.c, cls.adapter, cls.f = modules()
        if not (cls.c.CACHE/'tokenizer').exists():
            raise unittest.SkipTest('base bundle not mounted')
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        config = cls.c.make_config('cpu')
        cls.policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(cls.policy)
        cls.policy.config, cls.policy.model = config, SimpleNamespace(dim_proprio=20, dim_action=20)
        cls.pre, _ = make_xvla_pre_post_processors(config, {'action': {'mean': torch.zeros(8), 'std': torch.ones(8)}})
        cls.index, cls.ticks = 0, (0, 120, 260, 451)
        cls.states, cls.sealed = live_states(cls.index, set(cls.ticks))

    def rollout_proprio(self, tick):
        """What `XVLAForceAdapter.act` hands the processors for this live state, then what the policy builds from it."""
        from experiments.xvla_props_p2_eval import protocol
        c = self.c
        instruction = c.planned_instructions()[self.index]
        frame = {'observation.state': torch.tensor(self.states[tick], dtype=torch.float32), 'task': instruction,
                 **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
        with protocol.eval_instruction(instruction):
            batch = self.pre(c.observation(frame))
        return self.policy._build_model_inputs(batch)['proprio']

    def test_the_normalised_and_padded_proprio_is_identical_in_training_and_rollout(self):
        c = self.c
        rows = []
        for tick in self.ticks:
            rows.append({'episode_index': torch.tensor(self.index), 'task': c.planned_instructions()[self.index],
                         'observation.state': c.joined_state(torch.from_numpy(self.sealed['state'][tick]), self.index, tick),
                         **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS},
                         'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)})
        trained = self.policy._build_model_inputs(self.pre(c.collate_rows(rows)))['proprio']
        self.assertEqual(tuple(trained.shape), (4, 20))
        for k, tick in enumerate(self.ticks):
            live = self.rollout_proprio(tick)
            self.assertEqual(live.numpy().tobytes(), trained[k:k+1].numpy().tobytes(), tick)
            self.assertEqual(torch.count_nonzero(live[:, 17:]).item(), 0)
            self.assertGreater(torch.count_nonzero(live[:, 9:17]).item(), 0)

    def test_a_nine_value_state_is_not_accepted_by_the_rollout_check(self):
        c = self.c
        frame = {'observation.state': torch.zeros(9), 'task': c.planned_instructions()[self.index],
                 **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
        from experiments.xvla_props_p2_eval import protocol
        with protocol.eval_instruction(frame['task']), self.assertRaises(ValueError):
            c.observation(frame)


if __name__ == '__main__':
    unittest.main()
