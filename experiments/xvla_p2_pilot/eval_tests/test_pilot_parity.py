"""Train/rollout parity of the pilots' 19-value state for every variant, on replayed sealed episodes. Robosuite image, CPU; no
rendering.

The sealed episodes are replayed exactly (S0's `replay.frames`, which raises on any difference from the sealed `qpos` and prop
poses), and at every frame each variant's live state builder runs at the same instant the M0 rollout calls it. Its output must
equal, byte for byte, the row the training loader builds for that frame (`contract.joined_state`).
"""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

EPISODES = (0, 18, 271)  # first, one with clipped force frames (the clip must be exercised), a middle one
FRAMES = 452
VARIANTS = ('control', 'force', 'box')


def modules():
    from experiments.vla_inputs_signals import replay
    return (replay, importlib.import_module('experiments.xvla_p2_pilot.contract'),
            importlib.import_module('experiments.xvla_p2_pilot.eval_adapter'))


def mounted():
    replay, c, adapter = modules()
    return all(path.exists() for path in (c.forces.SIDECAR/'seal.json', c.PARENT/'seal.json'))


def live_states(index, wanted=None):
    """{variant: {frame: (19,) state the rollout's builder returns}}, and the sealed episode, from one exact replay."""
    replay, c, adapter = modules()
    sealed = replay.load_episode(index)
    target, states = sealed['episode']['object'], {v: {} for v in VARIANTS}

    def on_frame(tick, model, data):
        if wanted is None or tick in wanted:
            for variant in VARIANTS:
                states[variant][tick] = adapter.LIVE[variant](model, data, target)

    replay.frames(index, on_frame)
    return states, sealed


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest('P2 parent or S0 sidecar not mounted')
        cls.replay, cls.c, cls.adapter = modules()
        cls.cache = {index: live_states(index) for index in EPISODES}

    def test_the_rollout_state_equals_the_training_row_byte_for_byte_at_every_frame_for_every_variant(self):
        c = self.c
        for index, (states, sealed) in self.cache.items():
            for variant in VARIANTS:
                builder = c.BUILDERS[variant]
                self.assertEqual(sorted(states[variant]), list(range(FRAMES)))
                for tick in range(FRAMES):
                    row = c.joined_state(torch.from_numpy(sealed['state'][tick]), index, tick, builder).numpy()
                    live = np.asarray(states[variant][tick], dtype=np.float32)
                    self.assertEqual(live.shape, (19,))
                    self.assertEqual(live.tobytes(), row.tobytes(), (variant, index, tick))

    def test_each_variant_has_its_own_extras_and_zero_padding(self):
        c = self.c
        for index, (states, sealed) in self.cache.items():
            for variant, width in (('control', 0), ('force', 8), ('box', 10)):
                live = np.stack([states[variant][t] for t in range(FRAMES)])
                self.assertEqual(live.dtype, np.float64)
                np.testing.assert_array_equal(live[:, :9].astype(np.float32), sealed['state'])
                self.assertEqual(np.count_nonzero(live[:, 9+width:]), 0, (variant, index))
                if width:
                    self.assertGreater(np.count_nonzero(live[:, 9:9+width]), 0, (variant, index))
                else:
                    self.assertEqual(np.count_nonzero(live[:, 9:]), 0)

    def test_the_force_state_is_m2s_17_values_then_two_zeros_and_the_box_is_the_sidecars_target_box(self):
        c, f = self.c, self.c.forces
        from experiments.xvla_p2_force import eval_adapter as m2_adapter
        raw = f.load_raw(verify=False)['values']
        k = f.constants()
        for index, (states, sealed) in self.cache.items():
            force = np.stack([states['force'][t] for t in range(FRAMES)])
            expected = ((np.clip(raw[index], k['low'], k['high'])-k['mean'])/k['std']).astype(np.float32)
            self.assertEqual(force[:, 9:17].astype(np.float32).tobytes(), expected.tobytes(), index)
            self.assertEqual(np.count_nonzero(force[:, 17:]), 0)
            with np.load(f.SIDECAR/'episodes'/f'episode_{index:03d}.npz', allow_pickle=False) as stored:
                box = stored['target_box'].reshape(FRAMES, 10)
            live = np.stack([states['box'][t] for t in range(FRAMES)])
            np.testing.assert_array_equal(live[:, 9:], box)
        self.assertIs(self.adapter.state_force_m2, m2_adapter.state_force)

    def test_some_frames_are_clipped_so_the_clip_is_part_of_what_is_proven(self):
        f = self.c.forces
        raw, k = f.load_raw(verify=False)['values'], f.constants()
        clipped = sum(int(((raw[i] < k['low']) | (raw[i] > k['high'])).any(axis=1).sum()) for i in EPISODES)
        self.assertGreater(clipped, 0)

    def test_building_any_state_does_not_write_the_simulation(self):
        replay, c, adapter = modules()
        index, before, after = EPISODES[1], {}, {}
        target = replay.load_episode(index)['episode']['object']
        names = ('qpos', 'qvel', 'qacc', 'ctrl', 'act', 'xpos', 'cfrc_int', 'qfrc_constraint', 'efc_force')

        def on_frame(tick, model, data):
            if tick in (0, 150, 300):
                before[tick] = {n: getattr(data, n).copy() for n in names}
                for variant in VARIANTS:
                    adapter.LIVE[variant](model, data, target)
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
        cls.replay, cls.c, cls.adapter = modules()
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

    def rollout_proprio(self, variant, tick):
        """What `XVLAPilotAdapter.act` hands the processors for this live state, then what the policy builds from it."""
        from experiments.xvla_props_p2_eval import protocol
        c = self.c
        instruction = c.planned_instructions()[self.index]
        frame = {'observation.state': torch.tensor(self.states[variant][tick], dtype=torch.float32), 'task': instruction,
                 **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
        with protocol.eval_instruction(instruction):
            batch = self.pre(c.observation(frame))
        return self.policy._build_model_inputs(batch)['proprio']

    def test_the_normalised_and_padded_proprio_is_identical_in_training_and_rollout_for_every_variant(self):
        c = self.c
        for variant in VARIANTS:
            builder = c.BUILDERS[variant]
            rows = [{'episode_index': torch.tensor(self.index), 'task': c.planned_instructions()[self.index],
                     'observation.state': c.joined_state(torch.from_numpy(self.sealed['state'][tick]), self.index, tick, builder),
                     **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS},
                     'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}
                    for tick in self.ticks]
            trained = self.policy._build_model_inputs(self.pre(c.collate_rows(rows)))['proprio']
            self.assertEqual(tuple(trained.shape), (4, 20))
            for k, tick in enumerate(self.ticks):
                live = self.rollout_proprio(variant, tick)
                self.assertEqual(live.numpy().tobytes(), trained[k:k+1].numpy().tobytes(), (variant, tick))
                self.assertEqual(torch.count_nonzero(live[:, 19:]).item(), 0)
                self.assertEqual(torch.count_nonzero(live[:, 9+builder.width:]).item(), 0)

    def test_control_reaches_the_model_as_p2s_nine_value_state_does(self):
        # The control arm of a pilot sees exactly what P2's own policy saw: joints, then zeros in the 11 padding slots.
        for tick in self.ticks:
            live = self.rollout_proprio('control', tick)
            self.assertEqual(torch.count_nonzero(live[:, 9:]).item(), 0)
            np.testing.assert_array_equal(live[0, :9].numpy(), self.sealed['state'][tick])

    def test_a_nine_or_seventeen_value_state_is_not_accepted_by_the_rollout_check(self):
        from experiments.xvla_props_p2_eval import protocol
        c = self.c
        for width in (9, 17):
            frame = {'observation.state': torch.zeros(width), 'task': c.planned_instructions()[self.index],
                     **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
            with self.subTest(width=width), protocol.eval_instruction(frame['task']), self.assertRaises(ValueError):
                c.observation(frame)


if __name__ == '__main__':
    unittest.main()
