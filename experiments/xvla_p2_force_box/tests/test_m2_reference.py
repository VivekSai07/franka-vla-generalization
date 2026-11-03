"""The fork's recipe, state and statistics are M2's, read from M2's contract and compared here, never retyped.

This file and the hygiene test are the only places in the package that import the fork source, apart from the forces module
and the lazily imported reference runner.
"""
import dataclasses
import importlib
import itertools
import unittest
from pathlib import Path

import torch

from test_boxes_join import mounted
from test_contract import TASK_OF_EPISODE
from test_inputs import synthetic_rows

PACKAGE = Path(__file__).resolve().parents[1]


class RecipeIsM2Tests(unittest.TestCase):
    def setUp(self):
        self.m2 = importlib.import_module('experiments.xvla_p2_force.contract')
        self.c = importlib.import_module('experiments.xvla_p2_force_box.contract')

    def test_the_recipe_is_m2s_apart_from_the_purpose_string(self):
        for preflight in (True, False):
            a, b = self.m2.recipe(preflight), self.c.recipe(preflight)
            self.assertEqual(a.pop('purpose'), 'xvla_p2_force_preflight' if preflight else 'xvla_p2_force')
            self.assertEqual(b.pop('purpose'), 'xvla_p2_force_box_preflight' if preflight else 'xvla_p2_force_box')
            self.assertEqual(a, b)

    def test_constants_and_schedules_are_m2s(self):
        m2, c = self.m2, self.c
        for name in ('TOTAL_UPDATES', 'CHECKPOINT_UPDATES', 'SNAPSHOT_EVERY', 'KEEP_SNAPSHOTS', 'SEED', 'TRAIN', 'VAL',
                     'FRAMES_PER_EPISODE', 'CHUNK_SIZE', 'TRAINING_ROWS', 'COMPLETE_STARTS', 'VALIDATION_ROWS',
                     'TOTAL_FRAMES', 'SEAL', 'SEAL_RECORD', 'VALIDATION_RECORD', 'INSTRUCTIONS', 'CAMERAS',
                     'REAL_STATE_DIM', 'JOINT_DIM', 'OBS', 'DATASET_OBS'):
            self.assertEqual(getattr(c, name), getattr(m2, name), name)
        for preflight in (True, False):
            self.assertEqual(c.checkpoint_updates(preflight), m2.checkpoint_updates(preflight))
            self.assertEqual(c.snapshot_updates(preflight), m2.snapshot_updates(preflight))
        for update in (1, 2, 500, 501, 63_450, c.TOTAL_UPDATES):
            self.assertEqual(c.learning_rate(update), m2.learning_rate(update), update)

    def test_the_sampler_draws_m2s_batches_in_m2s_order(self):
        c, m2 = self.c, self.m2
        starts = [i*452+j for i in c.TRAIN for j in range(423)]
        ours = list(itertools.islice(c.sample_batches(starts), 300))
        theirs = list(itertools.islice(m2.sample_batches(starts), 300))
        self.assertEqual(ours, theirs)

    def test_the_output_directories_cannot_collide_with_m2s(self):
        m2, c = self.m2, self.c
        for name in ('OUTPUT', 'SCRATCH', 'PROOF', 'EQUIVALENCE', 'ZERO_EQUIVALENCE', 'ADMISSION'):
            self.assertNotEqual(getattr(c, name), getattr(m2, name), name)
        self.assertEqual(c.OUTPUT.name, 'xvla_p2_force_box_v1')

    def test_forces_is_m2s_own_module_not_a_copy(self):
        self.assertIs(self.c.forces, self.m2.forces)
        self.assertEqual(self.c.forces.__name__, 'experiments.xvla_p2_force.forces')
        self.assertFalse((PACKAGE/'forces.py').exists())

    def test_the_joined_state_equals_m2s_byte_for_byte(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        m2, c = self.m2, self.c
        state = torch.linspace(-1, 1, 9)
        for episode, frame in ((0, 0), (18, 200), (271, 451), (539, 451)):
            self.assertEqual(c.joined_state(state, episode, frame).numpy().tobytes(),
                             m2.joined_state(state, episode, frame).numpy().tobytes(), (episode, frame))

    def test_the_17_value_statistics_equal_m2s_byte_for_byte(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        m2, c = self.m2, self.c
        rows = synthetic_rows(c, 9)
        selected = list(range(c.TRAINING_ROWS))
        ours = c.selected_stats(rows, selected, TASK_OF_EPISODE)
        theirs = m2.selected_stats(rows, selected, TASK_OF_EPISODE)
        for key in ('observation.state', 'action'):
            for name in ('mean', 'std', 'min', 'max'):
                self.assertEqual(ours[key][name].numpy().tobytes(), theirs[key][name].numpy().tobytes(), (key, name))

    def test_the_model_config_is_m2s(self):
        m2, c = self.m2, self.c
        if not (c.CACHE/'model/config.json').exists():
            self.skipTest('base bundle not mounted')
        self.assertEqual(dataclasses.asdict(m2.make_config('cpu')), dataclasses.asdict(c.make_config('cpu')))


if __name__ == '__main__':
    unittest.main()
