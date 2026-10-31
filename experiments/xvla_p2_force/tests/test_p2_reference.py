"""The fork's recipe is P2's, read from P2's contract and compared here, never retyped.

This file and the hygiene test are the only places in the package that import the fork source.
"""
import dataclasses
import importlib
import itertools
import unittest

from test_contract import TASK_OF_EPISODE
from test_forces import mounted
from test_inputs import synthetic_rows


class RecipeIsP2Tests(unittest.TestCase):
    def setUp(self):
        self.p2 = importlib.import_module('experiments.xvla_props_p2.contract')
        self.c = importlib.import_module('experiments.xvla_p2_force.contract')

    def test_the_recipe_is_p2s_apart_from_the_purpose_string(self):
        for preflight in (True, False):
            a, b = self.p2.recipe(preflight), self.c.recipe(preflight)
            self.assertEqual(a.pop('purpose'), 'xvla_props_p2_preflight' if preflight else 'xvla_props_p2')
            self.assertEqual(b.pop('purpose'), 'xvla_p2_force_preflight' if preflight else 'xvla_p2_force')
            self.assertEqual(a, b)

    def test_constants_and_schedules_are_p2s(self):
        p2, c = self.p2, self.c
        for name in ('TOTAL_UPDATES', 'CHECKPOINT_UPDATES', 'SNAPSHOT_EVERY', 'KEEP_SNAPSHOTS', 'SEED', 'TRAIN', 'VAL',
                     'FRAMES_PER_EPISODE', 'CHUNK_SIZE', 'TRAINING_ROWS', 'COMPLETE_STARTS', 'VALIDATION_ROWS',
                     'TOTAL_FRAMES', 'SEAL', 'SEAL_RECORD', 'VALIDATION_RECORD', 'INSTRUCTIONS', 'CAMERAS'):
            self.assertEqual(getattr(c, name), getattr(p2, name), name)
        for preflight in (True, False):
            self.assertEqual(c.checkpoint_updates(preflight), p2.checkpoint_updates(preflight))
            self.assertEqual(c.snapshot_updates(preflight), p2.snapshot_updates(preflight))
        for update in (1, 250, 500, 501, 50_760, 101_520, 126_900):
            self.assertEqual(c.learning_rate(update), p2.learning_rate(update), update)

    def test_the_sampler_draws_p2s_batches_in_p2s_order(self):
        c, p2 = self.c, self.p2
        starts = [i*452+j for i in c.TRAIN for j in range(423)]
        ours = list(itertools.islice(c.sample_batches(starts), 300))
        theirs = list(itertools.islice(p2.sample_batches(starts), 300))
        self.assertEqual(ours, theirs)

    def test_the_output_directories_cannot_collide_with_p2s(self):
        p2, c = self.p2, self.c
        for name in ('OUTPUT', 'SCRATCH', 'PROOF', 'EQUIVALENCE', 'ADMISSION'):
            self.assertNotEqual(getattr(c, name), getattr(p2, name), name)
        self.assertEqual(c.OUTPUT.name, 'xvla_p2_force_v1')

    def test_the_first_nine_state_statistics_equal_p2s_byte_for_byte(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        p2, c = self.p2, self.c
        rows = synthetic_rows(c, 9)
        selected = list(range(c.TRAINING_ROWS))
        ours = c.selected_stats(rows, selected, TASK_OF_EPISODE)
        theirs = p2.selected_stats(rows, selected, TASK_OF_EPISODE)
        for name in ('mean', 'std', 'min', 'max'):
            self.assertEqual(ours['observation.state'][name][:9].numpy().tobytes(),
                             theirs['observation.state'][name].numpy().tobytes(), name)
            self.assertEqual(ours['action'][name].numpy().tobytes(), theirs['action'][name].numpy().tobytes(), name)

    def test_the_model_config_differs_from_p2s_only_in_the_state_width(self):
        p2, c = self.p2, self.c
        if not (c.CACHE/'model/config.json').exists():
            self.skipTest('base bundle not mounted')
        a, b = p2.make_config('cpu'), c.make_config('cpu')
        self.assertEqual(a.input_features['observation.state'].shape, (9,))
        self.assertEqual(b.input_features['observation.state'].shape, (17,))
        a_dict, b_dict = dataclasses.asdict(a), dataclasses.asdict(b)
        a_dict['input_features'].pop('observation.state')
        b_dict['input_features'].pop('observation.state')
        self.assertEqual(a_dict, b_dict)


if __name__ == '__main__':
    unittest.main()
