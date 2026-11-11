"""Statistics over 17 state values and the local model-input check (17 real values, 3 zeros of padding).

The extras come from a synthetic table (`test_forces.synthetic_table`, validation episodes made huge so a leak shows) patched
into `forces.table`, so none of this needs the real signals; the real signals' own tests are in `test_forces.py`.
"""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from test_contract import FRAMES, TASK_OF_EPISODE, frame, use_planned
from test_forces import synthetic_table, use_synthetic_table


def contract():
    return importlib.import_module('experiments.xvla_g1_rand.contract')


def check_rows(c):
    """The runner's check rows: starts 0, 421 and 422 of episode 0 and the last start of the last training episode."""
    return [0, 421, 422, (len(c.TRAIN)-1)*FRAMES+422]


def row(c, index, state=None, extras=None):
    """A training row at dataset position `index`, its state joined to the extras (or to `extras`)."""
    episode, position = divmod(index, FRAMES)
    state = torch.linspace(-1, 1, 9)+episode*1e-3 if state is None else state
    joined = c.joined_state(state, episode, position)
    if extras is not None:
        joined[9:] = torch.as_tensor(extras, dtype=torch.float32)
    return {**frame(episode), 'observation.state': joined}


_RANDOM = {}


def synthetic_rows(c, width):
    if 'values' not in _RANDOM:
        generator = np.random.default_rng(0)
        _RANDOM['values'] = {w: generator.normal(size=(c.TRAINING_ROWS, w)).astype(np.float32) for w in (8, 9, 17)}
    values = _RANDOM['values']

    class Rows:
        def __getitem__(self, i):
            if i >= c.TRAINING_ROWS:
                raise AssertionError('validation accessed for stats')
            return dict(episode_index=i//FRAMES, frame_index=i % FRAMES, task_index=TASK_OF_EPISODE[i//FRAMES],
                        action=torch.from_numpy(values[8][i]),
                        **{'observation.state': torch.from_numpy(values[width][i])})
    return Rows()


class ModelInputCheckTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        if not (self.c.CACHE/'tokenizer').exists():
            self.skipTest('base bundle not mounted')
        use_planned(self, self.c)
        use_synthetic_table(self)
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        self.rows = check_rows(self.c)
        self.config = self.c.make_config('cpu')
        self.policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(self.policy)
        self.policy.config, self.policy.model = self.config, SimpleNamespace(dim_proprio=20, dim_action=20)
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)}}
        self.pre, _ = make_xvla_pre_post_processors(self.config, stats)

    def batch(self, rows, **kwargs):
        raw = self.c.collate_rows([row(self.c, i, **kwargs) for i in rows])
        return raw, self.pre(raw)

    def test_state_normalisation_stays_identity_and_the_state_feature_has_17_values(self):
        mapping = self.config.normalization_mapping
        self.assertEqual(mapping['STATE'].value, 'IDENTITY')
        self.assertEqual(mapping['ACTION'].value, 'MEAN_STD')
        self.assertEqual(self.config.input_features['observation.state'].shape, (17,))
        self.assertTrue(self.config.use_proprio)
        self.assertEqual(self.config.max_state_dim, 20)

    def test_the_check_passes_on_a_real_batch_and_reports_17_real_values(self):
        raw, batch = self.batch(self.rows)
        evidence = self.c.check_model_inputs(self.policy, batch, raw['observation.state'], self.rows)
        self.assertEqual(evidence['proprio_shape'], [4, 20])
        self.assertEqual((evidence['real_state_dim'], evidence['extra_state_dim'], evidence['real_action_dim']), (17, 8, 8))
        self.assertTrue(evidence['state_identity_exact'])
        self.assertTrue(evidence['extras_standardised_exact'])
        self.assertEqual(evidence['domain'], 0)
        proprio = self.policy._build_model_inputs(batch)['proprio']
        self.assertEqual(torch.count_nonzero(proprio[:, 17:]).item(), 0)
        self.assertTrue(torch.equal(proprio[:, :17], raw['observation.state']))

    def test_the_check_fails_if_the_state_is_truncated_to_nine_values(self):
        raw, batch = self.batch(self.rows)
        batch9 = dict(batch)
        batch9['observation.state'] = batch['observation.state'][:, :9]
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch9, raw['observation.state'], self.rows)  # padded zeros, not extras
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch9, raw['observation.state'][:, :9], self.rows)

    def test_the_check_fails_if_the_extras_are_left_raw(self):
        raw_like = [synthetic_table()[i//FRAMES, i % FRAMES]*40.+7. for i in self.rows]  # unstandardised magnitudes
        for k, i in enumerate(self.rows):
            self.assertFalse(np.allclose(raw_like[k], self.c.forces.extra_state(i//FRAMES, i % FRAMES)))
        raw = self.c.collate_rows([row(self.c, i, extras=raw_like[k]) for k, i in enumerate(self.rows)])
        batch = self.pre(raw)
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], self.rows)

    def test_the_check_fails_if_the_extras_belong_to_other_frames(self):
        raw, batch = self.batch(self.rows)
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], [r+1 for r in self.rows])
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], self.rows[:3])

    def test_the_check_fails_if_the_batch_state_differs_from_the_raw_state(self):
        raw, batch = self.batch(self.rows)
        other = dict(batch)
        other['observation.state'] = batch['observation.state'].clone()
        other['observation.state'][0, 3] += 1e-3
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, other, raw['observation.state'], self.rows)

    def test_the_check_fails_if_the_raw_state_extras_differ_from_the_expected_ones(self):
        raw, batch = self.batch(self.rows)
        other = raw['observation.state'].clone()
        other[2, 11] += 1e-3  # the batch is untouched: only the raw state disagrees with the signals
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, other, self.rows)

    def test_the_check_fails_if_the_padding_is_not_zero(self):
        raw, batch = self.batch(self.rows)
        wide = dict(batch)
        wide['observation.state'] = torch.cat([batch['observation.state'], torch.ones(4, 1)], dim=1)  # an 18th value
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, wide, raw['observation.state'], self.rows)

    def test_the_check_fails_on_nonfinite_input(self):
        raw, batch = self.batch(self.rows)
        bad = dict(batch)
        bad['observation.state'] = batch['observation.state'].clone()
        bad['observation.state'][1, 12] = float('nan')
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, bad, raw['observation.state'], self.rows)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        use_planned(self, self.c)
        use_synthetic_table(self)

    def stats(self):
        if 'stats' not in _RANDOM:
            c = self.c
            _RANDOM['stats'] = c.selected_stats(synthetic_rows(c, 9), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)
        return _RANDOM['stats']

    def test_statistics_have_17_state_and_8_action_values(self):
        stats = self.stats()
        self.assertEqual({k: tuple(v.shape) for k, v in stats['observation.state'].items()},
                         {'mean': (17,), 'std': (17,), 'min': (17,), 'max': (17,)})
        self.assertEqual({k: tuple(v.shape) for k, v in stats['action'].items()},
                         {'mean': (8,), 'std': (8,), 'min': (8,), 'max': (8,)})

    def test_the_first_nine_statistics_are_of_the_dataset_state_and_the_action_of_the_action(self):
        stats, rows = self.stats(), _RANDOM['values']
        for name, reduce in (('mean', lambda x: x.mean(0)), ('std', lambda x: x.std(0)), ('min', lambda x: x.min(0)),
                             ('max', lambda x: x.max(0))):
            np.testing.assert_allclose(stats['observation.state'][name][:9].numpy(), reduce(rows[9].astype(np.float64)), atol=2e-5)
            np.testing.assert_allclose(stats['action'][name].numpy(), reduce(rows[8].astype(np.float64)), atol=2e-5)

    def test_the_extra_statistics_are_of_the_standardised_training_rows(self):
        c = self.c
        stats = self.stats()['observation.state']
        table = synthetic_table()[:len(c.TRAIN)].reshape(-1, 8)  # the 1,220,400 training rows, never the validation episodes
        self.assertEqual(len(table), c.TRAINING_ROWS)
        exact = table.astype(np.float64)
        np.testing.assert_allclose(stats['mean'][9:].numpy(), exact.mean(0), atol=2e-5)
        np.testing.assert_allclose(stats['std'][9:].numpy(), exact.std(0), atol=2e-5)
        np.testing.assert_array_equal(stats['min'][9:].numpy(), table.min(0))
        np.testing.assert_array_equal(stats['max'][9:].numpy(), table.max(0))
        self.assertLess(float(stats['max'][9:].max()), 10.)  # the 1e3-scaled validation rows did not get in

    def test_statistics_refuse_a_row_that_is_not_at_its_sealed_position(self):
        c = self.c
        rows = synthetic_rows(c, 9)

        class Shifted:
            def __getitem__(self, i):
                value = rows[i]
                return {**value, 'frame_index': (value['frame_index']+1) % FRAMES} if i == 700 else value

        class WrongTask:
            def __getitem__(self, i):
                value = rows[i]
                return {**value, 'task_index': value['task_index']+1} if i == 1000 else value

        for bad in (Shifted(), WrongTask()):
            with self.assertRaises(ValueError):
                c.selected_stats(bad, list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)

    def test_statistics_refuse_a_state_that_is_not_the_nine_dataset_values(self):
        c = self.c
        with self.assertRaises(ValueError):
            c.selected_stats(synthetic_rows(c, 17), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)


if __name__ == '__main__':
    unittest.main()
