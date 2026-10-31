"""Statistics over 17 state values and the local model-input check (17 real values, 3 zeros of padding)."""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from test_contract import FRAMES, TASK_OF_EPISODE, frame
from test_forces import mounted

ROWS = [0, 421, 422, 203_039]  # the runner's check rows: first and last starts of episodes 0 and 449


def contract():
    return importlib.import_module('experiments.xvla_p2_force.contract')


def row(c, index, state=None, extras=None):
    """A training row at dataset position `index`, its state joined to the sealed extras (or to `extras`)."""
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
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()
        if not (self.c.CACHE/'tokenizer').exists():
            self.skipTest('base bundle not mounted')
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
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
        raw, batch = self.batch(ROWS)
        evidence = self.c.check_model_inputs(self.policy, batch, raw['observation.state'], ROWS)
        self.assertEqual(evidence['proprio_shape'], [4, 20])
        self.assertEqual((evidence['real_state_dim'], evidence['extra_state_dim'], evidence['real_action_dim']), (17, 8, 8))
        self.assertTrue(evidence['state_identity_exact'])
        self.assertTrue(evidence['extras_standardised_exact'])
        self.assertEqual(evidence['domain'], 0)
        proprio = self.policy._build_model_inputs(batch)['proprio']
        self.assertEqual(torch.count_nonzero(proprio[:, 17:]).item(), 0)
        self.assertTrue(torch.equal(proprio[:, :17], raw['observation.state']))

    def test_the_check_fails_if_the_state_is_truncated_to_nine_values(self):
        raw, batch = self.batch(ROWS)
        batch9 = dict(batch)
        batch9['observation.state'] = batch['observation.state'][:, :9]
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch9, raw['observation.state'], ROWS)  # padded zeros, not extras
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch9, raw['observation.state'][:, :9], ROWS)

    def test_the_check_fails_if_the_extras_are_left_raw(self):
        raw_values = self.c.forces.load_raw(verify=False)['values']
        rows = [raw_values[i//FRAMES, i % FRAMES] for i in ROWS]
        for k, i in enumerate(ROWS):
            self.assertFalse(np.allclose(rows[k], self.c.forces.extra_state(i//FRAMES, i % FRAMES)))
        raw = self.c.collate_rows([row(self.c, i, extras=rows[k]) for k, i in enumerate(ROWS)])
        batch = self.pre(raw)
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], ROWS)

    def test_the_check_fails_if_the_extras_belong_to_other_frames(self):
        raw, batch = self.batch(ROWS)
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], [r+1 for r in ROWS])
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, raw['observation.state'], ROWS[:3])

    def test_the_check_fails_if_the_batch_state_differs_from_the_raw_state(self):
        raw, batch = self.batch(ROWS)
        other = dict(batch)
        other['observation.state'] = batch['observation.state'].clone()
        other['observation.state'][0, 3] += 1e-3
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, other, raw['observation.state'], ROWS)

    def test_the_check_fails_if_the_raw_state_extras_differ_from_the_expected_ones(self):
        raw, batch = self.batch(ROWS)
        other = raw['observation.state'].clone()
        other[2, 11] += 1e-3  # the batch is untouched: only the raw state disagrees with the sidecar
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, batch, other, ROWS)

    def test_the_check_fails_if_the_padding_is_not_zero(self):
        raw, batch = self.batch(ROWS)
        wide = dict(batch)
        wide['observation.state'] = torch.cat([batch['observation.state'], torch.ones(4, 1)], dim=1)  # an 18th value
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, wide, raw['observation.state'], ROWS)

    def test_the_check_fails_on_nonfinite_input(self):
        raw, batch = self.batch(ROWS)
        bad = dict(batch)
        bad['observation.state'] = batch['observation.state'].clone()
        bad['observation.state'][1, 12] = float('nan')
        with self.assertRaises(ValueError):
            self.c.check_model_inputs(self.policy, bad, raw['observation.state'], ROWS)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()

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

    def test_the_extra_statistics_are_of_the_standardised_training_rows(self):
        c = self.c
        stats = self.stats()['observation.state']
        table = c.forces.table()[:480].reshape(-1, 8)  # the 216,960 training rows, never the validation episodes
        self.assertEqual(len(table), c.TRAINING_ROWS)
        exact = table.astype(np.float64)
        np.testing.assert_allclose(stats['mean'][9:].numpy(), exact.mean(0), atol=2e-5)
        np.testing.assert_allclose(stats['std'][9:].numpy(), exact.std(0), atol=2e-5)
        np.testing.assert_array_equal(stats['min'][9:].numpy(), table.min(0))
        np.testing.assert_array_equal(stats['max'][9:].numpy(), table.max(0))
        np.testing.assert_allclose(stats['mean'][9:].numpy(), 0., atol=1e-4)
        np.testing.assert_allclose(stats['std'][9:].numpy(), 1., atol=1e-4)

    def test_statistics_refuse_a_row_that_is_not_at_its_sealed_position(self):
        c = self.c
        rows = synthetic_rows(c, 9)

        class Shifted:
            def __getitem__(self, i):
                value = rows[i]
                return {**value, 'frame_index': (value['frame_index']+1) % FRAMES} if i == 700 else value

        with self.assertRaises(ValueError):
            c.selected_stats(Shifted(), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)

    def test_statistics_refuse_a_state_that_is_not_the_nine_dataset_values(self):
        c = self.c
        with self.assertRaises(ValueError):
            c.selected_stats(synthetic_rows(c, 17), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)


class ClippingRateTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.f = importlib.import_module('experiments.xvla_p2_force.forces')

    def test_clipping_rates_per_dimension_for_training_and_validation_rows(self):
        f = self.f
        raw = f.load_raw(verify=False)
        k = f.constants()
        rates = f.clipping_rates()
        self.assertEqual(set(rates), {'train', 'validation'})
        for split, rows in (('train', raw['values'][raw['train']]), ('validation', raw['values'][~raw['train']])):
            rows = rows.reshape(-1, 8)
            expected = ((rows < k['low']).mean(0)+(rows > k['high']).mean(0)).tolist()
            self.assertEqual(rates[split]['fraction_clipped'], expected)
            self.assertEqual(rates[split]['rows'], len(rows))
        # The bounds are the training rows' 0.1 and 99.9 percentiles, so about 0.2% of them are clipped.
        self.assertTrue(all(0. <= x <= 0.0025 for x in rates['train']['fraction_clipped']))


if __name__ == '__main__':
    unittest.main()
