"""Independent budget, marginal coverage and immutable schedule tests."""
import copy
import hashlib
import importlib
import unittest

import numpy as np


class PlanTests(unittest.TestCase):
    def setUp(self):
        try:
            self.p = importlib.import_module('experiments.local_varied_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing local-varied plan: {exc}')

    def test_schedule_and_marginal_extremes(self):
        p = self.p.make_plan()
        self.assertEqual(len(p['layouts']), 77)
        self.assertEqual(len(p['episodes']), 60)
        self.assertEqual(p['train_ids'], list(range(48)))
        self.assertEqual(p['validation_ids'], list(range(48, 60)))
        self.assertEqual(p['preparation_ids'], list(range(12)))
        ids = [e['layout_id'] for e in p['episodes']]
        self.assertEqual(ids[:4], ['center'] * 4)
        self.assertEqual(ids[12:16], ['center'] * 4)
        self.assertEqual(ids.count('center'), 8)
        candidates = [x for x in p['layouts'] if x['split'] == 'train']
        selected = []
        for axis in range(4):
            for sign in (1, -1):
                ordered = sorted(candidates, key=lambda x: (sign*x['offsets_m'][axis], x['candidate_index']))
                selected.append(next(x['layout_id'] for x in ordered if x['layout_id'] not in selected))
        self.assertEqual(ids[4:12], selected)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(ids[16:48], [x['layout_id'] for x in candidates if x['layout_id'] not in selected])
        self.assertFalse(set(ids) & set(p['evaluation_layout_ids']))
        self.assertEqual(len(p['evaluation_layout_ids']), 24)
        self.assertTrue(all(e['split'] == 'validation' and e['layout_id'] != 'center' for e in p['episodes'][48:]))

    def test_independent_lhs_streams_bins_and_resolved_sha_seeds(self):
        p = self.p.make_plan()
        all_offsets = []
        for split, n in (('train', 40), ('validation', 12), ('evaluation', 24)):
            a = np.asarray([x['offsets_m'] for x in p['layouts'] if x['split'] == split])
            self.assertEqual(a.shape, (n, 4))
            self.assertTrue(np.all(np.abs(a) <= .010))
            for axis in range(4):
                bins = np.floor((a[:, axis]+.010)/.020*n).astype(int)
                self.assertEqual(sorted(bins.tolist()), list(range(n)))
                for kind in ('permutation', 'jitter'):
                    identity = f'layout:{split}:{axis}:{kind}'
                    expected = int.from_bytes(hashlib.sha256(f'local_varied_data_v1:20260919:{identity}'.encode()).digest()[:4], 'big')
                    self.assertEqual(p['rng_seeds'][identity], expected)
            all_offsets.extend(map(tuple, a))
        self.assertEqual(len(set(all_offsets)), 76)
        seeds = [e['seed'] for e in p['episodes']]
        self.assertEqual(len(set(seeds)), 60)
        self.assertEqual(len(set(p['policy_seeds'])), 3)
        self.assertFalse(set(p['policy_seeds']) & (set(seeds) | set(range(20260918, 20260938))))
        for i, seed in enumerate(p['policy_seeds']):
            self.assertEqual(seed, int.from_bytes(hashlib.sha256(f'local_varied_data_v1:20260919:policy:{i}'.encode()).digest()[:4], 'big'))
        self.assertEqual(self.p.make_plan(), p)

    def test_mutations_nonfinite_and_type_spoofs_are_rejected(self):
        original = self.p.make_plan()
        self.p.validate_plan(original)
        for key, value in (('expert_attempts', 24), ('automatic_training', 0), ('master_seed', 20260919.), ('policy_seeds', [1, 2, 3])):
            p = copy.deepcopy(original); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.p.validate_plan(p)
        for value in (float('nan'), float('inf'), .011, False):
            p = copy.deepcopy(original); p['layouts'][1]['offsets_m'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.p.validate_plan(p)
        p = copy.deepcopy(original); p['source']['episode_index'] = 6
        with self.assertRaises(ValueError): self.p.validate_plan(p)


if __name__ == '__main__': unittest.main()