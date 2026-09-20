"""Broad-workspace plan: F7 envelope, per-episode jitter, disjoint splits."""
import copy
import hashlib
import importlib
import unittest

import numpy as np


class PlanTests(unittest.TestCase):
    def setUp(self):
        try:
            self.p = importlib.import_module('experiments.broad_workspace_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing broad-workspace plan: {exc}')

    def test_counts_and_split_ids(self):
        p = self.p.make_plan()
        self.assertEqual(len(p['episodes']), 200)
        self.assertEqual(len(p['layouts']), 264)
        self.assertEqual(p['train_ids'], list(range(160)))
        self.assertEqual(p['validation_ids'], list(range(160, 200)))
        self.assertEqual(len(p['evaluation_layout_ids']), 80)
        self.assertEqual(len(set(p['policy_seeds'])), 2)
        train = [e for e in p['episodes'] if e['split'] == 'train']
        val = [e for e in p['episodes'] if e['split'] == 'validation']
        self.assertEqual(len(train), 160); self.assertEqual(len(val), 40)
        base_train = [e for e in train if e['layout_id'] == 'base']
        lhs_train = [e for e in train if e['layout_id'] != 'base']
        self.assertEqual(len(base_train), 16); self.assertEqual(len(lhs_train), 144)
        for e in val: self.assertNotEqual(e['layout_id'], 'base')
        for e in p['episodes']:
            self.assertNotIn(e['layout_id'], p['evaluation_layout_ids'])

    def test_envelope_containment_and_no_overlap(self):
        p = self.p.make_plan()
        for x in p['layouts']:
            rx, ry, gx, gy = x['cube_xy_m']
            self.assertTrue(.20 <= rx <= .35 and -.45 <= ry <= -.20, f'red XY {rx},{ry}')
            self.assertTrue(.20 <= gx <= .35 and .20 <= gy <= .45, f'green XY {gx},{gy}')
        # Red rectangle Y in [-.45, -.20], green in [.20, .45]; overlap impossible.
        self.assertLess(-.20, .20)

    def test_per_episode_jitter_within_bounds(self):
        p = self.p.make_plan()
        for e in p['episodes']:
            self.assertEqual(len(e['home_pose_jitter_deg']), 7)
            self.assertTrue(all(type(v) is float and abs(v) <= 1.0 for v in e['home_pose_jitter_deg']))
            self.assertEqual(len(e['hover_z_jitter_m']), 2)
            self.assertTrue(all(type(v) is float and abs(v) <= .03 for v in e['hover_z_jitter_m']))
            self.assertEqual(set(e['hover_z_offset']), {'pre_grasp', 'pre_place'})
            for v in e['hover_z_offset'].values():
                self.assertTrue(type(v) is float and abs(v) <= .03)

    def test_seed_collision_and_derivation(self):
        p = self.p.make_plan()
        seeds = p['rng_seeds']
        self.assertEqual(len(set(seeds.values())), len(seeds))
        for identity, seed in seeds.items():
            expected = int.from_bytes(hashlib.sha256(
                f'broad_workspace_data_v1:20260920:{identity}'.encode()).digest()[:4], 'big')
            self.assertEqual(seed, expected)
        expert = [e['seed'] for e in p['episodes']]
        self.assertEqual(len(set(expert)), 200)
        self.assertFalse(set(p['policy_seeds']) & (set(expert) | set(range(20260918, 20260938))))

    def test_reproducibility_bit_identical(self):
        self.assertEqual(self.p.make_plan(), self.p.make_plan())

    def test_evaluation_holdout_frozen(self):
        p = self.p.make_plan()
        for layout_id in p['evaluation_layout_ids']:
            eh = p['evaluation_holdout'][layout_id]
            self.assertEqual(len(eh['home_pose_jitter_deg']), 7)
            self.assertTrue(all(type(v) is float and abs(v) <= 1.0 for v in eh['home_pose_jitter_deg']))
            self.assertEqual(len(eh['hover_z_jitter_m']), 2)
            self.assertEqual(set(eh['hover_z_offset']), {'pre_grasp', 'pre_place'})

    def test_tamper_rejection(self):
        original = self.p.make_plan(); self.p.validate_plan(original)
        for key, value in (('expert_attempts', 24), ('master_seed', 20260919),
                           ('policy_seeds', [1, 2, 3]),
                           ('automatic_training', 0)):
            plan = copy.deepcopy(original); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.p.validate_plan(plan)
        for value in (float('nan'), float('inf'), 1.5, False):
            plan = copy.deepcopy(original); plan['episodes'][0]['home_pose_jitter_deg'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.p.validate_plan(plan)
        plan = copy.deepcopy(original); plan['layouts'][0]['cube_xy_m'][0] = 0.10
        with self.assertRaises(ValueError):
            self.p.validate_plan(plan)

    def test_base_pose_never_in_layouts(self):
        p = self.p.make_plan()
        self.assertNotIn('base', [x['layout_id'] for x in p['layouts']])
        self.assertEqual(p['base_layout']['layout_id'], 'base')


if __name__ == '__main__':
    unittest.main()
