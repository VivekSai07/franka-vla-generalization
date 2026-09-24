"""Broad-workspace plan: F7 envelope, per-episode jitter, disjoint splits."""
import copy
import hashlib
import importlib
import unittest

import numpy as np


class PlanTests(unittest.TestCase):
    def setUp(self):
        try:
            self.p = importlib.import_module('experiments.b1_language_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing broad-workspace plan: {exc}')

    def test_counts_and_split_ids(self):
        p = self.p.make_plan()
        self.assertEqual(len(p['episodes']), 200)
        self.assertEqual(len(p['layouts']), 280)  # 160 train + 40 validation + 80 reserved
        self.assertEqual(p['train_ids'], list(range(160)))
        self.assertEqual(p['validation_ids'], list(range(160, 200)))
        self.assertEqual(len(p['evaluation_layout_ids']), 80)
        self.assertEqual(len(set(p['policy_seeds'])), 2)
        train = [e for e in p['episodes'] if e['split'] == 'train']
        val = [e for e in p['episodes'] if e['split'] == 'validation']
        self.assertEqual(len(train), 160); self.assertEqual(len(val), 40)
        # B1 uses 200 distinct geometries; the repeated base geometry is not collected.
        self.assertEqual([e for e in p['episodes'] if e['layout_id'] == 'base'], [])
        self.assertEqual(len({e['layout_id'] for e in p['episodes']}), 200)
        for e in p['episodes']:
            self.assertNotIn(e['layout_id'], p['evaluation_layout_ids'])

    def test_cells_are_balanced_in_every_split(self):
        from collections import Counter
        p = self.p.make_plan()
        for split, n in (('train', 160), ('validation', 40)):
            counts = Counter(e['cell'] for e in p['episodes'] if e['split'] == split)
            self.assertEqual(set(counts), set(self.p.CELL_IDS))
            self.assertEqual(set(counts.values()), {n // 4}, f'{split} unbalanced: {counts}')

    def test_neither_position_nor_instruction_alone_predicts_the_target(self):
        """The design property: each region is correct in exactly half the cells, and each
        instruction maps to region A in one cell and region B in the other."""
        p = self.p.make_plan()
        by_cell = {c: self.p.CELLS[c] for c in self.p.CELL_IDS}
        # region holding the pick cube, per cell
        region = {}
        for cell, spec in by_cell.items():
            familiar = spec['arrangement'] == 'familiar'
            region[cell] = ('A' if spec['pick'] == 'red' else 'B') if familiar else \
                           ('B' if spec['pick'] == 'red' else 'A')
        self.assertEqual(sorted(region.values()), ['A', 'A', 'B', 'B'])
        for instruction in self.p.INSTRUCTIONS:
            cells = [c for c, s in by_cell.items()
                     if self.p.instruction_for(s['pick'], s['place']) == instruction]
            self.assertEqual(sorted(region[c] for c in cells), ['A', 'B'])

    def test_task_index_is_sorted_and_order_independent(self):
        p = self.p.make_plan()
        self.assertEqual(self.p.INSTRUCTIONS, tuple(sorted(self.p.INSTRUCTIONS)))
        for e in p['episodes']:
            self.assertEqual(e['instruction'], self.p.instruction_for(e['pick'], e['place']))
            self.assertEqual(e['task_index'], self.p.INSTRUCTIONS.index(e['instruction']))
        self.assertEqual({e['task_index'] for e in p['episodes']}, {0, 1})

    def test_envelope_containment_and_no_overlap(self):
        p = self.p.make_plan()
        a_lo, a_hi = self.p.REGION_A_LOW, self.p.REGION_A_HIGH
        b_lo, b_hi = self.p.REGION_B_LOW, self.p.REGION_B_HIGH

        def in_a(xy): return a_lo[0] <= xy[0] <= a_hi[0] and a_lo[1] <= xy[1] <= a_hi[1]
        def in_b(xy): return b_lo[0] <= xy[0] <= b_hi[0] and b_lo[1] <= xy[1] <= b_hi[1]

        for x in p['layouts']:
            ra, rb = x['region_xy_m'][:2], x['region_xy_m'][2:]
            self.assertTrue(in_a(ra), f'region A {ra}')
            self.assertTrue(in_b(rb), f'region B {rb}')
            xy = x.get('cube_xy_m')
            if xy is None:
                continue
            red, green = xy[:2], xy[2:]
            # colour may occupy either region, but exactly one each
            self.assertTrue((in_a(red) and in_b(green)) or (in_b(red) and in_a(green)),
                           f'cubes must occupy one region each: {xy}')
        # Region A y in [-.45,-.20], region B in [.20,.45]; overlap impossible.
        self.assertLess(-.20, .20)

    def test_both_colours_occupy_both_regions_across_the_dataset(self):
        """Regression for the confound this dataset exists to remove: if red never appears in
        region B, position predicts the pick target and colour is untestable."""
        p = self.p.make_plan()
        by_id = {x['layout_id']: x for x in p['layouts']}
        red_regions, green_regions = set(), set()
        for e in p['episodes']:
            xy = by_id[e['layout_id']]['cube_xy_m']
            red_regions.add('A' if xy[1] < 0 else 'B')
            green_regions.add('A' if xy[3] < 0 else 'B')
        self.assertEqual(red_regions, {'A', 'B'})
        self.assertEqual(green_regions, {'A', 'B'})

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
                f'b1_language_data_v1:20260925:{identity}'.encode()).digest()[:4], 'big')
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
