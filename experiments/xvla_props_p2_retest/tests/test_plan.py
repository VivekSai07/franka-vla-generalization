import collections
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.props_p2_data import plan as p2
from experiments.xvla_props_p2_retest import plan as rt

ROOT = Path(__file__).resolve().parents[4]
P2_PLAN_CANDIDATES = (Path(os.environ.get('P2_PLAN', '/nonexistent')), ROOT/'outputs/props_p2_data/props_p2_20261013/plan.json',
                      Path('/bundle_prov/plan.json'))


def sealed_p2_plan():
    for path in P2_PLAN_CANDIDATES:
        if path.is_file():
            return json.loads(path.read_text())
    raise AssertionError('the sealed P2 plan.json is required (mount outputs/props_p2_data or the bundle provenance); '
                         'the no-overlap test must not be skipped')


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = rt.make_plan()

    def test_counts_and_ids(self):
        h = self.plan['evaluation_holdout']
        self.assertEqual(len(h), 480)
        self.assertEqual(len(self.plan['evaluation_report_ids']), 480)
        self.assertEqual(set(self.plan['evaluation_report_ids']), set(h))
        self.assertEqual(len(self.plan['pairs']), 240)
        self.assertEqual(sorted(collections.Counter(v['object'] for v in h.values()).items()), [('bread', 240), ('milk', 240)])

    def test_every_layout_is_a_report_tier_entry_in_p2_format(self):
        for key, v in self.plan['evaluation_holdout'].items():
            self.assertEqual((v['tier'], v['layout_id']), ('report', key))
            for field in ('object', 'bin', 'blue_slot', 'cell', 'distractor_count', 'distractors', 'start',
                          'home_pose_jitter_deg', 'seed', 'instruction', 'held_out'):
                self.assertIn(field, v, key)
            self.assertEqual(v['instruction'], p2.instruction(v['object'], v['bin']))
            self.assertEqual(v['cell'], [v['object'], v['bin'], v['blue_slot']])

    def test_arms_have_the_right_bins_and_kind(self):
        # milk: trained milk->orange, held-out milk->blue; bread: trained bread->blue, held-out bread->orange
        want = {('milk', 'trained'): ('orange', False), ('milk', 'held'): ('blue', True),
                ('bread', 'trained'): ('blue', False), ('bread', 'held'): ('orange', True)}
        for pair in self.plan['pairs']:
            for arm in ('trained', 'held'):
                v = self.plan['evaluation_holdout'][pair[arm]]
                self.assertEqual((v['bin'], v['held_out']), want[(pair['object'], arm)], pair)
                self.assertEqual((v['object'], v['held_out']), (pair['object'], (v['object'], v['bin']) in p2.HELD_OUT_PAIRS))

    def test_the_two_arms_share_all_geometry_and_differ_only_in_bin_and_identity(self):
        shared = ('object', 'blue_slot', 'distractor_count', 'distractors', 'start', 'home_pose_jitter_deg', 'seed')
        for pair in self.plan['pairs']:
            a, b = (self.plan['evaluation_holdout'][pair[arm]] for arm in ('trained', 'held'))
            for field in shared:
                self.assertEqual(a[field], b[field], (pair['pair_id'], field))
            self.assertNotEqual(a['bin'], b['bin'])
            self.assertNotEqual(a['layout_id'], b['layout_id'])

    def test_balance_per_object(self):
        for obj in ('milk', 'bread'):
            pairs = [q for q in self.plan['pairs'] if q['object'] == obj]
            self.assertEqual(len(pairs), 120)
            cells = collections.Counter((self.plan['evaluation_holdout'][q['trained']]['distractor_count'],
                                         self.plan['evaluation_holdout'][q['trained']]['blue_slot']) for q in pairs)
            self.assertEqual(dict(cells), {(1, 'left'): 30, (1, 'right'): 30, (2, 'left'): 30, (2, 'right'): 30})

    def test_distractors_are_other_objects_and_every_option_is_used(self):
        for obj in ('milk', 'bread'):
            seen = collections.Counter()
            for q in self.plan['pairs']:
                v = self.plan['evaluation_holdout'][q['trained']]
                if v['object'] != obj:
                    continue
                names = [d['object'] for d in v['distractors']]
                self.assertNotIn(obj, names)
                self.assertEqual(len(names), v['distractor_count'])
                self.assertEqual(len(set(names)), len(names))
                seen.update(names)
            self.assertEqual(set(seen), set(p2.OBJECTS) - {obj}, 'every other object appears as a distractor')

    def test_every_layout_satisfies_the_p2_region_and_spacing_rules(self):
        for v in self.plan['evaluation_holdout'].values():
            names = [v['object'], *(d['object'] for d in v['distractors'])]
            poses = [v['start'], *(d['pose'] for d in v['distractors'])]
            self.assertTrue(p2.spaced(names, poses), v['layout_id'])
            for name, (x, y, yaw) in zip(names, poses):
                self.assertGreaterEqual((x * x + y * y) ** 0.5, p2.START_RADIUS_MIN_M - 1e-12)
                self.assertTrue(p2.START_X_M[0] <= x <= p2.START_X_M[1] and p2.START_Y_M[0] <= y <= p2.START_Y_M[1])
                self.assertLessEqual(abs(yaw), p2.YAW_MAX_DEG_BY_OBJECT[name] + 1e-9)

    def test_deterministic_and_digest_stable(self):
        self.assertEqual(rt.digest(rt.make_plan()), rt.digest(self.plan))


class FreshnessTests(unittest.TestCase):
    """Review focus 1: nothing in the re-test coincides with anything P2 used, checked against the sealed plan itself."""

    @classmethod
    def setUpClass(cls):
        cls.new = rt.make_plan()
        cls.old = sealed_p2_plan()

    @staticmethod
    def poses(entry):
        return [tuple(entry['start']), *(tuple(d['pose']) for d in entry['distractors'])]

    def test_no_pose_tuple_coincides_with_any_p2_episode_or_reserved_layout(self):
        old = set()
        for e in [*self.old['episodes'], *self.old['evaluation_holdout'].values()]:
            old.update(self.poses(e))
        self.assertGreater(len(old), 1000, 'the P2 plan was read as intended')
        new = {t for v in self.new['evaluation_holdout'].values() for t in self.poses(v)}
        self.assertEqual(new & old, set())

    def test_no_expert_seed_or_home_jitter_coincides_with_p2(self):
        old_seeds = {e['seed'] for e in [*self.old['episodes'], *self.old['evaluation_holdout'].values()]}
        old_jit = {tuple(e['home_pose_jitter_deg']) for e in [*self.old['episodes'], *self.old['evaluation_holdout'].values()]}
        self.assertEqual({v['seed'] for v in self.new['evaluation_holdout'].values()} & old_seeds, set())
        self.assertEqual({tuple(v['home_pose_jitter_deg']) for v in self.new['evaluation_holdout'].values()} & old_jit, set())

    def test_layout_ids_are_outside_the_p2_namespace(self):
        self.assertEqual(set(self.new['evaluation_holdout']) & set(self.old['evaluation_holdout']), set())


if __name__ == '__main__':
    unittest.main()
