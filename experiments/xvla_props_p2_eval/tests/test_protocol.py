import unittest
from collections import Counter

from experiments.xvla_props_p2_eval import protocol


class NeedsPlan(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P2 sealed parent not mounted')


class TierTests(NeedsPlan):
    def test_tier_sizes_and_composition(self):
        self.assertEqual([len(protocol.screen_trials()), len(protocol.report_trials()),
                          len(protocol.control_trials()), len(protocol.confirm_trials())], [20, 120, 20, 20])
        report = protocol.report_trials()
        self.assertEqual(len({t['layout_id'] for t in report}), 60)
        self.assertEqual(Counter(t['kind'] for t in report), {'trained': 80, 'held_out': 40})
        self.assertEqual(Counter(t['distractor_count'] for t in report if t['kind'] == 'trained'), {1: 40, 2: 40})
        self.assertEqual({t['distractor_count'] for t in protocol.control_trials()}, {0})
        for tier in ('screen', 'confirm'):
            self.assertEqual(Counter(t['distractor_count'] for t in protocol.select_trials(tier)), {1: 10, 2: 10})

    def test_tiers_use_disjoint_layouts(self):
        sets = {m: {t['layout_id'] for t in protocol.select_trials(m)} for m in ('screen', 'report', 'control', 'confirm')}
        keys = list(sets)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                self.assertFalse(sets[a] & sets[b], (a, b))

    def test_held_out_pairs_only_in_the_report_tier(self):
        held = {tuple(p) for p in protocol._plan()['held_out_pairs']}
        for mode in ('screen', 'control', 'confirm'):
            self.assertFalse(any((t['object'], t['bin']) in held for t in protocol.select_trials(mode)), mode)
        self.assertEqual({(t['object'], t['bin'], t['blue_slot']) for t in protocol.report_trials()
                          if t['kind'] == 'held_out'},
                         {(o, b, s) for o, b in held for s in ('left', 'right')})

    def test_trials_carry_everything_needed_to_restore_the_scene(self):
        for t in protocol.report_trials()[:10]:
            self.assertEqual(len(t['distractors']), t['distractor_count'])
            self.assertEqual(len(t['start']), 3)
            self.assertEqual(len(t['home_pose_jitter_deg']), 7)
            self.assertNotIn(t['object'], [d['object'] for d in t['distractors']])

    def test_seeds_are_stable_distinct_and_follow_the_rule(self):
        report = protocol.report_trials()
        self.assertEqual(report, protocol.report_trials())
        seeds = [t['seed'] for tier in ('screen', 'report', 'control', 'confirm') for t in protocol.select_trials(tier)]
        self.assertEqual(len(seeds), len(set(seeds)))
        for t in report:
            self.assertEqual(t['seed'], protocol.plan_seed(f"evaluation:report:{t['layout_id']}:{t['seed_index']}"))
        self.assertEqual({t['seed_index'] for t in report}, {0, 1})
        self.assertEqual(Counter(t['layout_id'] for t in report).most_common(1)[0][1], 2)
        self.assertEqual([t['ordinal'] for t in report], list(range(1, 121)))

    def test_smoke_is_a_two_distractor_screen_trial_and_a_held_out_report_trial(self):
        smoke = protocol.smoke_trials()
        self.assertEqual((smoke[0]['distractor_count'], smoke[1]['kind']), (2, 'held_out'))
        self.assertEqual([t['ordinal'] for t in smoke], [1, 2])


class JobTests(NeedsPlan):
    def test_jobs_cover_every_candidate_trial_pair_once(self):
        jobs = protocol.jobs('screen', 'p2', protocol.CANDIDATES)
        self.assertEqual(len(jobs), 120)
        self.assertEqual(len({(j['candidate'], j['trial']['ordinal']) for j in jobs}), 120)

    def test_shards_partition_the_jobs_and_load_each_policy_once_per_worker(self):
        jobs = protocol.jobs('report', 'p2', (88830, 126900))
        shards = [protocol.shard_jobs(jobs, k, 8) for k in range(8)]
        got = [(j['candidate'], j['trial']['ordinal']) for s in shards for j in s]
        self.assertEqual(sorted(got), sorted((j['candidate'], j['trial']['ordinal']) for j in jobs))
        self.assertEqual(len(got), len(set(got)))
        for s in shards:
            order = [j['candidate'] for j in s]
            self.assertEqual(order, sorted(order))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_invalid_jobs_and_shards_are_refused(self):
        for bad in (dict(mode='screen', policy='p1', candidates=(50760,)),
                    dict(mode='screen', policy='p2', candidates=(1,)),
                    dict(mode='screen', policy='p2', candidates=()),
                    dict(mode='screen', policy='p2', candidates=(50760, 50760)),
                    dict(mode='nope', policy='p2', candidates=(50760,)),
                    dict(mode='screen', policy='x', candidates=(50760,))):
            with self.assertRaises(ValueError, msg=bad):
                protocol.jobs(**bad)
        jobs = protocol.jobs('control', 'p2', (126900,))
        for k, n in ((8, 8), (-1, 8), (0, 0), (0, 1.5)):
            with self.assertRaises(ValueError):
                protocol.shard_jobs(jobs, k, n)

    def test_p1_policy_runs_only_its_126900_checkpoint(self):
        self.assertEqual({j['candidate'] for j in protocol.jobs('report', 'p1', (126900,))}, {126900})
        self.assertEqual(len(protocol.jobs('report', 'p1', (126900,))), 120)


class InstructionAndPathTests(unittest.TestCase):
    def test_twelve_instructions_and_the_context_restores(self):
        self.assertEqual(len(protocol.ALL_INSTRUCTIONS), 12)
        original = protocol.single_task.TASK
        with protocol.eval_instruction('put the milk in the blue bin'):
            self.assertEqual(protocol.single_task.TASK, 'put the milk in the blue bin')
        self.assertEqual(protocol.single_task.TASK, original)
        with self.assertRaises(ValueError):
            with protocol.eval_instruction('put the can in the green bin'):
                pass
        with self.assertRaises(RuntimeError):
            with protocol.eval_instruction('put the can in the blue bin'):
                raise RuntimeError
        self.assertEqual(protocol.single_task.TASK, original)

    def test_output_dir_rejects_unapproved_values(self):
        self.assertTrue(str(protocol.output_dir('report', 'p2', 88830)).endswith('report_p2_step_088830'))
        for bad in (('nope', 'p2', 88830), ('report', 'x', 88830), ('report', 'p2', 1), ('report', 'p1', 88830)):
            with self.assertRaises(ValueError):
                protocol.output_dir(*bad)

    def test_candidates_are_the_six_non_trivial_checkpoints(self):
        self.assertEqual(protocol.CANDIDATES, (50760, 76140, 88830, 101520, 114210, 126900))


if __name__ == '__main__':
    unittest.main()
