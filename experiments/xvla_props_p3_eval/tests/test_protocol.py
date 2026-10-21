"""Protocol tests: the trials come from the sealed plan, one seed each, the one prompt, sharding. Admission image."""
import json
import unittest

from experiments.props_p3_data import plan as p3_plan
from experiments.xvla_props_p3_eval import protocol

# Written out, not read from the code under test.
EXPECTED_BIN = {'can': 'orange', 'milk': 'orange', 'bottle': 'orange', 'bread': 'blue', 'lemon': 'blue', 'cereal': 'blue'}
TRAINED = ('can', 'milk', 'bread', 'lemon')
HELD_OUT = ('bottle', 'cereal')
PROMPT = 'sort the item'


class NeedsPlan(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P3 sealed parent not mounted')


class TrialTests(NeedsPlan):
    def test_tier_sizes_and_ordinals(self):
        self.assertEqual((len(protocol.screen_trials()), len(protocol.report_trials()), len(protocol.confirm_trials())),
                         (24, 240, 24))
        self.assertEqual(protocol.LAYOUTS, 288)
        for trials in (protocol.screen_trials(), protocol.report_trials(), protocol.confirm_trials()):
            self.assertEqual([t['ordinal'] for t in trials], list(range(1, len(trials) + 1)))
            self.assertEqual(len({t['layout_id'] for t in trials}), len(trials))

    def test_every_trial_has_one_seed_the_fixed_prompt_and_the_objects_container(self):
        every = protocol.screen_trials() + protocol.report_trials() + protocol.confirm_trials()
        self.assertEqual(len({t['seed'] for t in every}), 288, 'one distinct rollout seed per layout')
        for t in every:
            self.assertEqual(t['instruction'], PROMPT)
            self.assertEqual(t['bin'], EXPECTED_BIN[t['object']], t['layout_id'])
            self.assertEqual(t['seed_index'], 0)
            self.assertEqual(t['kind'], 'held_out' if t['object'] in HELD_OUT else 'trained')
            self.assertNotIn('distractors', t)
        self.assertNotIn('put the', json.dumps(every))

    def test_rollout_seeds_are_not_the_collection_or_expert_seeds(self):
        plan = protocol._plan()
        used = set(plan['rng_seeds'].values())
        self.assertFalse(used & {t['seed'] for t in protocol.screen_trials() + protocol.report_trials()
                                 + protocol.confirm_trials()})

    def test_the_composition_is_the_pre_registered_one(self):
        report = protocol.report_trials()
        trained = [t for t in report if t['kind'] == 'trained']
        held = [t for t in report if t['kind'] == 'held_out']
        self.assertEqual((len(trained), len(held)), (80, 160))
        self.assertEqual({o: sum(t['object'] == o for t in held) for o in HELD_OUT}, {'bottle': 80, 'cereal': 80})
        for slot in ('left', 'right'):
            for obj in HELD_OUT:
                self.assertEqual(sum(t['object'] == obj and t['blue_slot'] == slot for t in held), 40)
            for obj in TRAINED:
                self.assertEqual(sum(t['object'] == obj and t['blue_slot'] == slot for t in trained), 10)
        for trials in (protocol.screen_trials(), protocol.confirm_trials()):
            self.assertEqual({t['kind'] for t in trials}, {'trained'})
            for obj in TRAINED:
                self.assertEqual(sum(t['object'] == obj for t in trials), 6)

    def test_trials_carry_the_sealed_layouts(self):
        hold = protocol._plan()['evaluation_holdout']
        for t in protocol.report_trials() + protocol.screen_trials():
            layout = hold[t['layout_id']]
            for key in ('start', 'home_pose_jitter_deg', 'blue_slot', 'object', 'bin', 'category'):
                self.assertEqual(t[key], layout[key], (t['layout_id'], key))
            self.assertEqual(t['cell'], list(layout['cell']))
            self.assertEqual(t['tier'], layout['tier'])

    def test_the_seed_comes_from_the_collection_seed_function_and_the_layout_id(self):
        t = protocol.confirm_trials()[0]
        self.assertEqual(t['seed'], p3_plan.seed(f"rollout:{t['layout_id']}"))
        self.assertNotEqual(t['seed'], p3_plan.seed(f"rollout:{protocol.confirm_trials()[1]['layout_id']}"))

    def test_smoke_trials_are_one_trained_and_one_held_out(self):
        smoke = protocol.smoke_trials()
        self.assertEqual([t['kind'] for t in smoke], ['trained', 'held_out'])
        self.assertEqual([t['ordinal'] for t in smoke], [1, 2])

    def test_select_trials_refuses_other_modes(self):
        for mode in protocol.MODES:
            self.assertTrue(protocol.select_trials(mode))
        for bad in ('control', 'p1', '', 'Screen'):
            with self.assertRaises(ValueError):
                protocol.select_trials(bad)


class JobTests(NeedsPlan):
    def test_jobs_are_ordered_by_candidate_then_ordinal(self):
        jobs = protocol.jobs('screen', (101520, 50760))
        self.assertEqual(len(jobs), 48)
        self.assertEqual([j['candidate'] for j in jobs], [50760] * 24 + [101520] * 24)
        self.assertEqual([j['trial']['ordinal'] for j in jobs[:24]], list(range(1, 25)))

    def test_unapproved_candidates_are_refused(self):
        for bad in ((), (1,), (50760, 50760), (50760, 12345), (126901,)):
            with self.assertRaises(ValueError, msg=str(bad)):
                protocol.jobs('screen', bad)

    def test_shards_partition_the_jobs_interleaved_and_sorted_by_candidate(self):
        jobs = protocol.jobs('screen', protocol.CANDIDATES)
        shards = [protocol.shard_jobs(jobs, k, 5) for k in range(5)]
        key = lambda j: (j['candidate'], j['trial']['ordinal'])
        self.assertEqual(sorted(map(key, (j for s in shards for j in s))), sorted(map(key, jobs)))
        self.assertEqual(sum(len(s) for s in shards), len(jobs))
        for s in shards:
            self.assertEqual(list(map(key, s)), sorted(map(key, s)))
            self.assertEqual({j['candidate'] for j in s}, set(protocol.CANDIDATES), 'every worker sees every checkpoint')
        first = [j for j in jobs if j['candidate'] == 50760]
        self.assertEqual({j['trial']['ordinal'] for j in shards[0] if j['candidate'] == 50760},
                         {j['trial']['ordinal'] for j in first[0::5]})

    def test_a_share_is_ordered_by_candidate_even_if_the_stage_was_not(self):
        jobs = list(reversed(protocol.jobs('screen', (50760, 76140))))
        share = protocol.shard_jobs(jobs, 1, 2)
        key = [(j['candidate'], j['trial']['ordinal']) for j in share]
        self.assertEqual(key, sorted(key))

    def test_invalid_shards_are_refused(self):
        jobs = protocol.jobs('screen', (50760,))
        for shard, shards in ((-1, 3), (3, 3), (0, 0), (0, 1.5), (1.0, 3)):
            with self.assertRaises(ValueError, msg=f'{shard}/{shards}'):
                protocol.shard_jobs(jobs, shard, shards)


class InstructionTests(unittest.TestCase):
    def test_only_the_fixed_prompt_is_approved_and_the_shared_task_is_restored(self):
        before = protocol.single_task.TASK
        with protocol.eval_instruction(PROMPT):
            self.assertEqual(protocol.single_task.TASK, PROMPT)
        self.assertEqual(protocol.single_task.TASK, before)
        for bad in ('put the can in the orange bin', 'put the cereal in the blue bin', 'Sort the item', 'sort the items', ''):
            with self.assertRaises(ValueError, msg=bad):
                with protocol.eval_instruction(bad):
                    pass
        with self.assertRaises(RuntimeError):
            with protocol.eval_instruction(PROMPT):
                raise RuntimeError('boom')
        self.assertEqual(protocol.single_task.TASK, before)

    def test_output_dirs_and_trial_folders_are_validated(self):
        self.assertEqual(protocol.output_dir('screen', 50760).name, 'screen_step_050760')
        self.assertEqual(protocol.output_dir('confirm', 126900).parent, protocol.EVAL_ROOT)
        for mode, candidate in (('control', 50760), ('screen', 1), ('screen', 126901), ('', 50760)):
            with self.assertRaises(ValueError, msg=f'{mode} {candidate}'):
                protocol.output_dir(mode, candidate)
        trial = dict(ordinal=7, layout_id='report_006')
        self.assertEqual(protocol.trial_folder('/x', trial).as_posix(), '/x/trial_007_report_006')


if __name__ == '__main__':
    unittest.main()
