"""Contract mutations must never change the authorized physical/policy budget."""
import copy
import unittest

from experiments.local_position_probe import contract as c


class ContractTests(unittest.TestCase):
    def test_canonical_slots_keep_source_identity_and_baseline_ordinals(self):
        plan = c.make_plan()
        self.assertEqual(len(plan['conditions']), 17)
        self.assertEqual(len(plan['trials']), 68)
        self.assertEqual(len({t['trial_id'] for t in plan['trials']}), 68)
        self.assertEqual([t['model'] for t in plan['trials']], ['smolvla']*34+['xvla']*34)
        for offset in (0, 34):
            self.assertEqual([t['condition_id'] for t in plan['trials'][offset:offset+2]], ['control']*2)
        for i, trial in enumerate(plan['trials'], 1):
            self.assertEqual(trial['study_index'], i)
            self.assertEqual(trial['source_episode_index'], 7)
            self.assertEqual(trial['baseline_ordinal'], {20260918: 1, 20260923: 6}[trial['seed']])
        expected = [('control', None, None, 0.)]
        for color in ('red', 'green'):
            for mm in (5, 10):
                for direction, axis, sign in (('px', 0, 1), ('nx', 0, -1), ('py', 1, 1), ('ny', 1, -1)):
                    expected.append((f'{color}_{mm}mm_{direction}', color, axis, sign*mm/1000))
        self.assertEqual([(v['condition_id'], v['object'], v['axis'], v['delta_m']) for v in plan['conditions']], expected)

    def test_mutations_types_and_extra_budget_rejected(self):
        for key, value in [('seed', 20260919), ('baseline_ordinal', 2), ('source_episode_index', 8), ('study_index', True)]:
            plan = c.make_plan(); plan['trials'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_plan(plan)
        plan = c.make_plan(); plan['trials'].append(copy.deepcopy(plan['trials'][0]))
        with self.assertRaises(ValueError): c.validate_plan(plan)
        plan = c.make_plan(); plan['conditions'][1]['delta_m'] = float('nan')
        with self.assertRaises(ValueError): c.validate_plan(plan)

    def test_plan_is_fresh_not_mutable_global_state(self):
        c.make_plan()['conditions'][0]['condition_id'] = 'changed'
        c.validate_plan(c.make_plan())

    def test_conditions_reject_resampling_or_hidden_fields(self):
        for changed in ({'condition_id': 'other'}, {**c.make_plan()['conditions'][1], 'delta_m': .02},
                        {**c.make_plan()['conditions'][1], 'resettle': True}):
            with self.assertRaises(ValueError): c.validate_condition(changed)

    def test_original_evaluated_controls_are_pinned_not_just_self_consistent(self):
        from unittest.mock import patch
        self.assertEqual(len(c.verify_baselines()), 10)
        original = c.sha256
        def corrupted(path):
            return '0'*64 if str(path).endswith('trial_06_seed_20260923/trace.npz') else original(path)
        with patch.object(c, 'sha256', side_effect=corrupted), self.assertRaises(ValueError): c.verify_baselines()


if __name__ == '__main__': unittest.main()