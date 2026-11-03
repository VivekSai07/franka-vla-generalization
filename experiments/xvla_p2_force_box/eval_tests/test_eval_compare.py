"""The A3-against-A2 test on stage fixtures with hand-computed counts. Robosuite image, CPU.

12 comparison layouts (+1 dropped for both arms): A3 and A2 both succeed on 4, both fail on 2, A3 only on 5, A2 only on 1, so
b = 5 (new only), c = 1, exact McNemar p = 2 * (C(6,0) + C(6,1)) / 2^6 = 14/64 = 0.21875, A3 9/12, A2 5/12, estimate +1/3.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

CORE = 'c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d'
LAYOUTS = [f'L{i:02d}' for i in range(12)]
OBJECT = {i: ('can' if i % 2 == 0 else 'milk') for i in range(12)}
HELD_OUT = {5, 7}
# (A3 success, A2 success)
OUTCOME = {**{i: (True, True) for i in range(4)}, 4: (False, False), 5: (False, False),
           **{i: (True, False) for i in range(6, 11)}, 11: (False, True)}
A3_CAUSE = {4: 'never_lifted', 5: 'timeout_carrying', 11: 'never_lifted'}
A2_CAUSE = {4: 'never_lifted', 5: 'never_lifted', 6: 'never_lifted', 7: 'wrong_object', 8: 'never_lifted', 9: 'timeout_carrying',
            10: 'never_lifted'}


def episode(i, success, cause):
    return dict(layout_id=LAYOUTS[i], ordinal=i+1, tier='comparison', object=OBJECT[i], bin='blue', blue_slot='left',
                cell=[OBJECT[i], 'blue', 'left', 1], kind='held_out' if i in HELD_OUT else 'trained', distractor_count=1,
                task_success=success, cause='success' if success else cause, wrong_object=cause == 'wrong_object' and not success)


def stage(arm, checkpoint, which, dropped=('LX',), layouts=LAYOUTS, planned=12):
    causes = A3_CAUSE if which == 0 else A2_CAUSE
    episodes = [episode(i, OUTCOME[i][which], causes.get(i)) for i in range(12) if LAYOUTS[i] in layouts]
    return dict(arm=arm, checkpoint=checkpoint, status='complete', tier='comparison', planned_trials=planned,
                dropped_layouts=list(dropped), dropped_by_cell={'can-blue-left-1': len(dropped)}, missing_ordinals=[],
                episodes=episodes)


def selection(arm, winner, phase=101520):
    return dict(arm=arm, winner=winner, phase_checkpoint=phase, comparison_checkpoints=sorted({winner, phase}),
                counts={}, tied=[], rule='x', note='x')


def prepared(arm, core=CORE, checkpoints=(88830, 101520)):
    return dict(arm=dict(name=arm, adapter='x', files={}, inputs={}, sha256=arm*4), core=dict(sha256=core, provenance={}),
                phase_checkpoint=101520, checkpoints={str(c): {} for c in checkpoints})


class Fixture(unittest.TestCase):
    def setUp(self):
        from experiments.vla_inputs_eval import evaluate
        from experiments.xvla_p2_force_box import eval_compare
        self.evaluate, self.ec = evaluate, eval_compare
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write_all()
        plan = {'layouts': {lid: dict(tier='comparison', cell=['can', 'blue', 'left', 1]) for lid in [*LAYOUTS, 'LX']},
                'tiers': {'comparison': [*LAYOUTS, 'LX']}}
        patches = [patch.object(evaluate, 'load_manifests', return_value=(plan, {})),
                   patch.object(evaluate, 'verify_core'), patch.object(evaluate, 'verify_manifests'),
                   patch.object(evaluate, 'verify_arm')]
        self.mocks = {p.attribute: p.start() for p in patches}
        for p in patches:
            self.addCleanup(p.stop)

    def put(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def write_all(self, a3_winner=88830, a2_winner=88830):
        for arm in ('A2', 'A3'):
            self.put(f'arms/{arm}/prepared.json', prepared(arm))
        self.put('arms/A3/selection.json', selection('A3', a3_winner))
        self.put('arms/A2/selection.json', selection('A2', a2_winner))
        for ck in (88830, 101520):
            self.put(f'arms/A3/comparison/step_{ck:06d}/result.json', stage('A3', ck, 0))
            self.put(f'arms/A2/comparison/step_{ck:06d}/result.json', stage('A2', ck, 1))
        self.put('arms/A2/compare_step_088830.json', dict(arm='A2', checkpoint=88830, mcnemar_exact_p=0.001))


class CompareTests(Fixture):
    def test_counts_p_and_effect_are_the_hand_computed_ones(self):
        r = self.ec.compare(self.root, 88830, 88830)
        self.assertEqual((r['arm'], r['control_arm'], r['checkpoint'], r['control_checkpoint']), ('A3', 'A2', 88830, 88830))
        self.assertEqual((r['layouts'], r['new_successes'], r['control_successes']), (12, 9, 5))
        self.assertEqual(r['discordant'], dict(new_only=5, control_only=1))
        self.assertAlmostEqual(r['mcnemar_exact_p'], 0.21875)
        self.assertAlmostEqual(r['effect']['estimate'], 1/3)
        self.assertLess(r['effect']['lo'], r['effect']['estimate'])
        self.assertGreater(r['effect']['hi'], r['effect']['estimate'])
        self.assertEqual(r['dropped_layouts'], ['LX'])
        self.assertEqual(r['dropped_by_cell'], {'can-blue-left-1': 1})
        self.assertEqual(r['core_sha256'], CORE)
        written = json.loads((self.root/'arms/A3/compare_A2_step_088830_vs_088830.json').read_text())
        self.assertEqual(written['mcnemar_exact_p'], r['mcnemar_exact_p'])
        self.assertEqual(sorted(written['discordant_layouts']['new_only']), LAYOUTS[6:11])
        self.assertEqual(written['discordant_layouts']['control_only'], [LAYOUTS[11]])

    def test_it_never_compares_against_a0_and_the_result_is_never_overwritten(self):
        r = self.ec.compare(self.root, 88830, 88830)
        self.assertEqual(r['control_arm'], 'A2')
        self.assertFalse((self.root/'arms/A3/compare_step_088830.json').exists())  # M0's own A0 comparison file is not this one
        with self.assertRaises(Exception):
            self.ec.compare(self.root, 88830, 88830)

    def test_the_role_is_primary_for_the_winners_matched_phase_for_101520_and_a_mixed_pair_is_refused(self):
        self.assertIn('primary', self.ec.compare(self.root, 88830, 88830)['role'])
        self.assertEqual(self.ec.compare(self.root, 101520, 101520)['role'], 'matched-phase secondary')
        for pair in ((88830, 101520), (101520, 88830)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                self.ec.compare(self.root, *pair)

    def test_a_checkpoint_that_is_neither_the_winner_nor_the_phase_checkpoint_is_refused(self):
        self.put('arms/A3/selection.json', selection('A3', 76140))
        self.put('arms/A3/comparison/step_076140/result.json', stage('A3', 76140, 0))
        with self.assertRaises(ValueError):
            self.ec.compare(self.root, 88830, 88830)  # 88,830 is no longer one of A3's comparison checkpoints
        self.assertIn('primary', self.ec.compare(self.root, 76140, 88830)['role'])  # the winner against A2's winner

    def test_a_winner_that_is_also_the_phase_checkpoint_gives_two_distinct_pairs_and_two_files(self):
        self.put('arms/A3/selection.json', selection('A3', 101520))
        primary = self.ec.compare(self.root, 101520, 88830)  # A3's winner against A2's winner
        secondary = self.ec.compare(self.root, 101520, 101520)  # the matched-phase pair
        self.assertEqual((primary['role'], secondary['role']), ('primary', 'matched-phase secondary'))
        for name in ('compare_A2_step_101520_vs_088830.json', 'compare_A2_step_101520_vs_101520.json'):
            self.assertTrue((self.root/'arms/A3'/name).is_file(), name)
        self.assertEqual(self.ec.final(self.root)['files']['A3_vs_A2'], 'arms/A3/compare_A2_step_101520_vs_088830.json')

    def test_the_exploratory_read_outs_are_the_hand_computed_ones(self):
        r = self.ec.compare(self.root, 88830, 88830)
        e = r['exploratory']
        self.assertEqual(e['never_lifted'], dict(new=2, control=5, new_only=1, control_only=4, both=1))
        self.assertEqual(e['wrong_object'], dict(new=0, control=1))
        self.assertEqual(e['milk'], dict(n=6, new=4, control=3))
        self.assertEqual(e['held_out'], dict(n=2, new=1, control=0))
        self.assertEqual(e['gripper_box_left_frame'], dict(checked=0, left_frame=0, missing=3))  # A3's 3 failing trials, no traces

    def test_readouts_for_both_arms_are_m0s(self):
        r = self.ec.compare(self.root, 88830, 88830)
        self.assertEqual(r['readouts']['new']['overall']['count'], 9)
        self.assertEqual(r['readouts']['control']['overall']['count'], 5)

    def test_it_refuses_unless_both_arms_share_the_pinned_core(self):
        for arm in ('A2', 'A3'):
            self.put(f'arms/{arm}/prepared.json', prepared(arm, core='0'*64))
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.ec.compare(self.root, 88830, 88830)
            self.put(f'arms/{arm}/prepared.json', prepared(arm))

    def test_it_runs_m0s_own_verifications_for_both_arms_and_refuses_when_one_fails(self):
        self.ec.compare(self.root, 88830, 88830)
        verified = [call.args[0]['arm']['name'] for call in self.mocks['verify_arm'].call_args_list]
        self.assertEqual(sorted(verified), ['A2', 'A3'])
        self.assertEqual(len(self.mocks['verify_core'].call_args_list), 2)
        for name in ('verify_core', 'verify_arm', 'verify_manifests'):
            self.mocks[name].side_effect = ValueError(f'{name} failed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ec.compare(self.root, 101520, 101520)
            self.mocks[name].side_effect = None

    def test_it_refuses_different_layout_sets_dropped_layouts_or_trial_counts(self):
        self.put('arms/A3/comparison/step_088830/result.json', stage('A3', 88830, 0, layouts=LAYOUTS[:11], planned=11))
        with self.assertRaises(ValueError):
            self.ec.compare(self.root, 88830, 88830)
        self.put('arms/A3/comparison/step_088830/result.json', stage('A3', 88830, 0, dropped=('LX', 'LY')))
        with self.assertRaises(ValueError):
            self.ec.compare(self.root, 88830, 88830)
        self.put('arms/A3/comparison/step_088830/result.json', stage('A3', 88830, 0, planned=13))
        with self.assertRaises(ValueError):
            self.ec.compare(self.root, 88830, 88830)

    def test_an_incomplete_stage_is_refused(self):
        broken = stage('A3', 88830, 0)
        broken['status'] = 'partial'
        broken['missing_ordinals'] = [3]
        self.put('arms/A3/comparison/step_088830/result.json', broken)
        with self.assertRaises(RuntimeError):
            self.ec.compare(self.root, 88830, 88830)


class BoxLeftFrameTests(unittest.TestCase):
    def test_failing_trials_whose_gripper_box_was_out_of_frame_at_any_tick_are_counted(self):
        from experiments.vla_inputs_eval import protocol
        from experiments.xvla_p2_force_box import eval_compare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = protocol.stage_dir(root, 'A3', 'comparison', 88830)
            episodes = [episode(4, False, 'never_lifted'), episode(5, False, 'timeout_carrying'), episode(11, False, 'never_lifted'),
                        episode(0, True, None)]
            traces = {4: [1, 1, 1], 5: [1, 0, 1], 11: None}  # in_frame of the gripper camera per tick
            for e in episodes:
                folder = protocol.trial_folder(directory, e)
                folder.mkdir(parents=True)
                values = traces.get(int(e['layout_id'][1:]), [1, 1])
                if values is not None:
                    box = np.zeros((len(values), 2, 5))
                    box[:, 1, 4] = values
                    box[:, 0, 4] = 1
                    np.savez_compressed(folder/'trace.npz', target_box=box)
            out = eval_compare.box_left_frame(directory, [e for e in episodes if not e['task_success']])
            self.assertEqual(out, dict(checked=2, left_frame=1, missing=1))


class AmendedHolmTests(Fixture):
    def test_the_amended_family_is_a1_vs_a0_a2_vs_a0_a3_vs_a2_and_a4_vs_a0_with_unrun_arms_at_one(self):
        self.ec.compare(self.root, 88830, 88830)
        out = self.ec.final(self.root)
        self.assertEqual(sorted(out['raw']), ['A1_vs_A0', 'A2_vs_A0', 'A3_vs_A2', 'A4_vs_A0'])
        self.assertEqual(out['raw'], {'A1_vs_A0': 1.0, 'A2_vs_A0': 0.001, 'A3_vs_A2': 0.21875, 'A4_vs_A0': 1.0})
        self.assertAlmostEqual(out['holm']['A2_vs_A0'], 0.004)
        self.assertAlmostEqual(out['holm']['A3_vs_A2'], 0.65625)  # (4 - 1) * 0.21875
        self.assertEqual((out['holm']['A1_vs_A0'], out['holm']['A4_vs_A0']), (1.0, 1.0))
        self.assertFalse(out['criterion_met'])
        self.assertEqual(json.loads((self.root/'final_amended.json').read_text()), out)

    def test_the_criterion_needs_a_positive_effect_and_an_adjusted_p_below_005(self):
        self.put('arms/A2/compare_step_088830.json', dict(arm='A2', checkpoint=88830, mcnemar_exact_p=0.001))
        self.put('arms/A3/compare_A2_step_088830_vs_088830.json',
                 dict(arm='A3', checkpoint=88830, mcnemar_exact_p=0.002, effect=dict(estimate=0.04, lo=0.01, hi=0.07)))
        self.assertTrue(self.ec.final(self.root)['criterion_met'])  # holm: A2 0.004 then A3 3 * 0.002 = 0.006
        self.put('arms/A3/compare_A2_step_088830_vs_088830.json',
                 dict(arm='A3', checkpoint=88830, mcnemar_exact_p=0.002, effect=dict(estimate=-0.04, lo=-0.07, hi=-0.01)))
        self.assertFalse(self.ec.final(self.root)['criterion_met'])  # significant but negative
        self.put('arms/A3/compare_A2_step_088830_vs_088830.json',
                 dict(arm='A3', checkpoint=88830, mcnemar_exact_p=0.02, effect=dict(estimate=0.04, lo=0.01, hi=0.07)))
        self.assertFalse(self.ec.final(self.root)['criterion_met'])  # 3 * 0.02 = 0.06

    def test_an_arm_that_was_not_compared_enters_at_p_equal_one(self):
        out = self.ec.final(self.root)  # A3's comparison not run
        self.assertEqual(out['raw']['A3_vs_A2'], 1.0)
        self.assertFalse(out['criterion_met'])

    def test_the_family_is_read_at_the_selection_winners_not_at_a_chosen_checkpoint(self):
        self.put('arms/A3/compare_A2_step_088830_vs_088830.json', dict(arm='A3', mcnemar_exact_p=0.5, effect=dict(estimate=0.1)))
        self.put('arms/A3/compare_A2_step_101520_vs_101520.json', dict(arm='A3', mcnemar_exact_p=0.001, effect=dict(estimate=0.1)))
        self.assertEqual(self.ec.final(self.root)['raw']['A3_vs_A2'], 0.5)  # the winner is 88,830; the matched-phase one is secondary


class CommandLineTests(unittest.TestCase):
    def test_compare_and_final_dispatch_with_their_arguments(self):
        from experiments.xvla_p2_force_box import eval_compare
        with patch.object(eval_compare, 'compare') as compare, patch.object(eval_compare, 'final') as final:
            self.assertEqual(eval_compare.main(['--root', '/out', 'compare', '--checkpoint', '88830', '--control-checkpoint', '101520']), 0)
            compare.assert_called_once_with('/out', 88830, 101520)
            final.assert_not_called()
            self.assertEqual(eval_compare.main(['--root', '/out', 'final']), 0)
            final.assert_called_once_with('/out')

    def test_missing_or_unknown_arguments_are_refused(self):
        from experiments.xvla_p2_force_box import eval_compare
        for argv in ([], ['--root', '/out'], ['--root', '/out', 'compare'], ['--root', '/out', 'compare', '--checkpoint', '1'],
                     ['--root', '/out', 'compare', '--checkpoint', 'x', '--control-checkpoint', '1'], ['compare', '--checkpoint', '1'],
                     ['--root', '/out', 'select']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                eval_compare.main(argv)


if __name__ == '__main__':
    unittest.main()
