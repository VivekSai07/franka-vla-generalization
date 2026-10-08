import unittest
from collections import Counter, defaultdict
from pathlib import Path

from experiments.props_p1_data.plan import HELD_OUT_PAIRS, TRAINED_PAIRS, instruction, seed as plan_seed


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = __import__('experiments.xvla_props_p1_eval.protocol', fromlist=['x'])
        if not self.p.PLAN_JSON.exists():
            self.skipTest('Props P1 parent not mounted')

    def test_screen(self):
        trials = self.p.screen_trials()
        self.assertEqual(len(trials), 20)
        self.assertEqual({t['kind'] for t in trials}, {'trained'})
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 21)))
        self.assertEqual(len({t['layout_id'] for t in trials}), 20)
        self.assertEqual(Counter(t['blue_slot'] for t in trials), {'left': 10, 'right': 10})

    def test_report(self):
        trials = self.p.report_trials()
        self.assertEqual(len(trials), 80)
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 81)))
        self.assertEqual(Counter(t['kind'] for t in trials), {'trained': 40, 'held_out': 40})
        self.assertEqual(len({t['layout_id'] for t in trials}), 40)
        by_layout = defaultdict(list)
        for t in trials:
            by_layout[t['layout_id']].append(t)
        self.assertEqual(len(by_layout), 40)
        for pair in by_layout.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(sorted(t['seed_index'] for t in pair), [0, 1])
            self.assertEqual(len({t['seed'] for t in pair}), 2)

    def test_confirm(self):
        trials = self.p.confirm_trials()
        self.assertEqual(len(trials), 20)
        self.assertEqual({t['kind'] for t in trials}, {'trained'})
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 21)))
        self.assertEqual(len({t['layout_id'] for t in trials}), 20)

    def test_screen_report_confirm_layouts_are_disjoint(self):
        ids = lambda trials: {t['layout_id'] for t in trials}
        screen, report, confirm = ids(self.p.screen_trials()), ids(self.p.report_trials()), ids(self.p.confirm_trials())
        self.assertFalse(screen & report)
        self.assertFalse(screen & confirm)
        self.assertFalse(report & confirm)

    def test_held_out_instructions_only_on_held_out_layouts(self):
        for t in self.p.screen_trials() + self.p.report_trials() + self.p.confirm_trials():
            held = (t['object'], t['bin']) in HELD_OUT_PAIRS
            self.assertEqual(held, t['kind'] == 'held_out')
            self.assertEqual(t['instruction'], instruction(t['object'], t['bin']))
            self.assertIn(t['instruction'], self.p.ALL_INSTRUCTIONS)
        held = [t for t in self.p.report_trials() if t['kind'] == 'held_out']
        self.assertEqual({(t['object'], t['bin']) for t in held}, set(HELD_OUT_PAIRS))

    def test_seeds_are_stable_distinct_and_match_the_rule(self):
        screen, report, confirm = self.p.screen_trials(), self.p.report_trials(), self.p.confirm_trials()
        self.assertEqual(screen, self.p.screen_trials())
        self.assertEqual(report, self.p.report_trials())
        self.assertEqual(confirm, self.p.confirm_trials())
        for t in screen:
            self.assertEqual(t['seed'], plan_seed(f"evaluation:screen:{t['layout_id']}"))
        for t in confirm:
            self.assertEqual(t['seed'], plan_seed(f"evaluation:confirm:{t['layout_id']}"))
        for t in report:
            self.assertEqual(t['seed'], plan_seed(f"evaluation:report:{t['layout_id']}:{t['seed_index']}"))
        self.assertEqual(len({t['seed'] for t in screen}), 20)
        self.assertEqual(len({t['seed'] for t in report}), 80)
        self.assertEqual(len({t['seed'] for t in confirm}), 20)

    def test_twelve_instructions(self):
        expected = tuple(sorted(instruction(obj, bin_) for obj, bin_ in (*TRAINED_PAIRS, *HELD_OUT_PAIRS)))
        self.assertEqual(self.p.ALL_INSTRUCTIONS, expected)
        self.assertEqual(len(self.p.ALL_INSTRUCTIONS), 12)

    def test_eval_instruction_accepts_12_rejects_others_and_restores(self):
        from experiments.xvla_single_task import contract as single_task
        before = single_task.TASK
        for text in self.p.ALL_INSTRUCTIONS:
            with self.p.eval_instruction(text):
                self.assertEqual(single_task.TASK, text)
            self.assertEqual(single_task.TASK, before)
        for bad in ('put the can in the green bin', 'put the can in the blue bin ', ''):
            with self.assertRaises(ValueError):
                with self.p.eval_instruction(bad):
                    pass
        self.assertEqual(single_task.TASK, before)
        with self.assertRaises(RuntimeError):
            with self.p.eval_instruction(self.p.ALL_INSTRUCTIONS[0]):
                raise RuntimeError('boom')
        self.assertEqual(single_task.TASK, before)

    def test_output_dir_rejects_unapproved_mode_or_candidate(self):
        self.assertEqual(self.p.output_dir('screen', 50760).name, 'screen_step_050760')
        for bad_mode in ('train', 'eval', ''):
            with self.assertRaises(ValueError):
                self.p.output_dir(bad_mode, 50760)
        for bad_candidate in (1, 2, 126901, 0):
            with self.assertRaises(ValueError):
                self.p.output_dir('screen', bad_candidate)

    def test_candidates_are_the_six_non_trivial_checkpoints(self):
        self.assertEqual(self.p.CANDIDATES, (50760, 76140, 88830, 101520, 114210, 126900))
        self.assertNotIn(1, self.p.CANDIDATES)


if __name__ == '__main__':
    unittest.main()
