import ast
import importlib
import unittest
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

from experiments.b3_robosuite_data.plan import COLOURS, HELD_OUT_PAIRS, seed as plan_seed
from experiments.paired_demonstrations.plan import instruction

PROTOCOL_PY = Path(__file__).parents[1]/'protocol.py'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('experiments.xvla_b3_robosuite_eval.protocol')
        if not self.p.PLAN_JSON.exists():
            self.skipTest('B3 parent not mounted')

    def test_screen(self):
        trials = self.p.screen_trials()
        self.assertEqual(len(trials), 20)
        self.assertEqual({t['condition'] for t in trials}, {'distractor'})
        self.assertEqual({t['kind'] for t in trials}, {'trained'})
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 21)))
        self.assertEqual(len({t['layout_id'] for t in trials}), 20)
        self.assertEqual(Counter(t['edge'] != 'interior' for t in trials)[True], 5)

    def test_report(self):
        trials = self.p.report_trials()
        self.assertEqual(len(trials), 160)
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 161)))
        self.assertEqual(Counter(t['kind'] for t in trials), {'trained': 80, 'held_out': 80})
        self.assertEqual(Counter(t['condition'] for t in trials), {'control': 80, 'distractor': 80})
        self.assertEqual(Counter(t['edge'] != 'interior' for t in trials)[True], 40)
        self.assertEqual(len({t['layout_id'] for t in trials}), 40)
        by_layout_seed = defaultdict(list)  # both conditions share the seed
        for t in trials:
            by_layout_seed[(t['layout_id'], t['seed_index'])].append(t)
        self.assertEqual(len(by_layout_seed), 80)
        for pair in by_layout_seed.values():
            self.assertEqual(sorted(t['condition'] for t in pair), ['control', 'distractor'])
            self.assertEqual(len({t['seed'] for t in pair}), 1)
        self.assertEqual({t['seed_index'] for t in trials}, {0, 1})

    def test_screen_and_report_layouts_are_disjoint(self):
        self.assertFalse({t['layout_id'] for t in self.p.screen_trials()}
                         & {t['layout_id'] for t in self.p.report_trials()})

    def test_held_out_instructions_only_on_held_out_layouts(self):
        for t in self.p.screen_trials() + self.p.report_trials():
            held = (t['pick'], t['place']) in HELD_OUT_PAIRS
            self.assertEqual(held, t['kind'] == 'held_out')
            self.assertEqual(t['instruction'], instruction(t['pick'], t['place']))
            self.assertIn(t['instruction'], self.p.ALL_INSTRUCTIONS)
        held = [t for t in self.p.report_trials() if t['kind'] == 'held_out']
        self.assertEqual({(t['pick'], t['place']) for t in held}, set(HELD_OUT_PAIRS))

    def test_colours_present(self):
        for t in self.p.screen_trials() + self.p.report_trials():
            if t['condition'] == 'control':
                self.assertEqual(t['colours'], sorted({t['pick'], t['place']}))
            else:
                self.assertEqual(t['colours'], sorted(COLOURS))
            self.assertEqual(t['bank_key'], f"{t['layout_id']}:{t['condition']}")

    def test_seeds_are_stable_distinct_and_match_the_rule(self):
        screen, report = self.p.screen_trials(), self.p.report_trials()
        self.assertEqual(screen, self.p.screen_trials())
        self.assertEqual(report, self.p.report_trials())
        for t in screen:
            self.assertEqual(t['seed'], plan_seed(f"evaluation:screen:{t['layout_id']}"))
        for t in report:
            self.assertEqual(t['seed'], plan_seed(f"evaluation:report:{t['layout_id']}:{t['seed_index']}"))
        self.assertEqual(len({t['seed'] for t in screen}), 20)
        self.assertEqual(len({t['seed'] for t in report}), 80)

    def test_twelve_instructions(self):
        expected = tuple(sorted(instruction(p, q) for p, q in permutations(COLOURS, 2)))
        self.assertEqual(self.p.ALL_INSTRUCTIONS, expected)
        self.assertEqual(len(self.p.ALL_INSTRUCTIONS), 12)

    def test_eval_instruction_accepts_12_rejects_others_and_restores(self):
        from experiments.xvla_single_task import contract as single_task
        before = single_task.TASK
        for text in self.p.ALL_INSTRUCTIONS:
            with self.p.eval_instruction(text):
                self.assertEqual(single_task.TASK, text)
            self.assertEqual(single_task.TASK, before)
        for bad in ('pick up the red cube', 'pick up the red cube and place it on top of the red cube', ''):
            with self.assertRaises(ValueError):
                with self.p.eval_instruction(bad):
                    pass
            self.assertEqual(single_task.TASK, before)
        with self.assertRaises(RuntimeError):
            with self.p.eval_instruction(self.p.ALL_INSTRUCTIONS[0]):
                raise RuntimeError('boom')
        self.assertEqual(single_task.TASK, before)

    def test_output_dir(self):
        self.assertEqual(self.p.output_dir('screen', 62400).name, 'screen_step_062400')
        self.assertEqual(self.p.output_dir('report', 156000).parent, self.p.EVAL_ROOT)
        self.assertEqual(self.p.EVAL_ROOT.name, 'xvla_b3_robosuite_eval')
        self.assertEqual(self.p.CANDIDATES, (62400, 93600, 124800, 156000))
        with self.assertRaises(ValueError):
            self.p.output_dir('screen', 31200)
        with self.assertRaises(ValueError):
            self.p.output_dir('nonsense', 62400)

    def test_nothing_is_written_under_the_dataset(self):
        self.assertFalse(str(self.p.EVAL_ROOT).startswith(str(self.p.PARENT)))
        self.assertTrue(str(self.p.PARENT).endswith('b3_robosuite_20261003'))


class PurityTests(unittest.TestCase):
    def test_no_simulator_imports(self):
        banned = {'robosuite', 'mujoco', 'av'}
        for node in ast.walk(ast.parse(PROTOCOL_PY.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            for module in modules:
                self.assertNotIn(module.split('.')[0], banned, f'protocol.py imports {module}')


if __name__ == '__main__':
    unittest.main()
