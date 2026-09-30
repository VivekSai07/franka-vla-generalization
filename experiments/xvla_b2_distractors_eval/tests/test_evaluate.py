"""CPU-only tests for the B2 evaluation package: roster, pick/cell mapping, checkpoint gate.

No CUDA, no MuJoCo rollout, no trained checkpoint required -- those are proven by a real
single-trial smoke test per condition before the full overnight roster runs (see
b2_eval_chain.sh). These tests exist so a roster or mapping bug is caught in seconds, not after
a GPU hour.
"""
import symtable
import unittest
from pathlib import Path

from experiments.b2_distractor_data import plan as bp
from experiments.xvla_b2_distractors_eval import evaluate as e

MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}


class UndefinedNameTests(unittest.TestCase):
    def test_no_function_reads_a_name_that_is_never_bound(self):
        """Most rollout functions never execute in CPU tests; FAILURES #24 caught this the hard way."""
        import builtins
        source = (Path(__file__).parents[1]/'evaluate.py').read_text()
        top = symtable.symtable(source, 'evaluate.py', 'exec')
        module_names = set(top.get_identifiers()) | MODULE_DUNDERS
        offenders = []

        def walk(table):
            for child in table.get_children():
                if child.get_type() == 'function':
                    for sym in child.get_symbols():
                        ident = sym.get_name()
                        if (sym.is_global() and not sym.is_assigned()
                                and ident not in module_names and not hasattr(builtins, ident)):
                            offenders.append(f'{child.get_name()}() reads undefined {ident!r}')
                walk(child)

        walk(top)
        self.assertEqual(offenders, [], '; '.join(offenders))


class RosterTests(unittest.TestCase):
    def test_screen_and_report_layouts_partition_all_sixty_disjointly(self):
        screen, report = set(e.screen_layout_ids()), set(e.report_layout_ids())
        self.assertEqual(len(screen), 20)
        self.assertEqual(len(report), 40)
        self.assertEqual(screen & report, set())
        self.assertEqual(screen | report, set(e._plan()['evaluation_layout_ids']))

    def test_screen_plan_is_twenty_trials_distractor_condition_only(self):
        trials = e.evaluation_plan('screen')
        self.assertEqual(len(trials), 20)
        self.assertEqual({t['condition'] for t in trials}, {'distractor'})
        self.assertEqual({t['layout_id'] for t in trials}, set(e.screen_layout_ids()))
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 21)))

    def test_report_plan_is_both_conditions_two_seeds_on_all_report_layouts(self):
        trials = e.evaluation_plan('report')
        self.assertEqual(len(trials), 160)
        self.assertEqual({t['condition'] for t in trials}, set(bp.CONDITIONS))
        self.assertEqual({t['layout_id'] for t in trials}, set(e.report_layout_ids()))
        for lid in e.report_layout_ids():
            for condition in bp.CONDITIONS:
                self.assertEqual(sum(1 for t in trials if t['layout_id'] == lid and t['condition'] == condition), 2)
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 161)))

    def test_unapproved_mode_rejected(self):
        with self.assertRaises(ValueError):
            e.evaluation_plan('both')

    def test_smoke_plan_covers_both_conditions_on_one_screen_and_one_report_layout(self):
        trials = e.evaluation_plan('smoke')
        self.assertEqual(len(trials), 4)
        self.assertEqual({t['condition'] for t in trials}, set(bp.CONDITIONS))
        self.assertEqual({t['layout_id'] for t in trials}, {e.screen_layout_ids()[0], e.report_layout_ids()[0]})


class PickAssignmentTests(unittest.TestCase):
    def test_picks_cover_all_sixty_reserved_layouts_red_or_green(self):
        picks = e.eval_picks()
        self.assertEqual(set(picks), set(e._plan()['evaluation_layout_ids']))
        self.assertEqual(set(picks.values()), {'red', 'green'})

    def test_picks_are_balanced_fifteen_per_cell(self):
        picks = e.eval_picks()
        holdout = e._plan()['evaluation_holdout']
        from collections import Counter
        cells = Counter()
        for lid, pick in picks.items():
            arrangement = holdout[lid]['arrangement']
            cell = next(cid for cid, spec in bp.CELLS.items()
                        if spec['pick'] == pick and spec['arrangement'] == arrangement)
            cells[cell] += 1
        self.assertEqual(dict(cells), {cid: 15 for cid in bp.CELL_IDS})

    def test_picks_are_deterministic_across_calls(self):
        self.assertEqual(e.eval_picks(), e.eval_picks())

    def test_every_entry_has_a_distinct_pick_place_instruction(self):
        picks = e.eval_picks()
        lid = e.screen_layout_ids()[0]
        entry = e.distractor_entry(lid, 'control', 'deadbeef', {'scene_hash': 'x'}, picks)
        self.assertNotEqual(entry['pick'], entry['place'])
        self.assertEqual({entry['pick'], entry['place']}, {'red', 'green'})
        self.assertEqual(entry['instruction'], bp.instruction_for(entry['pick'], entry['place']))

    def test_control_entry_carries_no_distractors_report_entry_does(self):
        picks = e.eval_picks()
        lid = e.report_layout_ids()[0]
        control = e.distractor_entry(lid, 'control', 'x', {'scene_hash': 'x'}, picks)
        distractor = e.distractor_entry(lid, 'distractor', 'x', {'scene_hash': 'x'}, picks)
        self.assertEqual(control['distractors'], [])
        self.assertEqual(control['distractor_xy_m'], {})
        self.assertEqual(set(distractor['distractors']), set(e._plan()['evaluation_holdout'][lid]['distractors']))
        self.assertEqual(control['cell'], distractor['cell'])
        self.assertEqual(control['pick'], distractor['pick'])


class CheckpointGateTests(unittest.TestCase):
    def test_candidates_are_the_five_non_trivial_checkpoints(self):
        self.assertEqual(e.CANDIDATES, (16700, 25000, 33400, 41700, 50000))
        self.assertNotIn(1, e.CANDIDATES)

    def test_verify_checkpoint_rejects_an_unapproved_candidate(self):
        for bad in (1, 8000, 16701, 0):
            with self.assertRaises(ValueError):
                e.verify_checkpoint(Path('/nonexistent'), bad)


class SummaryTests(unittest.TestCase):
    def test_summarize_counts_successes_and_flags_incomplete_ordinals(self):
        planned = [{'ordinal': 1}, {'ordinal': 2}]
        rows = [dict(ordinal=1, task_success=True, outcome='success'),
                dict(ordinal=2, task_success=False, outcome='timeout')]
        result = e.summarize(rows, planned)
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['success_rate'], 0.5)
        self.assertTrue(result['all_trials_valid'])

    def test_summarize_rejects_inconsistent_outcome_success_pairing(self):
        planned = [{'ordinal': 1}]
        rows = [dict(ordinal=1, task_success=True, outcome='timeout')]
        self.assertFalse(e.summarize(rows, planned)['all_trials_valid'])

    def test_summarize_by_condition_covers_both_conditions_with_eighty_each(self):
        by_condition = e.summarize_by_condition([])
        self.assertEqual(set(by_condition), set(bp.CONDITIONS))
        for condition in bp.CONDITIONS:
            self.assertEqual(by_condition[condition]['planned_trials'], 80)


class ForkHygieneTests(unittest.TestCase):
    def test_no_import_of_the_fork_source_or_b1_language(self):
        """Imports, not prose -- the docstring legitimately names the fork source for provenance."""
        source = (Path(__file__).parents[1]/'evaluate.py').read_text()
        for banned in ('import experiments.xvla_b1_language_eval',
                       'from experiments.xvla_b1_language_eval',
                       'from experiments.b1_language_data', 'import experiments.b1_language_data'):
            self.assertNotIn(banned, source, f'evaluate.py still references {banned!r}')

    def test_module_actually_calls_main_when_run_as_a_script(self):
        """Regression: `python -m ...evaluate --mode=X` silently did nothing and exited 0 in
        under 5 seconds, because the fork dropped the `if __name__ == '__main__':` guard. Every
        `docker run` in the eval chain "passed" instantly on a no-op, and the chain moved on to
        the next stage on a completely fake result. Caught only by checking wall-clock time was
        implausible for a real GPU rollout -- this test makes it a CPU-visible assertion instead."""
        source = (Path(__file__).parents[1]/'evaluate.py').read_text()
        self.assertIn("if __name__ == '__main__':", source)
        guard = source.split("if __name__ == '__main__':", 1)[1]
        self.assertIn('main(', guard, 'the __main__ guard must actually call main()')


if __name__ == '__main__':
    unittest.main()
