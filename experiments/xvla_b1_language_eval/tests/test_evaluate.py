"""CPU-only tests for the B1 evaluation package: roster, cell mapping, checkpoint gate.

No CUDA, no MuJoCo rollout, no trained checkpoint required -- those are proven by a real
single-trial smoke test per cell before the full overnight roster runs (see b1_eval_chain.sh).
These tests exist so a roster or mapping bug is caught in seconds, not after a GPU hour.
"""
import symtable
import unittest
from pathlib import Path

from experiments.b1_language_data import plan as bp
from experiments.xvla_b1_language_eval import evaluate as e

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
    def test_screen_and_report_layouts_partition_all_eighty_disjointly(self):
        screen, report = set(e.screen_layout_ids()), set(e.report_layout_ids())
        self.assertEqual(len(screen), 40)
        self.assertEqual(len(report), 40)
        self.assertEqual(screen & report, set())
        self.assertEqual(screen | report, set(e._plan()['evaluation_layout_ids']))

    def test_screen_plan_is_forty_trials_of_the_canonical_cell(self):
        trials = e.evaluation_plan('screen')
        self.assertEqual(len(trials), 40)
        self.assertEqual({t['cell'] for t in trials}, {'C1'})
        self.assertEqual({t['layout_id'] for t in trials}, set(e.screen_layout_ids()))
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 41)))

    def test_report_plan_is_all_four_cells_on_all_report_layouts(self):
        trials = e.evaluation_plan('report')
        self.assertEqual(len(trials), 160)
        self.assertEqual({t['cell'] for t in trials}, set(bp.CELL_IDS))
        self.assertEqual({t['layout_id'] for t in trials}, set(e.report_layout_ids()))
        for lid in e.report_layout_ids():
            self.assertEqual({t['cell'] for t in trials if t['layout_id'] == lid}, set(bp.CELL_IDS))
        self.assertEqual([t['ordinal'] for t in trials], list(range(1, 161)))

    def test_unapproved_mode_rejected(self):
        with self.assertRaises(ValueError):
            e.evaluation_plan('both')

    def test_smoke_plan_covers_all_four_cells_on_one_layout(self):
        trials = e.evaluation_plan('smoke')
        self.assertEqual(len(trials), 4)
        self.assertEqual({t['cell'] for t in trials}, set(bp.CELL_IDS))
        self.assertEqual({t['layout_id'] for t in trials}, {e.screen_layout_ids()[0]})


class CellMappingTests(unittest.TestCase):
    def test_every_cell_maps_to_a_distinct_pick_place_instruction(self):
        seen_instructions = set()
        for cell in bp.CELL_IDS:
            entry = e.language_entry('evaluation_000', cell, 'deadbeef', {'scene_hash': 'x'})
            self.assertNotEqual(entry['pick'], entry['place'])
            self.assertEqual({entry['pick'], entry['place']}, {'red', 'green'})
            self.assertEqual(entry['instruction'], bp.instruction_for(entry['pick'], entry['place']))
            seen_instructions.add(entry['instruction'])
        self.assertEqual(seen_instructions, set(bp.INSTRUCTIONS))

    def test_c1_and_c3_share_an_arrangement_but_differ_in_pick(self):
        c1 = e.language_entry('evaluation_000', 'C1', 'x', {'scene_hash': 'x'})
        c3 = e.language_entry('evaluation_000', 'C3', 'x', {'scene_hash': 'x'})
        self.assertEqual(c1['arrangement'], c3['arrangement'])
        self.assertNotEqual(c1['pick'], c3['pick'])
        self.assertNotEqual(c1['instruction'], c3['instruction'])

    def test_c1_and_c2_share_a_pick_but_differ_in_arrangement(self):
        c1 = e.language_entry('evaluation_000', 'C1', 'x', {'scene_hash': 'x'})
        c2 = e.language_entry('evaluation_000', 'C2', 'x', {'scene_hash': 'x'})
        self.assertEqual(c1['pick'], c2['pick'])
        self.assertNotEqual(c1['arrangement'], c2['arrangement'])
        self.assertEqual(c1['instruction'], c2['instruction'])


class CheckpointGateTests(unittest.TestCase):
    def test_candidates_are_the_three_viable_checkpoints_not_the_full_five(self):
        self.assertEqual(e.CANDIDATES, (16700, 25000, 33400))
        self.assertNotIn(1, e.CANDIDATES)
        self.assertNotIn(8000, e.CANDIDATES)

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

    def test_summarize_by_cell_covers_all_four_cells_with_forty_each(self):
        by_cell = e.summarize_by_cell([])
        self.assertEqual(set(by_cell), set(bp.CELL_IDS))
        for cell in bp.CELL_IDS:
            self.assertEqual(by_cell[cell]['planned_trials'], 40)


class ForkHygieneTests(unittest.TestCase):
    def test_no_import_of_the_fork_source_or_broad_workspace(self):
        """Imports, not prose -- the docstring legitimately names the fork source for provenance.

        The real hardcoding risk (`pick`/`place` fixed to red/green) is covered by
        CellMappingTests, which prove all four cells produce distinct pick/place/instruction.
        """
        source = (Path(__file__).parents[1]/'evaluate.py').read_text()
        for banned in ('import experiments.xvla_broad_workspace_long_eval',
                       'from experiments.xvla_broad_workspace_long_eval',
                       'from experiments.broad_workspace_data', 'import experiments.broad_workspace_data'):
            self.assertNotIn(banned, source, f'evaluate.py still references {banned!r}')


if __name__ == '__main__':
    unittest.main()
