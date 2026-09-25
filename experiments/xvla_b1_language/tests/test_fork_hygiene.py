"""Fork hygiene for the X-VLA B1 training package.

Forked from `xvla_broad_workspace_long`. Two classes of bug have shipped from forks before:
a name left dangling after a rename (#24), and tests that import the *source* package and
therefore pass while the fork is broken (#20). Both are statically detectable.
"""
import builtins
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = ('contract.py', 'runner.py')
TESTS = ('tests/test_contract.py', 'tests/test_runner.py')
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
SOURCE_PACKAGES = ('experiments.xvla_broad_workspace_long', 'experiments.broad_workspace_data')


class UndefinedNameTests(unittest.TestCase):
    def test_no_module_reads_a_name_that_is_never_bound(self):
        offenders = []
        for name in MODULES:
            source = (PACKAGE / name).read_text()
            top = symtable.symtable(source, name, 'exec')
            module_names = set(top.get_identifiers()) | MODULE_DUNDERS

            def walk(table):
                for child in table.get_children():
                    if child.get_type() == 'function':
                        for sym in child.get_symbols():
                            ident = sym.get_name()
                            if (sym.is_global() and not sym.is_assigned()
                                    and ident not in module_names and not hasattr(builtins, ident)):
                                offenders.append(f'{name}:{child.get_name()}() reads undefined {ident!r}')
                    walk(child)

            walk(top)
        self.assertEqual(offenders, [], '; '.join(offenders))

    def test_nothing_imports_or_patches_the_fork_source(self):
        for name in MODULES + TESTS:
            source = (PACKAGE / name).read_text()
            for package in SOURCE_PACKAGES:
                for banned in (f'from {package}', f'import {package}', f"'{package}.", f'"{package}.'):
                    self.assertNotIn(banned, source, f'{name} still reaches into {package}')

    def test_no_stale_parent_output_names_survive(self):
        for name in MODULES + TESTS:
            source = (PACKAGE / name).read_text()
            self.assertNotIn('broadworkspace_long_v1', source, f'{name} names the old run directory')
            self.assertNotIn('broad_workspace_20260920', source, f'{name} names the old parent')


class ScheduleContractTests(unittest.TestCase):
    """The horizon and checkpoint set are the decisions this fork exists to change."""

    def setUp(self):
        import importlib
        self.c = importlib.import_module('experiments.xvla_b1_language.contract')

    def test_horizon_is_double_the_matched_exposure_run(self):
        self.assertEqual(self.c.MATCHED_EXPOSURE_UPDATES, 16700)
        self.assertEqual(self.c.TOTAL_UPDATES, 33400)

    def test_learning_rate_spans_the_full_horizon(self):
        c = self.c
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 1e-5, places=9)
        with self.assertRaises(ValueError):
            c.learning_rate(c.TOTAL_UPDATES + 1)

    def test_matched_exposure_lr_differs_from_the_original_run(self):
        """Stretching the cosine means 16700 is mid-decay here, not the floor — by design."""
        self.assertGreater(self.c.learning_rate(self.c.MATCHED_EXPOSURE_UPDATES), 1e-5)

    def test_three_evaluation_candidates_are_checkpointed(self):
        updates = self.c.checkpoint_updates(False)
        for wanted in (self.c.MATCHED_EXPOSURE_UPDATES, 25000, self.c.TOTAL_UPDATES):
            self.assertIn(wanted, updates)
        self.assertEqual(updates, sorted(set(updates)))
        self.assertEqual(updates[-1], self.c.TOTAL_UPDATES)

    def test_checkpoint_count_fits_the_disk_budget(self):
        """~3.4 GB each + ~9.9 GB final, against ~36 GB free once collection lands."""
        updates = self.c.checkpoint_updates(False)
        self.assertLessEqual(3.4 * (len(updates) - 1) + 9.9, 30.0)

    def test_seal_placeholder_blocks_training_until_admission(self):
        if self.c.SEAL != 'PENDING_B1_ADMISSION':
            self.skipTest('seal already injected by the collection chain')
        with self.assertRaises(ValueError):
            self.c.current_binding()


class InstructionContractTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.c = importlib.import_module('experiments.xvla_b1_language.contract')

    def test_two_instructions_are_expected(self):
        self.assertEqual(len(self.c.INSTRUCTIONS), 2)

    def test_planned_instructions_cover_every_episode_and_both_tasks(self):
        planned = self.c.planned_instructions()
        self.assertEqual(len(planned), 200)
        self.assertEqual(set(planned), set(self.c.INSTRUCTIONS))

    def test_training_split_contains_both_instructions(self):
        """A split that accidentally held one instruction out would make B1 untestable."""
        planned = self.c.planned_instructions()
        for name, ids in (('train', self.c.TRAIN), ('validation', self.c.VAL)):
            self.assertEqual(set(planned[i] for i in ids), set(self.c.INSTRUCTIONS), name)

    def test_task_index_map_rejects_a_single_task_dataset(self):
        class _Tasks:
            index = ['pick up the red cube and place it on top of the green cube']

            def __getitem__(self, key):
                return [0]

        class _DS:
            class meta:
                tasks = _Tasks()

        with self.assertRaises(ValueError):
            self.c.task_index_map(_DS())


if __name__ == '__main__':
    unittest.main()
