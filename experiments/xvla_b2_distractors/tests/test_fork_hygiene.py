"""Fork hygiene for the X-VLA B2 distractors training package.

Forked from `xvla_b1_language`. Two classes of bug have shipped from forks before:
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
SOURCE_PACKAGES = ('experiments.xvla_b1_language', 'experiments.b1_language_data')


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
            self.assertNotIn('b1_language_v1_20260925', source, f'{name} names the old run directory')
            self.assertNotIn('b1_language_20260925', source, f'{name} names the old parent')


class ScheduleContractTests(unittest.TestCase):
    """The horizon and checkpoint set are the decisions this fork exists to change."""

    def setUp(self):
        import importlib
        self.c = importlib.import_module('experiments.xvla_b2_distractors.contract')

    def test_horizon_extends_past_b1s_winner(self):
        self.assertEqual(self.c.B1_CALIBRATION_UPDATES, 16700)
        self.assertEqual(self.c.B1_WINNER_UPDATES, 33400)
        self.assertEqual(self.c.TOTAL_UPDATES, 50000)
        self.assertGreater(self.c.TOTAL_UPDATES, self.c.B1_WINNER_UPDATES)

    def test_learning_rate_spans_the_full_horizon(self):
        c = self.c
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 1e-5, places=9)
        with self.assertRaises(ValueError):
            c.learning_rate(c.TOTAL_UPDATES + 1)

    def test_b1_calibration_points_are_mid_decay_here(self):
        """Stretching the cosine means B1's schedule points are mid-decay here, not the floor."""
        self.assertGreater(self.c.learning_rate(self.c.B1_CALIBRATION_UPDATES), 1e-5)
        self.assertGreater(self.c.learning_rate(self.c.B1_WINNER_UPDATES), 1e-5)

    def test_five_evaluation_candidates_are_checkpointed(self):
        updates = self.c.checkpoint_updates(False)
        for wanted in (self.c.B1_CALIBRATION_UPDATES, 25000, self.c.B1_WINNER_UPDATES, 41700,
                       self.c.TOTAL_UPDATES):
            self.assertIn(wanted, updates)
        self.assertEqual(updates, sorted(set(updates)))
        self.assertEqual(updates[-1], self.c.TOTAL_UPDATES)
        self.assertEqual(len(updates), 6)

    def test_checkpoint_count_fits_the_disk_budget(self):
        """~3.4 GB each + ~9.9 GB final, measured from this package's own 6-checkpoint schedule."""
        updates = self.c.checkpoint_updates(False)
        self.assertLessEqual(3.4 * (len(updates) - 1) + 9.9, 33.0)

    def test_seal_placeholder_blocks_training_until_admission(self):
        if self.c.SEAL != 'PENDING_B2_ADMISSION':
            self.skipTest('seal already injected from the sealed B2 collection')
        with self.assertRaises(ValueError):
            self.c.current_binding()


class InstructionContractTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.c = importlib.import_module('experiments.xvla_b2_distractors.contract')

    def test_two_instructions_are_expected(self):
        self.assertEqual(len(self.c.INSTRUCTIONS), 2)

    def test_planned_instructions_cover_every_episode_and_both_tasks(self):
        planned = self.c.planned_instructions()
        self.assertEqual(len(planned), 200)
        self.assertEqual(set(planned), set(self.c.INSTRUCTIONS))

    def test_training_split_contains_both_instructions(self):
        """A split that accidentally held one instruction out would make the contrast untestable."""
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
