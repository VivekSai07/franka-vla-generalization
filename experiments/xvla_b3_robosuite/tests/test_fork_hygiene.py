"""Fork hygiene for the X-VLA B3 robosuite training package.

Forked from the B2 X-VLA training package. Two classes of bug have shipped from forks before:
a name left dangling after a rename (#24), and tests that import the *source* package and
therefore pass while the fork is broken (#20). Both are statically detectable.
"""
import builtins
import importlib
import json
import os
import subprocess
import sys
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ('contract.py', 'runner.py')
TESTS = ('tests/test_contract.py', 'tests/test_runner.py')
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
SOURCE_PACKAGES = ('experiments.xvla_b2_distractors', 'experiments.b2_distractor_data')
RESIDUE = ('b2_distractor', 'xvla_b2_distractors', 'b2_distractor_20260928', 'B1_', '50000',
           'b1_language')


def package_files():
    return [path for path in sorted(PACKAGE.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.md')
            and path.name != 'test_fork_hygiene.py']


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

    def test_no_residue_names_survive_anywhere_in_the_package(self):
        for path in package_files():
            source = path.read_text()
            for needle in RESIDUE:
                self.assertNotIn(needle, source, f'{path.relative_to(PACKAGE)} still names {needle!r}')


class ScheduleContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_b3_robosuite.contract')

    def test_schedule_is_the_epoch_grid(self):
        c = self.c
        updates = c.checkpoint_updates(False)
        self.assertEqual(updates, [1, 31200, 62400, 93600, 124800, 156000])
        self.assertEqual(updates[-1], c.TOTAL_UPDATES)
        self.assertEqual(c.recipe(False)['max_optimizer_updates'], updates[-1])
        self.assertEqual(len(updates), 6)

    def test_learning_rate_spans_the_full_horizon(self):
        c = self.c
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 1e-5, places=9)
        with self.assertRaises(ValueError):
            c.learning_rate(c.TOTAL_UPDATES + 1)

    def test_checkpoint_count_fits_the_disk_budget(self):
        """~3.4 GB each + ~9.9 GB final, measured from this package's own 6-checkpoint schedule."""
        updates = self.c.checkpoint_updates(False)
        self.assertLessEqual(3.4 * (len(updates) - 1) + 9.9, 33.0)

    def test_exposure_matches_b2_at_five_epochs(self):
        c = self.c
        self.assertAlmostEqual(c.TOTAL_UPDATES*8/c.COMPLETE_STARTS, 5.0, delta=0.01)


class InstructionContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_b3_robosuite.contract')

    def test_ten_instructions_are_expected(self):
        self.assertEqual(len(self.c.INSTRUCTIONS), 10)

    def test_planned_instructions_cover_every_episode_and_every_task(self):
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('B3 parent not mounted')
        planned = self.c.planned_instructions()
        self.assertEqual(len(planned), 560)
        self.assertEqual(set(planned), set(self.c.INSTRUCTIONS))

    def test_training_split_contains_every_instruction(self):
        """A split that accidentally held an instruction out would make its colours untestable."""
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('B3 parent not mounted')
        planned = self.c.planned_instructions()
        for name, ids in (('train', self.c.TRAIN), ('validation', self.c.VAL)):
            self.assertEqual(set(planned[i] for i in ids), set(self.c.INSTRUCTIONS), name)

    def test_task_index_map_rejects_a_single_task_table(self):
        class _Tasks:
            index = ['pick up the red cube and place it on top of the green cube']

            def __getitem__(self, key):
                return [0]

        class _DS:
            class meta:
                tasks = _Tasks()

        with self.assertRaises(ValueError):
            self.c.task_index_map(_DS())


class SourceFilesTests(unittest.TestCase):
    def test_source_files_are_exactly_the_loaded_modules(self):
        """Every repo module that importing contract and runner loads is bound, and nothing else is."""
        c = importlib.import_module('experiments.xvla_b3_robosuite.contract')
        script = ("import sys, json, experiments.xvla_b3_robosuite.contract, experiments.xvla_b3_robosuite.runner;"
                  "print(json.dumps(sorted({getattr(m, '__file__', None) or '' for m in sys.modules.values()})))")
        out = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True)
        loaded = set()
        for file in json.loads(out.stdout.strip().splitlines()[-1]):
            if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(ROOT.resolve()):
                loaded.add(str(Path(file).resolve().relative_to(ROOT.resolve())))
        own_extra = {'experiments/xvla_b3_robosuite/README.md'} | {
            f'experiments/xvla_b3_robosuite/tests/{name}' for name in
            ('test_contract.py', 'test_runner.py', 'test_fork_hygiene.py')}
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own_extra
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        for name in own_extra:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
