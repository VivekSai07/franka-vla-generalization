"""Fork hygiene for the X-VLA props P3 training package.

Forked from `experiments.xvla_props_p2`. Two classes of bug have shipped from forks before:
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
# Split so this file does not match its own scan.
SOURCE_PACKAGES = ('experiments.xvla_props_' + 'p1', 'experiments.props_' + 'p1_data',
                   'experiments.xvla_props_' + 'p2', 'experiments.props_' + 'p2_data')
# P1's and P2's seals, parents and record fields: no reason to appear in a correct P3 fork. The README may
# name them as the fork source; imports are caught precisely above.
RESIDUE = ('86e3aadb', 'props_' + 'p1_20261006', "'74': 1", 'b1d642b3', 'props_' + 'p2_20261013',
           '955e80a6', '372cbe74', 'distractor_counts', 'held_out_pairs_collected', 'TRAINED_PAIRS')


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

    def test_readme_describes_this_package(self):
        # An absence-only residue check passes on an empty README; this one does not.
        readme = (PACKAGE / 'README.md').read_text()
        c = importlib.import_module('experiments.xvla_props_p3.contract')
        for needle in (c.SEAL_RECORD['fingerprint'], 'props_p3_20261020', '--equivalence', '--stop-at',
                       'xvla_props_p3_v1', 'SOURCE_FILES', 'sort the item'):
            self.assertIn(needle, readme)


class ScheduleContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_props_p3.contract')

    def test_schedule_is_the_dense_half_epoch_grid(self):
        c = self.c
        updates = c.checkpoint_updates(False)
        self.assertEqual(updates, [1, 50760, 76140, 88830, 101520, 114210, 126900])
        self.assertEqual(updates[-1], c.TOTAL_UPDATES)
        self.assertEqual(c.recipe(False)['max_optimizer_updates'], updates[-1])
        self.assertEqual(len(updates), 7)

    def test_learning_rate_spans_the_full_horizon(self):
        c = self.c
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 1e-5, places=9)
        with self.assertRaises(ValueError):
            c.learning_rate(c.TOTAL_UPDATES + 1)

    def test_checkpoint_count_fits_the_disk_budget(self):
        """6x3.3 GB intermediate + 9.9 GB final, measured from this package's own schedule."""
        updates = self.c.checkpoint_updates(False)
        self.assertLessEqual(3.3 * (len(updates) - 1) + 9.9, 30.0)

    def test_exposure_is_exactly_five_epochs(self):
        c = self.c
        self.assertEqual(c.TOTAL_UPDATES*8, c.COMPLETE_STARTS*5)


class InstructionContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_props_p3.contract')

    def test_one_instruction_is_expected(self):
        self.assertEqual(self.c.INSTRUCTIONS, ('sort the item',))

    def test_planned_instructions_cover_every_episode(self):
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('Props P3 parent not mounted')
        planned = self.c.planned_instructions()
        self.assertEqual(len(planned), 544)
        self.assertEqual(set(planned), set(self.c.INSTRUCTIONS))

    def test_training_and_validation_both_carry_the_prompt(self):
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('Props P3 parent not mounted')
        planned = self.c.planned_instructions()
        for name, ids in (('train', self.c.TRAIN), ('validation', self.c.VAL)):
            self.assertEqual(set(planned[i] for i in ids), {'sort the item'}, name)

    def test_task_index_map_rejects_a_table_that_is_not_the_prompt(self):
        class _Tasks:
            index = ['put the lemon in the blue bin']

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
        c = importlib.import_module('experiments.xvla_props_p3.contract')
        script = ("import sys, json, experiments.xvla_props_p3.contract, experiments.xvla_props_p3.runner;"
                  "print(json.dumps(sorted({getattr(m, '__file__', None) or '' for m in sys.modules.values()})))")
        out = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True)
        loaded = set()
        for file in json.loads(out.stdout.strip().splitlines()[-1]):
            if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(ROOT.resolve()):
                loaded.add(str(Path(file).resolve().relative_to(ROOT.resolve())))
        own_extra = {'experiments/xvla_props_p3/README.md'} | {
            f'experiments/xvla_props_p3/tests/{name}' for name in
            ('test_contract.py', 'test_runner.py', 'test_fork_hygiene.py')}
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own_extra
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        for name in own_extra:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
