"""Fork hygiene for the X-VLA P2 + force/torque + drawn-box training package.

Forked from `experiments.xvla_p2_force` (M2). Two classes of bug have shipped from forks before: a name left dangling after a
rename (#24), and tests that import the *source* package and therefore pass while the fork is broken (#20). Both are
statically detectable. The fork may name its source only in these ways: M2's `forces` module (imported, never copied: the
force transform is frozen), the runner's one named reference (`REFERENCE_RUNNER`, imported lazily inside the zero-equivalence
gate; the files that import loads are bound, `contract.REFERENCE_FILES`), and the paths of those files in `SOURCE_FILES`.
`tests/test_m2_reference.py` and this file are exempt: they compare with the source.
"""
import builtins
import importlib
import json
import os
import re
import subprocess
import sys
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ('contract.py', 'boxes.py', 'overlay.py', 'runner.py')
BOUND_TESTS = ('test_contract.py', 'test_runner.py', 'test_fork_hygiene.py', 'test_overlay.py', 'test_boxes_join.py',
               'test_inputs.py', 'test_m2_reference.py', 'test_zero_equivalence.py')
EXEMPT = ('test_fork_hygiene.py', 'test_m2_reference.py')
TESTS = tuple('tests/'+name for name in BOUND_TESTS if name not in EXEMPT)
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
SOURCE_PACKAGE = 'experiments.xvla_p2_' + 'force'
SOURCE_PATH = 'experiments/xvla_p2_' + 'force'
# The only ways the fork may name its source (longest first).
ALLOWED = (SOURCE_PACKAGE + ' import forces', SOURCE_PACKAGE + '.forces', SOURCE_PACKAGE + '.runner',
           SOURCE_PATH + '/forces.py', SOURCE_PATH + '/__init__.py', SOURCE_PATH + '/contract.py', SOURCE_PATH + '/runner.py')
# The source's output names and purpose strings, and its own fork source: no reason to appear in the fork. The README may
# name the source package as the fork source; imports are caught precisely above.
RESIDUE = ("'xvla_p2_" + "force'", 'xvla_p2_' + 'force_v1', 'xvla_p2_' + 'force_preflight', 'xvla_props_' + 'p2')


def package_files():
    return [path for path in sorted(PACKAGE.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.md')
            and path.name not in EXEMPT]


def mentions(source):
    """Every remaining way the text names the fork source after the allowed forms are removed."""
    for form in sorted(ALLOWED, key=len, reverse=True):
        source = source.replace(form, '')
    return re.findall(r'experiments[./]xvla_p2_force(?![A-Za-z0-9_])', source)


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

    def test_the_fork_names_its_source_only_in_the_allowed_ways(self):
        for name in MODULES + TESTS:
            source = (PACKAGE / name).read_text()
            self.assertEqual(mentions(source), [], f'{name} reaches into {SOURCE_PACKAGE} beyond the allowed forms')
        for name in ('contract.py', 'boxes.py'):
            source = (PACKAGE / name).read_text()
            self.assertEqual(re.findall(r'(?:from|import) +' + re.escape(SOURCE_PACKAGE) + r'\b[^\n]*', source),
                             [f'from {SOURCE_PACKAGE} import forces'], name)
        for name in ('overlay.py', 'runner.py'):
            source = (PACKAGE / name).read_text()
            self.assertEqual(re.findall(r'(?:from|import) +' + re.escape(SOURCE_PACKAGE) + r'\b[^\n]*', source), [], name)
        runner = (PACKAGE / 'runner.py').read_text()
        self.assertIn(f"REFERENCE_RUNNER = '{SOURCE_PACKAGE}.runner'", runner)
        self.assertEqual(runner.count(SOURCE_PACKAGE + '.runner'), 1)

    def test_there_is_no_copy_of_the_force_module(self):
        self.assertFalse((PACKAGE / 'forces.py').exists())
        for name in MODULES:
            self.assertNotIn('def standardise', (PACKAGE / name).read_text(), name)

    def test_the_reference_test_really_reaches_the_fork_source(self):
        # Without this the exemption above would also hide a reference test that stopped testing anything.
        source = (PACKAGE / 'tests/test_m2_reference.py').read_text()
        self.assertIn(SOURCE_PACKAGE + '.contract', source)
        self.assertIn('experiments.xvla_p2_' + 'force_box.contract', source)

    def test_no_residue_names_survive_anywhere_in_the_package(self):
        for path in package_files():
            source = path.read_text()
            for needle in RESIDUE:
                self.assertNotIn(needle, source, f'{path.relative_to(PACKAGE)} still names {needle!r}')

    def test_readme_describes_this_package(self):
        # An absence-only residue check passes on an empty README; this one does not.
        readme = (PACKAGE / 'README.md').read_text()
        c = importlib.import_module('experiments.xvla_p2_force_box.contract')
        for needle in (c.SEAL_RECORD['fingerprint'], 'props_p2_20261013', '--equivalence', '--stop-at',
                       'xvla_p2_force_box_v1', 'SOURCE_FILES', c.forces.SIDECAR_RECORD['fingerprint'],
                       'p2_signals_20261030', 'IDENTITY', 'eval_adapter', 'zero-equivalence', 'magenta',
                       'colour_check.json', 'overlay_samples', c.boxes.TABLE_SHA256, 'OVERLAY'):
            self.assertIn(needle, readme)
        self.assertGreater(len(readme), 3000)


class ScheduleContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_p2_force_box.contract')

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
        self.c = importlib.import_module('experiments.xvla_p2_force_box.contract')

    def test_ten_instructions_are_expected(self):
        self.assertEqual(len(self.c.INSTRUCTIONS), 10)

    def test_planned_instructions_cover_every_episode_and_every_task(self):
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('Props P2 parent not mounted')
        planned = self.c.planned_instructions()
        self.assertEqual(len(planned), 540)
        self.assertEqual(set(planned), set(self.c.INSTRUCTIONS))

    def test_training_split_contains_every_instruction(self):
        """A split that accidentally held an instruction out would make its colours untestable."""
        if not (self.c.PARENT/'plan.json').exists():
            self.skipTest('Props P2 parent not mounted')
        planned = self.c.planned_instructions()
        for name, ids in (('train', self.c.TRAIN), ('validation', self.c.VAL)):
            self.assertEqual(set(planned[i] for i in ids), set(self.c.INSTRUCTIONS), name)

    def test_task_index_map_rejects_a_single_task_table(self):
        class _Tasks:
            index = ['put the lemon in the blue bin']

            def __getitem__(self, key):
                return [0]

        class _DS:
            class meta:
                tasks = _Tasks()

        with self.assertRaises(ValueError):
            self.c.task_index_map(_DS())


def loaded_repo_files(script_body):
    """Run `script_body` in a fresh interpreter that has printed json of the repo files loaded; return them as a set."""
    out = subprocess.run([sys.executable, '-c', script_body], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True)
    return {str(Path(f).resolve().relative_to(ROOT.resolve())) for f in json.loads(out.stdout.strip().splitlines()[-1])
            if f and Path(f).is_absolute() and Path(f).resolve().is_relative_to(ROOT.resolve())}


FILES = "{getattr(m, '__file__', None) or '' for m in sys.modules.values()}"


class SourceFilesTests(unittest.TestCase):
    def test_source_files_are_exactly_the_loaded_modules(self):
        """Every repo module that importing contract and runner loads is bound, and nothing else is."""
        c = importlib.import_module('experiments.xvla_p2_force_box.contract')
        loaded = loaded_repo_files("import sys, json, experiments.xvla_p2_force_box.contract, experiments.xvla_p2_force_box.runner;"
                                   f"print(json.dumps(sorted({FILES})))")
        own_extra = {'experiments/xvla_p2_force_box/README.md'} | {
            f'experiments/xvla_p2_force_box/tests/{name}' for name in BOUND_TESTS}
        reference_only = set(c.REFERENCE_FILES) - loaded  # M2's contract and runner: loaded only inside the zero-equivalence gate
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own_extra - reference_only
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        self.assertEqual(sorted(reference_only), ['experiments/xvla_p2_force/contract.py', 'experiments/xvla_p2_force/runner.py'])
        for name in own_extra:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)
        self.assertEqual(len(c.SOURCE_FILES), len(set(c.SOURCE_FILES)))

    def test_importing_the_reference_runner_loads_exactly_the_bound_reference_files(self):
        """The gate proves equality with M2's loop; the bytes of that loop must be exactly what is bound."""
        c = importlib.import_module('experiments.xvla_p2_force_box.contract')
        script = ("import sys, json, importlib, experiments.xvla_p2_force_box.contract, experiments.xvla_p2_force_box.runner as r;"
                  f"files = lambda: {FILES};"
                  "before = files(); importlib.import_module(r.REFERENCE_RUNNER);"
                  "print(json.dumps(sorted(files() - before)))")
        added = loaded_repo_files(script)
        before = loaded_repo_files("import sys, json, experiments.xvla_p2_force_box.contract, experiments.xvla_p2_force_box.runner;"
                                   f"print(json.dumps(sorted({FILES})))")
        self.assertEqual(sorted(added), sorted(set(c.REFERENCE_FILES) - before))
        self.assertTrue(set(c.REFERENCE_FILES) <= before | added)

    def test_the_rollout_side_is_not_bound_by_the_training_contract(self):
        """Fixing the evaluation adapter or the comparison after training must never invalidate the training binding."""
        c = importlib.import_module('experiments.xvla_p2_force_box.contract')
        for name in c.SOURCE_FILES:
            for rollout in ('eval_adapter', 'eval_tests', 'eval_compare'):
                self.assertNotIn(rollout, name)
        for entry in (PACKAGE/'tests').glob('*.py'):
            self.assertIn(f'experiments/xvla_p2_force_box/tests/{entry.name}', c.SOURCE_FILES)
        self.assertFalse((PACKAGE/'tests/__init__.py').exists())


if __name__ == '__main__':
    unittest.main()
