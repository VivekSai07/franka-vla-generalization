"""Fork hygiene for the X-VLA P2 continuation pilots.

The package imports M2 (`experiments.xvla_p2_force`: constants, helpers and the force transform) read-only and copies its
runner plumbing. Two classes of bug have shipped from forks before: a name left dangling after a copy (#24), and tests that
import the *source* package and therefore pass while the fork is broken (#20). Both are statically detectable.
"""
import builtins
import importlib
import json
import os
import re
import subprocess
import symtable
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ('contract.py', 'runner.py')
BOUND_TESTS = ('test_contract.py', 'test_runner.py', 'test_fork_hygiene.py')
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
# M2's output names and purpose strings: no reason to appear in the pilot (the README may name M2 as what it reads).
RESIDUE = ('xvla_p2_force' + '_v1', 'xvla_p2_force' + '_preflight', "'xvla_p2_force" + "'", 'xvla_props_' + 'p2_v1',
           'xvla_props_' + 'p2_preflight')
# The original P2 package is neither read nor patched by the pilot.
P2_PACKAGE = 'experiments.xvla_props_' + 'p2'


def package_files():
    return [path for path in sorted(PACKAGE.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.md')
            and path.name != 'test_fork_hygiene.py']


class UndefinedNameTests(unittest.TestCase):
    def test_no_module_reads_a_name_that_is_never_bound(self):
        offenders = []
        for name in MODULES:
            source = (PACKAGE/name).read_text()
            top = symtable.symtable(source, name, 'exec')
            module_names = set(top.get_identifiers()) | MODULE_DUNDERS

            def walk(table):
                for child in table.get_children():
                    if child.get_type() == 'function':
                        for sym in child.get_symbols():
                            ident = sym.get_name()
                            if (sym.is_global() and not sym.is_assigned() and ident not in module_names
                                    and not hasattr(builtins, ident)):
                                offenders.append(f'{name}:{child.get_name()}() reads undefined {ident!r}')
                    walk(child)

            walk(top)
        self.assertEqual(offenders, [], '; '.join(offenders))

    def test_the_check_would_see_a_dangling_name(self):
        # Without this the check above could pass on a symtable walk that never reaches a function body.
        source = 'def f():\n    return gone\n'
        top = symtable.symtable(source, 'x', 'exec')
        function = top.get_children()[0]
        names = [s.get_name() for s in function.get_symbols() if s.is_global() and not s.is_assigned()
                 and s.get_name() not in top.get_identifiers()]
        self.assertEqual(names, ['gone'])

    def test_every_module_name_the_runner_uses_from_the_contract_exists_there(self):
        c = importlib.import_module('experiments.xvla_p2_pilot.contract')
        used = set(re.findall(r'\bc\.([A-Za-z_][A-Za-z0-9_]*)', (PACKAGE/'runner.py').read_text()))
        self.assertEqual(sorted(name for name in used if not hasattr(c, name)), [])


class ResidueTests(unittest.TestCase):
    def test_nothing_imports_or_patches_the_original_p2_package(self):
        pattern = re.compile(r'(from|import) +' + re.escape(P2_PACKAGE) + r'\b|[\'"]' + re.escape(P2_PACKAGE) + r'\b')
        for path in package_files():
            if path.suffix == '.py':
                self.assertIsNone(pattern.search(path.read_text()), f'{path.name} reaches into {P2_PACKAGE}')

    def test_no_residue_names_survive_anywhere_in_the_package(self):
        for path in package_files():
            source = path.read_text()
            for needle in RESIDUE:
                self.assertNotIn(needle, source, f'{path.relative_to(PACKAGE)} still names {needle!r}')

    def test_readme_describes_this_package(self):
        # An absence-only residue check passes on an empty README; this one does not.
        readme = (PACKAGE/'README.md').read_text()
        c = importlib.import_module('experiments.xvla_p2_pilot.contract')
        for needle in (c.SEAL_RECORD['fingerprint'], 'props_p2_20261013', '--zero-equivalence', '--check', '--train',
                       '--updates', 'SOURCE_FILES', c.forces.SIDECAR_RECORD['fingerprint'], 'p2_signals_20261030',
                       'IDENTITY', 'eval_adapter', 'variants/', c.START_MODEL_SHA256, c.BOX_TABLE_SHA256,
                       'xvla_p2_force', '101,520', 'fresh', 'step_010000', 'Δ'):
            self.assertIn(needle, readme)
        self.assertGreater(len(readme), 3000)


class SourceFilesTests(unittest.TestCase):
    def test_source_files_are_exactly_the_loaded_modules(self):
        """Every repo module that importing contract and runner loads is bound, and nothing else is."""
        c = importlib.import_module('experiments.xvla_p2_pilot.contract')
        script = ("import sys, json, experiments.xvla_p2_pilot.contract, experiments.xvla_p2_pilot.runner;"
                  "print(json.dumps(sorted({getattr(m, '__file__', None) or '' for m in sys.modules.values()})))")
        out = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True)
        loaded = set()
        for file in json.loads(out.stdout.strip().splitlines()[-1]):
            if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(ROOT.resolve()):
                loaded.add(str(Path(file).resolve().relative_to(ROOT.resolve())))
        own_extra = {'experiments/xvla_p2_pilot/README.md'} | {f'experiments/xvla_p2_pilot/tests/{name}' for name in BOUND_TESTS}
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own_extra
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        for name in own_extra:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)

    def test_the_rollout_side_and_the_variants_are_not_bound_by_the_training_contract(self):
        """Fixing the evaluation adapter, or adding a variant, must never invalidate the training binding."""
        c = importlib.import_module('experiments.xvla_p2_pilot.contract')
        for name in c.SOURCE_FILES:
            for forbidden in ('eval_adapter', 'eval_tests', 'variants/', 'summary.py'):
                self.assertNotIn(forbidden, name)
        for entry in (PACKAGE/'tests').glob('*.py'):
            self.assertIn(f'experiments/xvla_p2_pilot/tests/{entry.name}', c.SOURCE_FILES)
        self.assertFalse((PACKAGE/'tests/__init__.py').exists())

    def test_the_frozen_sources_the_pilot_reads_are_listed_and_never_executed_as_a_runner(self):
        c = importlib.import_module('experiments.xvla_p2_pilot.contract')
        for name in ('experiments/xvla_p2_force/contract.py', 'experiments/xvla_p2_force/forces.py'):
            self.assertIn(name, c.SOURCE_FILES)
        self.assertNotIn('experiments/xvla_p2_force/runner.py', c.SOURCE_FILES)


if __name__ == '__main__':
    unittest.main()
