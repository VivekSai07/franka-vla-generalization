"""Fork hygiene for the pi0.5 P2 package.

Three classes of fork bug have shipped before: a name left dangling after a rename, tests that import the *source*
package and pass while the fork is broken, and constants re-typed instead of derived. All are statically detectable.
"""
import builtins
import importlib
import json
import os
import subprocess
import symtable
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ('contract.py', 'assets.py')
OWN = ('README.md', *(f'tests/{p.name}' for p in sorted((PACKAGE/'tests').glob('*.py'))))
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
# X-VLA packages the contract is designed from; it must copy, not import them. Only the parity test may.
SOURCE_PACKAGES = ('experiments.xvla_props_' + 'p2', 'experiments.xvla_trial', 'experiments.xvla_single_task',
                   'experiments.fixed_layout')
RESIDUE = ('xvla_props_p2_' + 'v1', 'lerobot/xvla-' + 'base', 'facebook/bart-' + 'large', 'Florence')
HYGIENE_AND_PARITY = ('test_fork_hygiene.py', 'test_p2_parity.py')


def package_files():
    return [p for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md')
            and p.name not in HYGIENE_AND_PARITY]


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
                            if (sym.is_global() and not sym.is_assigned()
                                    and ident not in module_names and not hasattr(builtins, ident)):
                                offenders.append(f'{name}:{child.get_name()}() reads undefined {ident!r}')
                    walk(child)

            walk(top)
        self.assertEqual(offenders, [], '; '.join(offenders))

    def test_the_check_itself_catches_a_dangling_name(self):
        source = 'def f():\n    return missing_name\n'
        top = symtable.symtable(source, 'x', 'exec')
        (child,) = top.get_children()
        bad = [s.get_name() for s in child.get_symbols()
               if s.is_global() and not s.is_assigned() and s.get_name() not in top.get_identifiers()]
        self.assertEqual(bad, ['missing_name'])


class ForkResidueTests(unittest.TestCase):
    def test_nothing_imports_the_fork_sources(self):
        for path in package_files():
            if path.suffix != '.py':
                continue
            source = path.read_text()
            for package in SOURCE_PACKAGES:
                for banned in (f'from {package}', f'import {package}', f"'{package}.", f'"{package}.'):
                    self.assertNotIn(banned, source, f'{path.name} reaches into {package}')

    def test_no_x_vla_residue_survives(self):
        for path in package_files():
            source = path.read_text()
            for needle in RESIDUE:
                self.assertNotIn(needle, source, f'{path.relative_to(PACKAGE)} still names {needle!r}')

    def test_the_two_exempt_files_exist_and_the_parity_test_is_the_only_importer(self):
        self.assertIn('experiments.xvla_props_' + 'p2', (PACKAGE/'tests/test_p2_parity.py').read_text())
        for name in HYGIENE_AND_PARITY:
            self.assertTrue((PACKAGE/'tests'/name).is_file())

    def test_readme_describes_this_package(self):
        readme = (PACKAGE/'README.md').read_text()
        c = importlib.import_module('experiments.pi05_props_p2.contract')
        for needle in (c.SEAL_RECORD['fingerprint'], 'props_p2_20261013', 'pi05_props_p2_v1', 'SOURCE_FILES',
                       c.IMAGE, 'a229f2fb', 'dcddb970176382c0fcf4521b0c0e6fc15894dfe0', '9e55186ad36e66b95cda57bc47818d9e6237ae30'):
            self.assertIn(needle, readme)


class SourceFilesTests(unittest.TestCase):
    def loaded(self):
        script = ("import sys, json, torch\n"
                  "import experiments.pi05_props_p2.contract as c, experiments.pi05_props_p2.assets\n"
                  "from lerobot.policies.pi05.modeling_pi05 import PI05Policy\n"
                  "from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
                  "from lerobot.datasets.utils import dataset_to_policy_features\n"
                  "cfg = c.make_config('cpu')\n"
                  "stats = {k: {n: torch.zeros(w) for n in ('mean','std','min','max','q01','q99')}"
                  " for k, w in (('observation.state', 9), ('action', 8))}\n"
                  "c.make_processors(cfg, stats)\n"
                  "print(json.dumps(sorted({getattr(m, '__file__', None) or '' for m in sys.modules.values()})))")
        out = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=False)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_source_files_are_exactly_the_loaded_repo_modules_plus_this_packages_own_files(self):
        c = importlib.import_module('experiments.pi05_props_p2.contract')
        loaded = set()
        for file in self.loaded():
            if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(ROOT.resolve()):
                loaded.add(str(Path(file).resolve().relative_to(ROOT.resolve())))
        own = {f'experiments/pi05_props_p2/{name}' for name in OWN}
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        for name in own:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)

    def test_every_installed_pi05_rtc_and_processor_module_that_loads_is_listed_and_exists(self):
        import lerobot
        c = importlib.import_module('experiments.pi05_props_p2.contract')
        base = Path(lerobot.__file__).parent.resolve()
        loaded = {str(Path(f).resolve().relative_to(base)) for f in self.loaded()
                  if f and Path(f).is_absolute() and Path(f).resolve().is_relative_to(base)}
        family = {f for f in loaded if f.startswith(('policies/pi05/', 'policies/rtc/', 'processor/'))}
        listed = {n.removeprefix('installed/lerobot/') for n in c.SOURCE_FILES if n.startswith('installed/lerobot/')}
        self.assertEqual(sorted(family - listed), [])
        for name in listed:
            self.assertTrue((base/name).is_file(), name)
        c.source_hashes()

    def test_every_test_file_is_bound_and_the_evaluation_adapter_is_not(self):
        c = importlib.import_module('experiments.pi05_props_p2.contract')
        for path in (PACKAGE/'tests').glob('*.py'):
            self.assertIn(f'experiments/pi05_props_p2/tests/{path.name}', c.SOURCE_FILES)
        self.assertFalse([n for n in c.SOURCE_FILES if 'eval_adapter' in n or 'eval_tests' in n])


if __name__ == '__main__':
    unittest.main()
