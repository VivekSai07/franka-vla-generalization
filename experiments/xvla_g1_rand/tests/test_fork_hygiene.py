"""Fork hygiene for the X-VLA G1 package.

Forked from `experiments.xvla_p2_force` (M2). Two classes of bug have shipped from forks before: a name left dangling after a
rename (#24), and tests that import the *source* package and therefore pass while the fork is broken (#20). Both are statically
detectable. The only files that may reach into M2 are `tests/test_m2_reference.py` (it compares the recipe with M2's),
`tests/test_m2_equivalence.py` (it tests the gate that runs M2's runner), this file, and the runner, which names M2's runner once
(`REFERENCE_RUNNER`) and imports it lazily, inside the M2-equivalence gate; the files that import loads are bound
(`contract.REFERENCE_FILES`). `contract.validate_m2_equivalence` repeats that one name, checked equal here.
"""
import ast
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

from test_contract import PLANNED

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MODULES = ('contract.py', 'forces.py', 'runner.py')
BOUND_TESTS = ('test_contract.py', 'test_runner.py', 'test_fork_hygiene.py', 'test_forces.py', 'test_signals_join.py',
               'test_inputs.py', 'test_m2_reference.py', 'test_m2_equivalence.py')
M2_TESTS = ('test_fork_hygiene.py', 'test_m2_reference.py', 'test_m2_equivalence.py')  # may name M2
TESTS = tuple('tests/'+name for name in BOUND_TESTS if name not in M2_TESTS)
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}
M2_PACKAGE = 'experiments.xvla_p2_' + 'force'
# M2's output names and purpose strings, and the old gate: no reason to appear anywhere else in the package.
RESIDUE = ('xvla_p2_' + 'force_v1', 'xvla_p2_' + 'force_preflight', "'xvla_p2_" + "force'", 'xvla_props_' + 'p2', '_zero' + 'equivalence')
# Words the G1 package must not use except where they are M2's, or S0's array checker (a module that is called that).
RETIRED = ('sidecar', 'SIDECAR_RECORD', 'zero_equivalence', 'props_p2_data')
ALLOWED_RETIRED = (r'experiments/props_p2_data/\w+\.py', r'experiments/vla_inputs_signals/sidecar\.py',
                   r'outputs/props_p2_data/props_p2_20261013',  # the mount M2's own loader reads in the M2 gate
                   r'from experiments\.vla_inputs_signals import sidecar as s0_sidecar', r's0_sidecar')


def python_files():
    return [path for path in sorted(PACKAGE.rglob('*.py')) if '__pycache__' not in path.parts]


def package_files():
    return [path for path in sorted(PACKAGE.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.md')
            and path.name not in M2_TESTS]


def exempt(path):
    return path.parent.name == 'tests' and path.name in M2_TESTS


def docstring_nodes(tree):
    return [node.body[0] for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)]


def imported_names(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [(node.module or '')]+[(node.module or '')+'.'+alias.name for alias in node.names]
        elif isinstance(node, ast.Call):
            function = node.func
            called = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', '')
            if called in ('import_module', '__import__') and node.args and isinstance(node.args[0], ast.Constant):
                names.append(str(node.args[0].value))
    return names


def string_constants(tree):
    skipped = {id(node.value) for node in docstring_nodes(tree)}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skipped]


def identifiers(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.alias):
            names.append(node.name)
    return names


def text_without_m2_reference(path):
    """The file's text without what is allowed to name the retired words: the M2 reference docstring and bound-file entries."""
    text = path.read_text()
    if path.name == 'runner.py':  # the gate's docstring describes M2's own inputs
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'm2_equivalence':
                text = text.replace(ast.get_source_segment(text, node.body[0]), '')
    for pattern in ALLOWED_RETIRED:
        text = re.sub(pattern, '', text)
    return text


def strip_allowed(text):
    for pattern in ALLOWED_RETIRED:
        text = re.sub(pattern, '', text)
    return text


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

    def test_the_symtable_scan_finds_a_dangling_name(self):
        """The scan above must be able to fail: a function reading a name nothing binds is reported."""
        table = symtable.symtable('def f():\n    return dangling\n', 'x', 'exec')
        function = table.get_children()[0]
        symbol = next(s for s in function.get_symbols() if s.get_name() == 'dangling')
        self.assertTrue(symbol.is_global() and not symbol.is_assigned())
        self.assertNotIn('dangling', set(table.get_identifiers()))


class ForkResidueTests(unittest.TestCase):
    def test_nothing_imports_or_patches_the_fork_source(self):
        for path in python_files():
            if exempt(path):
                continue
            tree = ast.parse(path.read_text())
            with self.subTest(file=str(path.relative_to(PACKAGE))):
                self.assertEqual([n for n in imported_names(tree) if n.startswith(M2_PACKAGE)], [],
                                 f'{path.name} imports M2')
                literal = [s for s in string_constants(tree) if M2_PACKAGE in s]
                allowed = 1 if path.name in ('runner.py', 'contract.py') and path.parent == PACKAGE else 0
                self.assertEqual(len(literal), allowed, f'{path.name} names M2: {literal}')
                self.assertEqual(M2_PACKAGE in path.read_text(), allowed == 1, f'{path.name} names M2 outside a string constant')

    def test_m2_is_named_once_by_the_runner_once_by_the_contract_and_the_two_agree(self):
        runner_source = (PACKAGE/'runner.py').read_text()
        contract_source = (PACKAGE/'contract.py').read_text()
        runner = importlib.import_module('experiments.xvla_g1_rand.runner')
        self.assertEqual(runner.REFERENCE_RUNNER, M2_PACKAGE+'.runner')
        self.assertEqual(runner_source.count(M2_PACKAGE), 1)
        self.assertIn(f"REFERENCE_RUNNER = '{M2_PACKAGE}.runner'", runner_source)
        self.assertEqual([s for s in string_constants(ast.parse(contract_source)) if M2_PACKAGE in s], [M2_PACKAGE+'.runner'])
        # in the contract it is the one string `validate_m2_equivalence` requires of the proof's `reference`
        gate = next(n for n in ast.walk(ast.parse(contract_source)) if isinstance(n, ast.FunctionDef) and n.name == 'validate_m2_equivalence')
        self.assertEqual([s for s in string_constants(gate) if M2_PACKAGE in s], [M2_PACKAGE+'.runner'])

    def test_the_reference_runner_is_imported_only_inside_the_m2_equivalence_gate(self):
        tree = ast.parse((PACKAGE/'runner.py').read_text())
        sites = []
        for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'import_module':
                    sites.append(function.name)
        self.assertEqual(sorted(set(sites)), ['m2_equivalence'])
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual([m for n in top_level for m in imported_names(ast.Module(body=[n], type_ignores=[]))
                          if 'xvla_p2' in m or 'xvla_props' in m], [])

    def test_the_m2_tests_really_reach_m2(self):
        # Without this the exemption above would also hide tests that stopped testing anything.
        for name, needles in (('test_m2_reference.py', ('.contract', '.runner')), ('test_m2_equivalence.py', ('.contract',))):
            source = (PACKAGE/'tests'/name).read_text()
            for needle in needles:
                self.assertIn("'" + M2_PACKAGE + needle + "'", source, (name, needle))

    def test_no_residue_names_survive_anywhere_in_the_package(self):
        for path in package_files():
            source = path.read_text()
            for needle in RESIDUE:
                self.assertNotIn(needle, source, f'{path.relative_to(PACKAGE)} still names {needle!r}')

    def test_the_retired_words_appear_only_where_they_belong_to_m2_or_to_s0s_checker(self):
        for path in package_files():
            if path.suffix == '.md' or path.parent == PACKAGE:
                text = text_without_m2_reference(path)
                for needle in RETIRED:
                    self.assertNotIn(needle.lower(), text.lower(), f'{path.relative_to(PACKAGE)} still uses {needle!r}')

    def test_no_test_uses_a_retired_name_as_an_identifier(self):
        for path in python_files():
            if path.parent.name != 'tests' or exempt(path):
                continue
            offenders = [n for n in identifiers(ast.parse(path.read_text()))
                         if re.search('sidecar|zero_equiv|zero_extras|props_p2', n, re.I)]
            self.assertEqual(offenders, [], path.name)

    def test_the_scans_can_fail(self):
        stripped = strip_allowed('from experiments.vla_inputs_signals import sidecar as s0_sidecar; s0_sidecar.f(); experiments/props_p2_data/plan.py')
        self.assertNotIn('sidecar', stripped)  # S0's checker and the bound-file entries are allowed by name ...
        self.assertNotIn('props_p2_data', stripped)
        self.assertIn('sidecar', strip_allowed('verify_sidecar()'))  # ... any other use is not
        self.assertIn('props_p2_data', strip_allowed('from experiments.props_p2_data import plan'))
        tree = ast.parse('import experiments.xvla_p2_' + 'force.runner\nx = "experiments.xvla_p2_' + 'force.contract"')
        self.assertEqual([n for n in imported_names(tree) if n.startswith(M2_PACKAGE)], [M2_PACKAGE+'.runner'])
        self.assertEqual([s for s in string_constants(tree) if M2_PACKAGE in s], [M2_PACKAGE+'.contract'])
        call = ast.parse("importlib.import_module('experiments.xvla_p2_" + "force.runner')")
        self.assertEqual(imported_names(call), [M2_PACKAGE+'.runner'])

    def test_readme_describes_this_package(self):
        # An absence-only residue check passes on an empty README; this one does not.
        readme = (PACKAGE / 'README.md').read_text()
        c = importlib.import_module('experiments.xvla_g1_rand.contract')
        needles = (c.SEAL_RECORD['fingerprint'], c.forces.CONSTANTS_SHA256, 'g1_20261110', 'xvla_g1_rand_v1', 'SOURCE_FILES', 'IDENTITY',
                   'eval_adapter', 'gen_g1_data', 'xvla_p2_force', '--check', '--preflight', '--resume-proof', '--equivalence',
                   '--m2-equivalence', '--train', '--resume', '--stop-at', 'p2_signals_20261030', 'props_p2_20261013',
                   f'{c.TOTAL_UPDATES:,}', f'{c.TOTAL_UPDATES*8:,}', f'{c.COMPLETE_STARTS:,}', f'{c.TRAINING_ROWS:,}', '0.89',
                   '203,040', '5.0', *(c.OUTPUT.name+suffix for suffix in ('_admission', '_preflight', '_resumeproof', '_equivalence', '_m2equivalence')))
        for needle in needles:
            self.assertIn(needle, readme)
        # what is true of M2's data and not of this package's must not stand as a claim about this one
        for stale in ('244,080', '27,120', '216,960', '540 episodes', 'zero_equivalence'):
            self.assertNotIn(stale, readme)
        self.assertGreater(len(readme), 3000)
        self.assertLess(len(readme.splitlines()), 130)
        for path in re.findall(r'`((?:experiments|scripts|docs)/[\w./-]+?\.(?:py|md|sh))`', readme):  # every file it names exists
            self.assertTrue((ROOT/path).is_file(), path)


class ScheduleContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')

    def test_schedule_is_the_dense_half_epoch_grid_of_m2(self):
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

    def test_exposure_is_under_one_epoch_by_construction_not_five(self):
        """Equal updates, not equal exposure: 126,900 x 8 = 1,015,200 starts drawn, 0.89 epoch of 1,142,100."""
        c = self.c
        self.assertEqual(c.TOTAL_UPDATES*8, 1_015_200)
        self.assertEqual(c.COMPLETE_STARTS, 1_142_100)
        self.assertLess(c.TOTAL_UPDATES*8, c.COMPLETE_STARTS)
        self.assertNotEqual(c.TOTAL_UPDATES*8, c.COMPLETE_STARTS*5)


class InstructionContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')

    def test_ten_instructions_are_expected(self):
        self.assertEqual(len(self.c.INSTRUCTIONS), 10)

    def test_the_plan_covers_every_episode_and_every_task(self):
        self.assertEqual(len(PLANNED), self.c.EPISODES)
        self.assertEqual(set(PLANNED), set(self.c.INSTRUCTIONS))

    def test_training_split_contains_every_instruction(self):
        """A split that accidentally held an instruction out would make its colours untestable."""
        for name, ids in (('train', self.c.TRAIN), ('validation', self.c.VAL)):
            self.assertEqual(set(PLANNED[i] for i in ids), set(self.c.INSTRUCTIONS), name)

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


def loaded_files(script):
    out = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True)
    loaded = set()
    for file in json.loads(out.stdout.strip().splitlines()[-1]):
        if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(ROOT.resolve()):
            loaded.add(str(Path(file).resolve().relative_to(ROOT.resolve())))
    return loaded


class SourceFilesTests(unittest.TestCase):
    def test_source_files_are_exactly_the_loaded_modules(self):
        """Every repo module that importing contract and runner loads is bound, and nothing else is."""
        c = importlib.import_module('experiments.xvla_g1_rand.contract')
        loaded = loaded_files("import sys, json, experiments.xvla_g1_rand.contract, experiments.xvla_g1_rand.runner;"
                              "print(json.dumps(sorted({getattr(m, '__file__', None) or '' for m in sys.modules.values()})))")
        own_extra = {'experiments/xvla_g1_rand/README.md'} | {f'experiments/xvla_g1_rand/tests/{name}' for name in BOUND_TESTS}
        listed = {name for name in c.SOURCE_FILES if not name.startswith('installed/')} - own_extra - set(c.REFERENCE_FILES)
        self.assertEqual(sorted(loaded - listed), [], 'loaded repo modules missing from SOURCE_FILES')
        self.assertEqual(sorted(listed - loaded), [], 'SOURCE_FILES entries that nothing loads')
        for name in own_extra:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue((ROOT/name).is_file(), name)
        for needed in ('experiments/gen_g1_data/plan.py', 'experiments/gen_g1_data/contract.py', 'experiments/gen_g1_data/validate.py',
                       'experiments/gen_scene/conditions.py', 'experiments/vla_inputs_signals/sidecar.py'):
            self.assertIn(needed, c.SOURCE_FILES)

    def test_importing_the_reference_runner_loads_exactly_the_bound_reference_files(self):
        """The gate proves equality with M2's loop; the bytes of that loop must be exactly what is bound."""
        c = importlib.import_module('experiments.xvla_g1_rand.contract')
        added = loaded_files("import sys, json, importlib, experiments.xvla_g1_rand.contract, experiments.xvla_g1_rand.runner as r;"
                             "files = lambda: {getattr(m, '__file__', None) or '' for m in sys.modules.values()};"
                             "before = files(); importlib.import_module(r.REFERENCE_RUNNER);"
                             "print(json.dumps(sorted(files() - before)))")
        self.assertEqual(sorted(added), sorted(c.REFERENCE_FILES))
        for name in c.REFERENCE_FILES:
            self.assertIn(name, c.SOURCE_FILES)

    def test_every_file_of_this_package_is_bound_and_nothing_else_of_it(self):
        c = importlib.import_module('experiments.xvla_g1_rand.contract')
        here = {str(path.relative_to(ROOT)) for path in PACKAGE.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        bound = {name for name in c.SOURCE_FILES if name.startswith('experiments/xvla_g1_rand/')}
        self.assertEqual(sorted(here ^ bound), [])

    def test_the_rollout_side_is_not_bound_by_the_training_contract(self):
        """Fixing the evaluation adapter after training must never invalidate the training binding."""
        c = importlib.import_module('experiments.xvla_g1_rand.contract')
        for name in c.SOURCE_FILES:
            self.assertNotIn('eval_adapter', name)
            self.assertNotIn('eval_tests', name)
        for entry in (PACKAGE/'tests').glob('*.py'):
            self.assertIn(f'experiments/xvla_g1_rand/tests/{entry.name}', c.SOURCE_FILES)
        self.assertFalse((PACKAGE/'tests/__init__.py').exists())
        self.assertFalse((PACKAGE/'eval_adapter.py').exists())


if __name__ == '__main__':
    unittest.main()
