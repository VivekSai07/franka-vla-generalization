"""The M2 equivalence gate: the proof's validator, the reference profile, the CLI, the wiring and the gate's flow. Admission image, CPU.

The GPU part, 30 updates of M2's own loop against 30 of this loop on M2's inputs, runs as `runner --m2-equivalence`; here its flow
is run with the two loops replaced by recorders, so what is tested is what the gate hands each loop, in what order, under which
profile, and what it records and refuses. Mutation checks of the validator, the profile and the flow are in the status log.
"""
import itertools
import inspect
import importlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_contract import FRAMES, PLANNED, frame, metadata, use_planned

_M2 = {}


def m2_planned():
    """P2's plan instructions (M2's `planned_instructions()`), from P2's plan generator; no data needed."""
    if 'planned' not in _M2:
        from experiments.props_p2_data import plan as p2_plan
        _M2['planned'] = tuple(e['instruction'] for e in p2_plan.make_plan()['episodes'])
    return _M2['planned']


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')
        self.r = importlib.import_module('experiments.xvla_g1_rand.runner')
        self.good = dict(status='complete', binding={'b': 1}, updates=30, metrics_equal=True, model_state_equal=True,
                         optimizer_state_equal=True, reference=self.r.REFERENCE_RUNNER)

    def test_an_exact_proof_passes(self):
        self.c.validate_m2_equivalence(self.good, {'b': 1})

    def test_each_inexact_or_stale_proof_is_refused(self):
        for change in ({'metrics_equal': False}, {'model_state_equal': False}, {'optimizer_state_equal': False},
                       {'updates': 29}, {'updates': 31}, {'updates': 30.0}, {'metrics_equal': 1}, {'binding': {'b': 2}},
                       {'binding': {}}, {'status': 'failed'}, {'status': 'preparing'}, {'reference': ''},
                       {'reference': 'experiments.xvla_g1_rand.runner'},  # this loop compared with itself proves nothing
                       {'reference': 'experiments.xvla_props_' + 'p2.runner'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.c.validate_m2_equivalence({**self.good, **change}, {'b': 1})
        for missing in self.good:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.c.validate_m2_equivalence({k: v for k, v in self.good.items() if k != missing}, {'b': 1})

    def test_a_stale_proof_for_another_binding_is_refused_in_both_directions(self):
        with self.assertRaises(ValueError):
            self.c.validate_m2_equivalence(self.good, {'b': 2})
        with self.assertRaises(ValueError):
            self.c.validate_m2_equivalence({**self.good, 'binding': {'b': 1, 'sources': {'x': 'y'}}}, {'b': 1})

    def test_the_old_zero_equivalence_gate_and_its_proof_are_gone(self):
        zero = dict(status='complete', binding={'b': 1}, updates=30, metrics_equal=True, model_state_equal=True,
                    optimizer_state_equal=True, extras_forced_to_zero=list(range(9, 17)))
        with self.assertRaises(ValueError):  # a zero-equivalence proof carries no reference to M2's runner
            self.c.validate_m2_equivalence(zero, {'b': 1})
        for name in ('validate_zero_equivalence', 'ZERO_EQUIVALENCE'):
            self.assertFalse(hasattr(self.c, name), name)
        for name in ('zero_equivalence', 'zero_extras', 'ZERO_EQUIVALENCE_UPDATES'):
            self.assertFalse(hasattr(self.r, name), name)

    def test_the_proof_names_the_reference_runner_the_gate_imports(self):
        self.assertEqual(self.r.REFERENCE_RUNNER, 'experiments.xvla_p2_' + 'force.runner')
        self.assertEqual(self.good['reference'], 'experiments.xvla_p2_' + 'force.runner')

    def test_the_output_directory_is_its_own_and_cannot_collide(self):
        c = self.c
        self.assertEqual(c.M2_EQUIVALENCE.name, 'xvla_g1_rand_v1_m2equivalence')
        self.assertEqual(c.M2_EQUIVALENCE.parent, c.OUTPUT.parent)
        others = {c.OUTPUT, c.SCRATCH, c.PROOF, c.EQUIVALENCE, c.ADMISSION}
        self.assertNotIn(c.M2_EQUIVALENCE, others)

    def test_the_reference_files_are_m2s_runner_and_what_it_adds_and_are_bound(self):
        c = self.c
        self.assertEqual(c.REFERENCE_FILES, ('experiments/xvla_p2_force/__init__.py', 'experiments/xvla_p2_force/contract.py',
                                             'experiments/xvla_p2_force/forces.py', 'experiments/xvla_p2_force/runner.py',
                                             'experiments/props_p2_data/contract.py', 'experiments/props_p2_data/validate.py'))
        for name in c.REFERENCE_FILES:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue(c.source_path(name).is_file())
        self.assertEqual(len(c.SOURCE_FILES), len(set(c.SOURCE_FILES)))


class ReferenceProfileTests(unittest.TestCase):
    NAMES = {'TRAIN', 'VAL', 'planned_instructions'}

    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')
        self.m2 = importlib.import_module('experiments.xvla_p2_force.contract')
        use_planned(self, self.c)  # this contract's plan instructions, as the logic tests of the package use them
        patcher = patch.object(self.m2, 'planned_instructions', return_value=m2_planned())  # P2's, without its data
        patcher.start()
        self.addCleanup(patcher.stop)
        self.differing = next(i for i in self.m2.TRAIN if PLANNED[i] != m2_planned()[i])  # premise: the two plans disagree here

    def reference(self, **changes):
        names = ('INSTRUCTIONS', 'FRAMES_PER_EPISODE', 'CHUNK_SIZE', 'SEED', 'TRAIN', 'VAL', 'planned_instructions')
        return SimpleNamespace(**{**{name: getattr(self.m2, name) for name in names}, **changes})

    def snapshot(self):
        return dict(vars(self.c))

    def changed_names(self, before, after):
        missing = object()
        return {key for key in set(before)|set(after) if before.get(key, missing) is not after.get(key, missing)}

    def test_the_profile_rebinds_exactly_the_three_names(self):
        c, m2 = self.c, self.m2
        before = self.snapshot()
        with c.reference_profile(m2):
            inside = self.snapshot()
            self.assertIs(c.TRAIN, m2.TRAIN)
            self.assertIs(c.VAL, m2.VAL)
            self.assertIs(c.planned_instructions, m2.planned_instructions)
            self.assertEqual(c.planned_instructions(), m2_planned())
        self.assertEqual(self.changed_names(before, inside), self.NAMES)
        self.assertEqual(self.changed_names(before, self.snapshot()), set())

    def test_the_names_are_restored_even_when_the_body_raises(self):
        c, m2 = self.c, self.m2
        before = self.snapshot()
        for error in (RuntimeError, ValueError, KeyboardInterrupt):
            with self.subTest(error=error), self.assertRaises(error):
                with c.reference_profile(m2):
                    self.assertIs(c.TRAIN, m2.TRAIN)
                    raise error('boom')
            self.assertEqual(self.changed_names(before, self.snapshot()), set())

    def test_a_reference_that_lacks_a_name_leaves_nothing_rebound(self):
        c = self.c
        before = self.snapshot()
        incomplete = self.reference()
        del incomplete.VAL  # TRAIN is rebound first; the failure on VAL must not leave it so
        with self.assertRaises(AttributeError):
            with c.reference_profile(incomplete):
                self.fail('the body must not run')
        self.assertEqual(self.changed_names(before, self.snapshot()), set())

    def test_a_reference_that_differs_in_more_than_the_data_is_refused_and_binds_nothing(self):
        c, m2 = self.c, self.m2
        before = self.snapshot()
        for name, value in (('INSTRUCTIONS', m2.INSTRUCTIONS[:-1]), ('INSTRUCTIONS', tuple(reversed(m2.INSTRUCTIONS))),
                            ('FRAMES_PER_EPISODE', m2.FRAMES_PER_EPISODE+1), ('CHUNK_SIZE', m2.CHUNK_SIZE+1), ('SEED', m2.SEED+1)):
            ran = []
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'more than the data'):
                with c.reference_profile(self.reference(**{name: value})):
                    ran.append(1)
            self.assertEqual(ran, [])
            self.assertEqual(self.changed_names(before, self.snapshot()), set())

    def test_a_reference_that_differs_only_in_the_data_is_accepted(self):
        c = self.c
        with c.reference_profile(self.reference(TRAIN=(0, 1), VAL=(2,))):
            self.assertEqual((c.TRAIN, c.VAL), ((0, 1), (2,)))
        self.assertEqual((c.TRAIN, c.VAL), (tuple(range(2700)), tuple(range(2700, 3000))))

    def test_under_the_profile_policy_row_accepts_m2s_episodes_and_outside_it_refuses_them(self):
        c, m2, ep = self.c, self.m2, self.differing
        row = {**frame(ep), 'task': m2_planned()[ep]}
        with self.assertRaises(ValueError):  # outside: this plan's instruction for the episode, not P2's
            c.policy_row(row, training=True)
        with c.reference_profile(m2):
            self.assertEqual(c.policy_row(row, training=True)['task'], m2_planned()[ep])
            with self.assertRaises(ValueError):  # and this plan's instruction is now the wrong one
                c.policy_row({**frame(ep)}, training=True)
            last_train, validation, beyond = len(m2.TRAIN)-1, len(m2.TRAIN), len(m2.TRAIN)+len(m2.VAL)
            c.policy_row({**frame(last_train), 'task': m2_planned()[last_train]}, training=True)
            with self.assertRaises(ValueError):  # M2's validation episodes are not training rows ...
                c.policy_row({**frame(validation), 'task': m2_planned()[validation]}, training=True)
            c.policy_row({**frame(validation), 'task': m2_planned()[validation]}, training=False)  # ... but are rollout rows
            with self.assertRaises(ValueError):  # P2 has no episode 540
                c.policy_row({**frame(beyond), 'task': PLANNED[beyond]}, training=False)
        with self.assertRaises(ValueError):
            c.policy_row(row, training=True)
        # outside the profile an episode P2 never had is this dataset's own
        c.policy_row(frame(beyond), training=True)

    def test_under_the_profile_sample_batches_accepts_m2s_starts_and_outside_it_refuses_them(self):
        c, m2 = self.c, self.m2
        m2_starts = [i*FRAMES+j for i in m2.TRAIN for j in range(FRAMES-c.CHUNK_SIZE+1)]
        g1_starts = c.indices(metadata())[1]
        self.assertEqual((len(m2_starts), len(g1_starts)), (203_040, 1_142_100))
        with self.assertRaises(ValueError):
            next(c.sample_batches(m2_starts))
        with c.reference_profile(m2):
            ours = list(itertools.islice(c.sample_batches(m2_starts), 50))
            with self.assertRaises(ValueError):
                next(c.sample_batches(g1_starts))
        self.assertEqual(ours, list(itertools.islice(m2.sample_batches(m2_starts), 50)))
        with self.assertRaises(ValueError):
            next(c.sample_batches(m2_starts))
        self.assertEqual(len(next(c.sample_batches(g1_starts))), 4)

    def test_m2_and_this_package_agree_on_what_the_two_plans_do_not_share(self):
        """Premises of the tests above: P2's split is a prefix of this one's, and the two plans are different."""
        c, m2 = self.c, self.m2
        self.assertEqual(m2.TRAIN, c.TRAIN[:len(m2.TRAIN)])
        self.assertNotEqual(m2_planned(), PLANNED[:len(m2_planned())])
        self.assertEqual(set(m2_planned()), set(PLANNED))


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_g1_rand.runner')
        self.source = (Path(__file__).parents[1]/'runner.py').read_text()

    def test_cli(self):
        r = self.r
        args = r.parse_args(['--m2-equivalence'])
        self.assertTrue(args.m2_equivalence)
        self.assertFalse(hasattr(args, 'zero_equivalence'))
        self.assertFalse(r.parse_args(['--train']).m2_equivalence)
        for argv in (['--m2-equivalence', '--resume'], ['--m2-equivalence', '--train'], ['--m2-equivalence', '--equivalence'],
                     ['--m2-equivalence', '--stop-at', '1'], ['--m2-equivalence', '--check'], ['--zero-equivalence'],
                     ['--zero-equivalence', '--m2-equivalence'], ['--m2', '--resume'], ['--m2-equiv']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                r.parse_args(argv)

    def test_train_requires_the_m2_proof_for_the_same_binding_and_the_loop_proof_too(self):
        train = inspect.getsource(self.r.train)
        gate = train.split('if not preflight:')[1].split('elif reference is not None:')[0]
        self.assertIn("c.validate_m2_equivalence(json.loads((c.M2_EQUIVALENCE/'result.json').read_text()), binding)", gate)
        self.assertIn("c.validate_equivalence(json.loads((c.EQUIVALENCE/'result.json').read_text()), binding)", gate)
        self.assertNotIn('zero', train)

    def test_the_reference_runner_is_named_once_and_imported_lazily(self):
        self.assertEqual(self.r.REFERENCE_RUNNER, 'experiments.xvla_p2_' + 'force.runner')
        self.assertEqual(self.source.count('REFERENCE_RUNNER = '), 1)
        self.assertEqual(self.source.count('importlib.import_module(REFERENCE_RUNNER)'), 1)
        top = self.source.split('def parse_args')[0]
        self.assertNotIn('xvla_p2_' + 'force', top.replace('experiments.xvla_p2_' + 'force.runner', '', 1))
        self.assertNotIn('xvla_props_', top)

    def test_both_loops_run_the_update_function_of_the_module_they_are_given(self):
        self.assertIn('mod.one_update(update, policy, params, optimizer, pre, fetch, mod.verify_precision_fast)',
                      inspect.getsource(self.r.thirty_updates))
        self.assertEqual(self.source.count('optimizer.step()'), 1)  # still only in `one_update`
        gate = inspect.getsource(self.r.m2_equivalence)
        self.assertLess(gate.index('thirty_updates(reference,'), gate.index('with c.reference_profile(reference.c):'))
        self.assertLess(gate.index('with c.reference_profile(reference.c):'), gate.index('thirty_updates(sys.modules[__name__],'))

    def test_the_main_dispatch_runs_the_gate_without_training(self):
        r = self.r
        with patch.object(r, 'm2_equivalence') as gate, patch.object(r, 'train') as train, \
                patch.object(r, 'equivalence') as loops:
            r.main(['--m2-equivalence'])
        gate.assert_called_once_with()
        train.assert_not_called()
        loops.assert_not_called()


class FlowTests(unittest.TestCase):
    """`runner.m2_equivalence()` with the two 30-update loops replaced by recorders."""

    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_g1_rand.runner')
        self.c = self.r.c
        self.m2c = importlib.import_module('experiments.xvla_p2_force.contract')
        use_planned(self, self.c)
        patcher = patch.object(self.m2c, 'planned_instructions', return_value=m2_planned())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.binding = {'b': 1}

    def loop(self, **changes):
        return {**dict(rows=[{'update': 1, 'loss': 1.5}], model='m'*64, optimizer='o'*64, seconds_per_update=.5), **changes}

    def run_gate(self, reference_result, own_result=None, cuda=True, out=None, failing=None):
        """Run the gate once in a fresh output; returns what the recorders saw. `failing` is raised by this package's loop."""
        calls, imported = [], []
        reference = SimpleNamespace(c=self.m2c)
        data = (object(), [1, 2], {'stats': 1})

        def thirty(mod, ds, starts, stats, wrap):
            calls.append(dict(mod=mod, ds=ds, starts=starts, stats=stats, wrap=wrap, profile=self.c.TRAIN is self.m2c.TRAIN,
                              plan=self.c.planned_instructions() == m2_planned()))
            if mod is not reference and failing is not None:
                raise failing
            return reference_result if mod is reference else own_result

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.out = out or Path(tmp)/'m2equivalence'
            for item in (patch.object(self.r, 'importlib', SimpleNamespace(import_module=lambda name: imported.append(name) or reference)),
                         patch.object(self.c, 'M2_EQUIVALENCE', self.out),
                         patch.object(self.c, 'current_binding', return_value=self.binding),
                         patch.object(self.c, 'load_data', side_effect=AssertionError("this package's data must not feed the gate")),
                         patch.object(self.m2c, 'load_data', return_value=data),
                         patch.object(self.r, 'thirty_updates', side_effect=thirty),
                         patch.object(self.r, 'require_resources', return_value={}),
                         patch.object(torch.cuda, 'is_available', return_value=cuda),
                         patch.object(torch.cuda, 'is_bf16_supported', return_value=True),
                         patch.object(torch, 'use_deterministic_algorithms')):
                stack.enter_context(item)
            try:
                self.r.m2_equivalence()
            finally:
                self.seen = dict(calls=calls, imported=imported, reference=reference, data=data)
                self.result = json.loads((self.out/'result.json').read_text()) if (self.out/'result.json').is_file() else None
                self.files = sorted(p.name for p in self.out.iterdir()) if self.out.is_dir() else []

    def test_equal_loops_are_recorded_as_a_proof_that_validates(self):
        self.run_gate(self.loop(), self.loop())
        report = self.result
        self.assertEqual(report['status'], 'complete')
        self.assertEqual((report['metrics_equal'], report['model_state_equal'], report['optimizer_state_equal']), (True, True, True))
        self.assertEqual((report['updates'], report['reference'], report['binding']), (30, self.r.REFERENCE_RUNNER, self.binding))
        self.c.validate_m2_equivalence(report, self.binding)
        self.assertEqual(self.seen['imported'], [self.r.REFERENCE_RUNNER])
        self.assertIn('run_claim.json', self.files)
        self.assertNotIn('failure.json', self.files)

    def test_each_loop_gets_the_same_inputs_the_reference_first_and_this_one_under_the_profile(self):
        self.run_gate(self.loop(), self.loop())
        first, second = self.seen['calls']
        self.assertIs(first['mod'], self.seen['reference'])
        self.assertIs(second['mod'], self.r)
        self.assertEqual((first['profile'], second['profile']), (False, True))  # only this loop runs on M2's profile
        self.assertEqual((first['plan'], second['plan']), (False, True))
        for position, key in enumerate(('ds', 'starts', 'stats')):
            self.assertIs(first[key], second[key])
            self.assertIs(first[key], self.seen['data'][position])
        self.assertEqual(first['wrap']('pre'), 'pre')  # neither loop's processors are wrapped
        self.assertEqual(second['wrap']('pre'), 'pre')
        self.assertEqual((self.c.TRAIN, self.c.VAL), (tuple(range(2700)), tuple(range(2700, 3000))))  # the profile ended with the gate

    def test_each_difference_between_the_loops_is_recorded_and_refused(self):
        for change in ({'rows': [{'update': 1, 'loss': 1.5000001}]}, {'model': 'x'*64}, {'optimizer': 'x'*64}):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    self.run_gate(self.loop(), self.loop(**change))
                self.assertEqual(self.result['status'], 'failed')
                self.assertIn('failure.json', self.files)
                self.assertEqual(self.c.TRAIN, tuple(range(2700)))

    def test_the_profile_is_removed_when_the_loop_raises(self):
        with self.assertRaises(RuntimeError):
            self.run_gate(self.loop(), failing=RuntimeError('loop failed'))
        self.assertEqual((self.c.TRAIN, self.c.VAL), (tuple(range(2700)), tuple(range(2700, 3000))))
        self.assertIs(self.seen['calls'][-1]['mod'], self.r)
        self.assertTrue(self.seen['calls'][-1]['profile'])
        self.assertEqual(self.result['status'], 'failed')

    def test_no_cuda_stops_the_gate_before_either_loop_runs(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA BF16 required'):
            self.run_gate(self.loop(), self.loop(), cuda=False)
        self.assertEqual(self.seen['calls'], [])
        self.assertEqual(self.result['status'], 'failed')

    def test_an_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'existing'
            out.mkdir()
            (out/'keep.txt').write_text('x')
            with self.assertRaises(FileExistsError):
                self.run_gate(self.loop(), self.loop(), out=out)
            self.assertEqual(self.seen['calls'], [])
            self.assertEqual((out/'keep.txt').read_text(), 'x')
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['keep.txt'])


if __name__ == '__main__':
    unittest.main()
