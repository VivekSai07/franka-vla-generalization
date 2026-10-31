"""The zero-equivalence gate: pure parts (the forcing hook, the proof's validator, the CLI, the wiring). Admission image, CPU.

The GPU part, 30 updates of P2's loop against 30 of this loop with the 8 extras forced to zero, runs as
`runner --zero-equivalence`; its mutation check (one column left un-zeroed must be reported unequal) is recorded in the status log.
"""
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class ForcingHookTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_p2_force.runner')
        self.c = self.r.c

    def batch(self):
        generator = torch.Generator().manual_seed(3)
        return {'observation.state': torch.randn(4, 17, generator=generator)+5., 'action': torch.randn(4, 30, 8, generator=generator),
                'task': 'x'}

    def test_columns_9_to_16_are_zero_after_normalisation_and_nothing_else_changes(self):
        batch = self.batch()
        out = self.r.zero_extras(lambda raw: batch)('raw')
        state = out['observation.state']
        self.assertEqual(state.shape, (4, 17))
        self.assertEqual(state[:, 9:17].abs().sum().item(), 0.)
        self.assertTrue(torch.equal(state[:, :9], batch['observation.state'][:, :9]))
        self.assertTrue(torch.equal(out['action'], batch['action']))
        self.assertEqual(out['task'], 'x')
        self.assertEqual((self.c.JOINT_DIM, self.c.REAL_STATE_DIM), (9, 17))

    def test_the_hook_does_not_write_the_batch_it_was_given(self):
        batch = self.batch()
        before = batch['observation.state'].clone()
        self.r.zero_extras(lambda raw: batch)('raw')
        self.assertTrue(torch.equal(batch['observation.state'], before))

    def test_the_hook_passes_the_raw_rows_through_to_the_processor(self):
        seen = []
        self.r.zero_extras(lambda raw: seen.append(raw) or self.batch())('the rows')
        self.assertEqual(seen, ['the rows'])

    def test_a_state_that_is_not_n_by_17_is_refused(self):
        for state in (torch.zeros(4, 9), torch.zeros(4, 20), torch.zeros(17), torch.zeros(4, 3, 17)):
            with self.assertRaises(ValueError):
                self.r.zero_extras(lambda raw: {'observation.state': state})('raw')


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_p2_force.contract')
        self.good = dict(status='complete', binding={'b': 1}, updates=30, metrics_equal=True, model_state_equal=True,
                         optimizer_state_equal=True, extras_forced_to_zero=list(range(9, 17)))

    def test_an_exact_proof_passes(self):
        self.c.validate_zero_equivalence(self.good, {'b': 1})

    def test_each_inexact_or_stale_proof_is_refused(self):
        for change in ({'metrics_equal': False}, {'model_state_equal': False}, {'optimizer_state_equal': False},
                       {'updates': 29}, {'binding': {'b': 2}}, {'status': 'failed'}, {'extras_forced_to_zero': list(range(9, 16))},
                       {'extras_forced_to_zero': list(range(8, 17))}, {'extras_forced_to_zero': []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.c.validate_zero_equivalence({**self.good, **change}, {'b': 1})
        for missing in self.good:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.c.validate_zero_equivalence({k: v for k, v in self.good.items() if k != missing}, {'b': 1})

    def test_the_output_directory_is_its_own_and_cannot_collide(self):
        c = self.c
        self.assertEqual(c.ZERO_EQUIVALENCE.name, 'xvla_p2_force_v1_zeroequivalence')
        others = {c.OUTPUT, c.SCRATCH, c.PROOF, c.EQUIVALENCE, c.ADMISSION}
        self.assertNotIn(c.ZERO_EQUIVALENCE, others)

    def test_the_reference_files_are_the_p2_runner_and_what_it_adds_and_are_bound(self):
        c = self.c
        self.assertEqual(c.REFERENCE_FILES, ('experiments/xvla_props_p2/__init__.py', 'experiments/xvla_props_p2/contract.py',
                                             'experiments/xvla_props_p2/runner.py'))
        for name in c.REFERENCE_FILES:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue(c.source_path(name).is_file())


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_p2_force.runner')
        self.source = (Path(__file__).parents[1]/'runner.py').read_text()

    def test_cli(self):
        r = self.r
        self.assertTrue(r.parse_args(['--zero-equivalence']).zero_equivalence)
        for argv in (['--zero-equivalence', '--resume'], ['--zero-equivalence', '--train'], ['--zero-equivalence', '--equivalence'],
                     ['--zero-equivalence', '--stop-at', '1'], ['--zero', '--resume'], ['--zero-equiv']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                r.parse_args(argv)

    def test_train_requires_the_zero_equivalence_proof_for_the_same_binding(self):
        self.assertIn("c.validate_zero_equivalence(json.loads((c.ZERO_EQUIVALENCE/'result.json').read_text()), binding)",
                      self.source)

    def test_the_reference_runner_is_named_once_and_imported_lazily(self):
        self.assertEqual(self.r.REFERENCE_RUNNER, 'experiments.xvla_props_' + 'p2.runner')
        self.assertEqual(self.source.count('REFERENCE_RUNNER = '), 1)
        self.assertEqual(self.source.count('importlib.import_module(REFERENCE_RUNNER)'), 1)
        top = self.source.split('def parse_args')[0]
        self.assertNotIn('xvla_props_', top.replace('xvla_props_' + 'p2.runner', ''))

    def test_both_loops_run_the_same_update_function_and_the_forcing_hook_is_looked_up_at_call_time(self):
        self.assertIn('zero_extras(pre)', self.source)
        self.assertIn('mod.one_update(update, policy, params, optimizer, pre, fetch, mod.verify_precision_fast)', self.source)
        self.assertEqual(self.source.count('optimizer.step()'), 1)  # still only in `one_update`

    def test_the_main_dispatch_runs_the_gate_without_training(self):
        r = self.r
        with patch.object(r, 'zero_equivalence') as gate, patch.object(r, 'train') as train:
            r.main(['--zero-equivalence'])
        gate.assert_called_once_with()
        train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
