"""The zero-equivalence gate: pure parts (the overlay-identity hook, the proof's validator, the CLI, the wiring). Admission image, CPU.

The GPU part, 30 updates of M2's loop on M2's own data against 30 of this loop with the overlay replaced by the identity, runs as
`runner --zero-equivalence`; the same gate then runs this loop with the overlay ON and must report it unequal (the mutation
check, recorded in `zero_equivalence.json`). Together they show the drawn box is the only difference from A2.
"""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class IdentityHookTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_p2_force_box.runner')
        self.c = self.r.c

    def test_inside_the_scope_the_loader_draws_nothing_and_returns_the_row_itself(self):
        c = self.c
        row = {'observation.images.image': torch.zeros(3, 512, 512)}
        with self.r.overlay_identity():
            self.assertIsNot(c.OVERLAY, c.overlay.draw_frame)
            self.assertIs(c.OVERLAY(row, 'a box that is never read'), row)

    def test_the_real_overlay_is_restored_afterwards_and_when_the_body_raises(self):
        c = self.c
        original = c.OVERLAY
        self.assertIs(original, c.overlay.draw_frame)
        with self.r.overlay_identity():
            pass
        self.assertIs(c.OVERLAY, original)
        with self.assertRaises(RuntimeError), self.r.overlay_identity():
            raise RuntimeError('body failed')
        self.assertIs(c.OVERLAY, original)

    def test_the_hook_is_looked_up_at_call_time_by_the_loader(self):
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn('return OVERLAY(row, boxes.target_box(episode, frame))', source)
        self.assertEqual(source.count('OVERLAY = overlay.draw_frame'), 1)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.xvla_p2_force_box.contract')
        self.good = dict(status='complete', binding={'b': 1}, updates=30, metrics_equal=True, model_state_equal=True,
                         optimizer_state_equal=True, overlay='identity', overlay_on_metrics_equal=False,
                         overlay_on_model_state_equal=False, overlay_on_optimizer_state_equal=False)

    def test_an_exact_proof_with_a_detected_mutation_passes(self):
        self.c.validate_zero_equivalence(self.good, {'b': 1})

    def test_each_inexact_or_stale_proof_is_refused(self):
        for change in ({'metrics_equal': False}, {'model_state_equal': False}, {'optimizer_state_equal': False},
                       {'updates': 29}, {'binding': {'b': 2}}, {'status': 'failed'}, {'overlay': 'magenta'},
                       {'overlay_on_metrics_equal': True}, {'overlay_on_model_state_equal': True},
                       {'overlay_on_optimizer_state_equal': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.c.validate_zero_equivalence({**self.good, **change}, {'b': 1})
        for missing in self.good:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.c.validate_zero_equivalence({k: v for k, v in self.good.items() if k != missing}, {'b': 1})

    def test_the_output_directory_is_its_own_and_cannot_collide(self):
        c = self.c
        self.assertEqual(c.ZERO_EQUIVALENCE.name, 'xvla_p2_force_box_v1_zeroequivalence')
        others = {c.OUTPUT, c.SCRATCH, c.PROOF, c.EQUIVALENCE, c.ADMISSION}
        self.assertNotIn(c.ZERO_EQUIVALENCE, others)

    def test_the_reference_files_are_m2s_runner_and_what_it_adds_and_are_bound(self):
        c = self.c
        self.assertEqual(c.REFERENCE_FILES, ('experiments/xvla_p2_force/__init__.py', 'experiments/xvla_p2_force/contract.py',
                                             'experiments/xvla_p2_force/runner.py'))
        for name in c.REFERENCE_FILES:
            self.assertIn(name, c.SOURCE_FILES)
            self.assertTrue(c.source_path(name).is_file())


class GateFlowTests(unittest.TestCase):
    """`zero_equivalence()` end to end with the GPU loop replaced by a stand-in that reports what it was given.

    Reference (M2's runner on M2's dataset class): result 'A'. This loop with the overlay the identity: 'A' (or 'B' if told to differ).
    This loop with the real overlay on: 'C' (or 'A' if told the mutation goes undetected).
    """

    def run_gate(self, tmp, identity_differs=False, mutation_undetected=False):
        from types import SimpleNamespace
        r = importlib.import_module('experiments.xvla_p2_force_box.runner')
        c = r.c
        reference = SimpleNamespace(__name__='reference', c=SimpleNamespace(JoinedDataset=lambda raw: ('m2 dataset over', raw)))
        calls = []

        def fake_thirty(mod, ds, starts, stats, wrap):
            overlay_on = c.OVERLAY is c.overlay.draw_frame
            calls.append(dict(runner='reference' if mod is reference else 'box', overlay_on=overlay_on, ds=ds))
            tag = 'A'
            if mod is not reference:
                tag = ('A' if mutation_undetected else 'C') if overlay_on else ('B' if identity_differs else 'A')
            return dict(rows=[{'loss': tag}], model=tag, optimizer=tag, seconds_per_update=0.1)

        def claim(path):
            path.mkdir()

        patches = [patch.object(c, 'ZERO_EQUIVALENCE', Path(tmp)/'zero'), patch.object(c, 'claim_output', side_effect=claim),
                   patch.object(c, 'prepare_data', return_value=(SimpleNamespace(ds='raw dataset'), [0], {}, {'b': 1})),
                   patch.object(r.importlib, 'import_module', return_value=reference),
                   patch.object(r, 'thirty_updates', side_effect=fake_thirty), patch.object(r, 'require_resources'),
                   patch.object(r.torch.cuda, 'is_available', return_value=True),
                   patch.object(r.torch.cuda, 'is_bf16_supported', return_value=True),
                   patch.object(r.torch, 'use_deterministic_algorithms')]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            try:
                r.zero_equivalence()
                error = None
            except ValueError as exc:
                error = exc
        return c, calls, error

    def test_the_three_runs_use_the_right_dataset_and_the_overlay_state_and_the_proof_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, calls, error = self.run_gate(tmp)
            self.assertIsNone(error)
            self.assertEqual([(x['runner'], x['overlay_on']) for x in calls[1:]], [('box', False), ('box', True)])
            self.assertEqual(calls[0]['runner'], 'reference')
            self.assertEqual(calls[0]['ds'], ('m2 dataset over', 'raw dataset'))  # M2's dataset class over the raw rows
            self.assertEqual(type(calls[1]['ds']).__name__, 'SimpleNamespace')  # this loop gets the fork's dataset object
            report = json.loads((Path(tmp)/'zero/result.json').read_text())
            self.assertEqual((report['status'], report['overlay']), ('complete', 'identity'))
            self.assertTrue(report['metrics_equal'] and report['model_state_equal'] and report['optimizer_state_equal'])
            self.assertFalse(report['overlay_on_metrics_equal'] or report['overlay_on_model_state_equal']
                             or report['overlay_on_optimizer_state_equal'])
            self.assertIs(c.OVERLAY, c.overlay.draw_frame)  # restored

    def test_an_identity_run_that_differs_from_m2_fails_the_gate_and_records_the_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, calls, error = self.run_gate(tmp, identity_differs=True)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(json.loads((Path(tmp)/'zero/result.json').read_text())['status'], 'failed')
            self.assertIs(c.OVERLAY, c.overlay.draw_frame)

    def test_an_undetected_overlay_on_mutation_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, calls, error = self.run_gate(tmp, mutation_undetected=True)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(json.loads((Path(tmp)/'zero/result.json').read_text())['status'], 'failed')


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('experiments.xvla_p2_force_box.runner')
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
        self.assertEqual(self.r.REFERENCE_RUNNER, 'experiments.xvla_p2_' + 'force.runner')
        self.assertEqual(self.source.count('REFERENCE_RUNNER = '), 1)
        self.assertEqual(self.source.count('importlib.import_module(REFERENCE_RUNNER)'), 1)

    def test_the_reference_runs_on_m2s_own_dataset_class_and_this_loop_on_the_forks(self):
        # The reference must not touch this package's loader or overlay, or the gate would compare the fork with itself.
        self.assertIn('reference.c.JoinedDataset(ds.ds)', self.source)
        self.assertIn('with overlay_identity():', self.source)

    def test_all_loops_run_the_same_update_function(self):
        self.assertIn('mod.one_update(update, policy, params, optimizer, pre, fetch, mod.verify_precision_fast)', self.source)
        self.assertEqual(self.source.count('optimizer.step()'), 1)  # still only in `one_update`

    def test_the_gate_runs_the_overlay_on_mutation_and_requires_it_unequal(self):
        self.assertIn("overlay_on_metrics_equal=a['rows'] == mutant['rows']", self.source)
        self.assertIn("overlay_on_model_state_equal=a['model'] == mutant['model']", self.source)
        self.assertIn("overlay_on_optimizer_state_equal=a['optimizer'] == mutant['optimizer']", self.source)

    def test_the_main_dispatch_runs_the_gate_without_training(self):
        r = self.r
        with patch.object(r, 'zero_equivalence') as gate, patch.object(r, 'train') as train:
            r.main(['--zero-equivalence'])
        gate.assert_called_once_with()
        train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
