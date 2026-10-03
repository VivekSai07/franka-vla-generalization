import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.xvla_b3_robosuite_eval import evaluate, protocol


class SelectionTests(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('B3 parent not mounted')

    def test_trial_sets(self):
        self.assertEqual(len(evaluate.select_trials('screen')), 20)
        self.assertEqual(len(evaluate.select_trials('report')), 160)
        smoke = evaluate.select_trials('smoke')
        self.assertEqual(len(smoke), 2)
        self.assertEqual(smoke[0], protocol.screen_trials()[0])
        self.assertEqual(smoke[1]['kind'], 'held_out')
        with self.assertRaises(ValueError):
            evaluate.select_trials('nonsense')

    def test_cli_accepts_only_approved_modes_and_candidates(self):
        self.assertEqual(evaluate.parse_args(['--mode', 'screen', '--checkpoint', '62400']).checkpoint, 62400)
        for argv in ([], ['--mode', 'screen'], ['--mode', 'screen', '--checkpoint', '31200'],
                     ['--mode', 'train', '--checkpoint', '62400'], ['--mode', 'screen', '--checkpoint', '1'],
                     ['--mode', 'screen', '--checkpoint', '62400', '--retries', '1'],
                     ['--mod', 'screen', '--checkpoint', '62400']):
            with self.assertRaises(SystemExit):
                evaluate.parse_args(argv)

    def test_the_script_has_a_main_guard(self):
        """FAILURES #36: a forked evaluate.py without the guard imports cleanly and runs nothing."""
        source = (Path(evaluate.__file__)).read_text()
        self.assertIn("if __name__ == '__main__':\n    main()", source)

    def test_nothing_writes_under_the_dataset_or_the_training_output(self):
        source = Path(evaluate.__file__).read_text()
        self.assertNotIn('PARENT/', source.replace('protocol.PARENT', ''))
        for directory in (protocol.output_dir(m, c) for m in protocol.MODES for c in protocol.CANDIDATES):
            self.assertEqual(directory.parent, protocol.EVAL_ROOT)


class RefusalTests(unittest.TestCase):
    """run() must refuse before it creates an output directory."""

    def setUp(self):
        import mujoco
        if mujoco.__version__ != evaluate.MUJOCO_VERSION:
            self.skipTest('robosuite image only')

    def test_refuses_without_the_feasibility_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(protocol, 'EVAL_ROOT', Path(tmp)), \
                    patch.object(evaluate, 'FEASIBILITY', Path(tmp)/'feasibility/feasibility.json'):
                with self.assertRaisesRegex(RuntimeError, 'feasibility'):
                    evaluate.run('screen', 62400)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_refuses_an_incomplete_feasibility_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'feasibility.json'
            path.write_text(json.dumps(dict(keys=99, failed_layouts=[], failed_keys=[], results={})))
            with patch.object(evaluate, 'FEASIBILITY', path), patch.object(protocol, 'EVAL_ROOT', Path(tmp)):
                with self.assertRaises(ValueError):
                    evaluate.run('screen', 62400)
                self.assertEqual([p.name for p in Path(tmp).iterdir()], ['feasibility.json'])

    def test_refuses_a_training_run_that_is_not_complete(self):
        from experiments.xvla_b3_robosuite import contract as training
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'train'
            out.mkdir()
            (out/'result.json').write_text(json.dumps({'status': 'failed'}))
            feasibility = Path(tmp)/'feasibility.json'
            feasibility.write_text(json.dumps(dict(keys=100, failed_layouts=[], failed_keys=[], results={})))
            eval_root = Path(tmp)/'eval'
            eval_root.mkdir()
            with patch.object(training, 'OUTPUT', out), patch.object(evaluate, 'FEASIBILITY', feasibility), \
                    patch.object(protocol, 'EVAL_ROOT', eval_root):
                with self.assertRaisesRegex(RuntimeError, 'Training has not completed'):
                    evaluate.run('screen', 62400)
                self.assertEqual(list(eval_root.iterdir()), [])

    def test_refuses_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_root = Path(tmp)/'eval'
            (eval_root/'screen_step_062400').mkdir(parents=True)
            feasibility = Path(tmp)/'feasibility.json'
            feasibility.write_text(json.dumps(dict(keys=100, failed_layouts=[], failed_keys=[], results={})))
            fake = ('ckpt', 'sha', 'f'*64, 'f'*64)
            with patch.object(protocol, 'EVAL_ROOT', eval_root), patch.object(evaluate, 'FEASIBILITY', feasibility), \
                    patch.object(evaluate, 'verify_checkpoint', return_value=fake):
                with self.assertRaises(FileExistsError):
                    evaluate.run('screen', 62400)

    def test_a_parent_that_does_not_reproduce_its_fingerprint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_root = Path(tmp)/'eval'
            eval_root.mkdir()
            feasibility = Path(tmp)/'feasibility.json'
            feasibility.write_text(json.dumps(dict(keys=100, failed_layouts=[], failed_keys=[], results={})))
            with patch.object(protocol, 'EVAL_ROOT', eval_root), patch.object(evaluate, 'FEASIBILITY', feasibility), \
                    patch.object(evaluate, 'verify_checkpoint', return_value=('ckpt', 'sha', '0'*64, 'f'*64)):
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    evaluate.run('screen', 62400)
                self.assertEqual(list(eval_root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
