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


class ResumeTests(unittest.TestCase):
    EXPECTED = dict(mode='screen', candidate=62400, model_sha256='sha', trials=[dict(ordinal=1), dict(ordinal=2)])

    def make_run(self, root, status='running', finished=(1,), half=(2,)):
        directory = Path(root)/'screen_step_062400'
        directory.mkdir()
        rows = []
        for n in finished:
            row = dict(ordinal=n, layout_id=f'evaluation_00{n}', condition='distractor', task_success=True)
            folder = directory/f'trial_{n:03d}_evaluation_00{n}_distractor'
            folder.mkdir()
            for name in evaluate.TRIAL_FILES:
                (folder/name).write_text(json.dumps(row) if name == 'episode.json' else 'x')
            rows.append(row)
        for n in half:
            (directory/f'trial_{n:03d}_evaluation_00{n}_distractor').mkdir()
        (directory/'result.json').write_text(json.dumps(dict(self.EXPECTED, status=status, episodes=rows)))
        return directory, rows

    def test_keeps_finished_trials_and_flags_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, rows = self.make_run(tmp)
            (directory/'failure.json').write_text('{}')
            done, stale, _ = evaluate.resume_state(directory, self.EXPECTED)
            self.assertEqual(done, rows)
            self.assertEqual(sorted(p.name for p in stale), ['failure.json', 'trial_002_evaluation_002_distractor'])

    def test_refuses_what_it_cannot_prove_is_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = self.make_run(tmp)
            for key, value in (('model_sha256', 'other'), ('candidate', 93600), ('mode', 'report'),
                               ('trials', [dict(ordinal=1)])):
                with self.assertRaisesRegex(ValueError, key):
                    evaluate.resume_state(directory, dict(self.EXPECTED, **{key: value}))

    def test_refuses_a_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = self.make_run(tmp, status='complete')
            with self.assertRaisesRegex(ValueError, 'complete'):
                evaluate.resume_state(directory, self.EXPECTED)

    def test_refuses_a_recorded_trial_with_a_missing_or_changed_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = self.make_run(tmp)
            folder = directory/'trial_001_evaluation_001_distractor'
            (folder/'trace.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'no complete folder'):
                evaluate.resume_state(directory, self.EXPECTED)
            (folder/'trace.npz').write_text('x')
            (folder/'episode.json').write_text(json.dumps(dict(ordinal=1, task_success=False)))
            with self.assertRaisesRegex(ValueError, 'differs from its episode.json'):
                evaluate.resume_state(directory, self.EXPECTED)

    def test_nothing_to_resume_without_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluate.resume_state(Path(tmp), self.EXPECTED)

    def test_quarantine_moves_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for _ in range(2):
                (directory/'failure.json').write_text('{"n": 1}')
                evaluate.quarantine(directory, [directory/'failure.json'])
            self.assertEqual(sorted(p.name for p in (directory/'quarantine').iterdir()),
                             ['failure.json.0', 'failure.json.1'])
            self.assertFalse((directory/'failure.json').exists())

    def test_only_unfinished_trials_are_left_to_run(self):
        trials = [dict(ordinal=n, colours=['blue', 'red'] if n % 2 else ['blue', 'red', 'yellow']) for n in range(1, 7)]
        left = evaluate.remaining_by_colours(trials, [dict(ordinal=1), dict(ordinal=4)])
        self.assertEqual(sorted(t['ordinal'] for group in left.values() for t in group), [2, 3, 5, 6])
        self.assertEqual(len(evaluate.remaining_by_colours(trials, [])), 2)
        self.assertEqual(len(evaluate.remaining_by_colours(trials, [dict(ordinal=n) for n in range(1, 7)])), 0)

    def test_cli_resume_is_opt_in(self):
        args = ['--mode', 'report', '--checkpoint', '62400']
        self.assertFalse(evaluate.parse_args(args).resume)
        self.assertTrue(evaluate.parse_args(args + ['--resume']).resume)


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

    def test_a_crashed_run_resumes_keeping_finished_trials_and_quarantining_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_root = Path(tmp)/'eval'
            eval_root.mkdir()
            feasibility = Path(tmp)/'feasibility.json'
            feasibility.write_text(json.dumps(dict(keys=100, failed_layouts=[], failed_keys=[], results={})))
            fake = ('ckpt', 'sha', 'f'*64, 'f'*64)
            seen = {}

            def crash(directory, report, trials, excluded, checkpoint, done=()):
                row = dict(trials[0], outcome='success', task_success=True, cause='success')
                folder = directory/f"trial_{row['ordinal']:03d}_{row['layout_id']}_{row['condition']}"
                folder.mkdir()
                for name in evaluate.TRIAL_FILES:
                    (folder/name).write_text(json.dumps(row) if name == 'episode.json' else 'x')
                report['episodes'] = [row]
                evaluate.write_json(directory/'result.json', report)
                (directory/f"trial_{trials[1]['ordinal']:03d}_{trials[1]['layout_id']}_{trials[1]['condition']}").mkdir()
                raise RuntimeError('killed')

            def resumed(directory, report, trials, excluded, checkpoint, done=()):
                seen.update(done=list(done), report=dict(report))

            with patch.object(protocol, 'EVAL_ROOT', eval_root), patch.object(evaluate, 'FEASIBILITY', feasibility), \
                    patch.object(evaluate, 'verify_checkpoint', return_value=fake):
                with patch.object(evaluate, '_run_trials', crash), self.assertRaises(RuntimeError):
                    evaluate.run('screen', 62400)
                with self.assertRaises(FileExistsError):
                    evaluate.run('screen', 62400)
                with patch.object(evaluate, '_run_trials', resumed):
                    evaluate.run('screen', 62400, resume=True)
            directory = eval_root/'screen_step_062400'
            self.assertEqual([r['ordinal'] for r in seen['done']], [1])
            self.assertEqual(seen['report']['resumes'][0]['finished_trials'], 1)
            self.assertEqual(sorted(p.name for p in (directory/'quarantine').iterdir()),
                             ['failure.json.0', 'trial_002_evaluation_001_distractor.0'])
            self.assertTrue((directory/'trial_001_evaluation_000_distractor').is_dir())

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
