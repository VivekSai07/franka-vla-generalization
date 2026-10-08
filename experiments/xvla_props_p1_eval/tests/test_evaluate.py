import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.xvla_props_p1_eval import evaluate, protocol


class ParseArgsTests(unittest.TestCase):
    def test_requires_mode_and_checkpoint(self):
        for argv in ([], ['--mode', 'screen'], ['--checkpoint', '50760']):
            with self.assertRaises(SystemExit):
                evaluate.parse_args(argv)

    def test_rejects_unapproved_checkpoint(self):
        with self.assertRaises(SystemExit):
            evaluate.parse_args(['--mode', 'screen', '--checkpoint', '1'])

    def test_accepts_every_mode_and_candidate(self):
        for mode in protocol.MODES:
            for candidate in protocol.CANDIDATES:
                args = evaluate.parse_args(['--mode', mode, '--checkpoint', str(candidate)])
                self.assertEqual((args.mode, args.checkpoint, args.resume), (mode, candidate, False))
        args = evaluate.parse_args(['--mode', 'screen', '--checkpoint', '50760', '--resume'])
        self.assertTrue(args.resume)


class SelectTrialsTests(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P1 parent not mounted')

    def test_smoke_is_one_screen_and_one_held_out_report(self):
        trials = evaluate.select_trials('smoke')
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0]['layout_id'], protocol.screen_trials()[0]['layout_id'])
        self.assertEqual(trials[1]['kind'], 'held_out')

    def test_each_mode_matches_its_protocol_function(self):
        self.assertEqual(evaluate.select_trials('screen'), protocol.screen_trials())
        self.assertEqual(evaluate.select_trials('report'), protocol.report_trials())
        self.assertEqual(evaluate.select_trials('confirm'), protocol.confirm_trials())

    def test_rejects_unapproved_mode(self):
        with self.assertRaises(ValueError):
            evaluate.select_trials('train')


class LoadFeasibilityTests(unittest.TestCase):
    def test_missing_file_refuses(self):
        with patch.object(evaluate, 'FEASIBILITY', Path('/nonexistent/feasibility.json')):
            with self.assertRaises(RuntimeError):
                evaluate.load_feasibility()

    def test_incomplete_coverage_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'feasibility.json'
            path.write_text(json.dumps(dict(keys=79, succeeded=79, failed_layouts=[])))
            with patch.object(evaluate, 'FEASIBILITY', path):
                with self.assertRaises(ValueError):
                    evaluate.load_feasibility()

    def test_complete_coverage_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'feasibility.json'
            path.write_text(json.dumps(dict(keys=80, succeeded=78, failed_layouts=['screen_003', 'report_010'])))
            with patch.object(evaluate, 'FEASIBILITY', path):
                summary = evaluate.load_feasibility()
                self.assertEqual(summary['keys'], 80)


class ResumeStateTests(unittest.TestCase):
    def test_nothing_to_resume_without_a_prior_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluate.resume_state(tmp, {})

    def test_resume_requires_running_or_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'result.json').write_text(json.dumps(dict(status='complete')))
            with self.assertRaises(ValueError):
                evaluate.resume_state(tmp, {})

    def test_resume_rejects_a_changed_recorded_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'result.json').write_text(json.dumps(dict(status='failed', mode='screen', episodes=[])))
            with self.assertRaises(ValueError):
                evaluate.resume_state(tmp, {'mode': 'report'})

    def test_resume_collects_finished_trials_and_stale_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(ordinal=1, layout_id='screen_000', task_success=True)
            folder = root/'trial_001_screen_000'
            folder.mkdir()
            for name in evaluate.TRIAL_FILES:
                (folder/name).write_bytes(b'x')
            (folder/'episode.json').write_text(json.dumps(row))
            (root/'result.json').write_text(json.dumps(dict(status='failed', mode='screen', episodes=[row])))
            (root/'failure.json').write_text('{}')
            done, stale, prior = evaluate.resume_state(root, {'mode': 'screen'})
            self.assertEqual(done, [row])
            self.assertEqual({p.name for p in stale}, {'failure.json'})

    def test_quarantine_moves_never_deletes_and_avoids_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root/'failure.json'
            victim.write_text('{}')
            evaluate.quarantine(root, [victim])
            self.assertFalse(victim.exists())
            self.assertTrue((root/'quarantine'/'failure.json.0').exists())


if __name__ == '__main__':
    unittest.main()
