"""Collect / decide / final wiring, quarantine and the thermal guard, on synthetic folders. No simulator, no GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.xvla_props_p2_eval import analysis, evaluate, protocol


class NeedsPlan(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P2 sealed parent not mounted')
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(protocol, 'EVAL_ROOT', self.tmp),
                        mock.patch.object(evaluate, 'PREPARED', self.tmp/'prepared.json'),
                        mock.patch.object(evaluate, 'DECISIONS', self.tmp/'decisions')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        facts = lambda c: dict(checkpoint=str(self.tmp/f'ck{c}'), model_sha256=f'sha{c}', model_bytes=1, model_mtime_ns=1)
        evaluate.write_json(evaluate.PREPARED, dict(
            excluded_layouts=[], p2=dict(parent_fingerprint='f2', candidates={str(c): facts(c) for c in protocol.CANDIDATES}),
            p1=dict(parent_fingerprint='f1', candidates={str(protocol.P1_CHECKPOINT): facts(protocol.P1_CHECKPOINT)})))

    def fake_stage(self, mode, policy, candidate, success_of, skip=()):
        """Write complete trial folders; `success_of(trial)` decides each trial."""
        directory = protocol.output_dir(mode, policy, candidate)
        for trial in protocol.select_trials(mode):
            if trial['ordinal'] in skip:
                continue
            folder = protocol.trial_folder(directory, trial)
            folder.mkdir(parents=True)
            ok = success_of(trial)
            row = dict(trial, outcome='success' if ok else 'timeout', task_success=ok,
                       cause='success' if ok else 'never_lifted', wrong_object=False, ticks=10, wall_seconds=1.0)
            (folder/'episode.json').write_text(json.dumps(row))
            for name in ('cameras.mp4', 'trace.npz'):
                (folder/name).write_bytes(b'x')
        return directory


class CollectTests(NeedsPlan):
    def test_complete_stage_collects_with_readouts_and_binding(self):
        self.fake_stage('screen', 'p2', 88830, lambda t: t['ordinal'] <= 15)
        report = evaluate.collect('screen', 'p2', 88830)
        self.assertEqual((report['status'], report['planned_trials'], len(report['episodes'])), ('complete', 20, 20))
        self.assertEqual((report['model_sha256'], report['parent_fingerprint']), ('sha88830', 'f2'))
        self.assertEqual(report['readouts']['overall']['count'], 15)
        self.assertEqual(evaluate.load_stage('screen', 'p2', 88830)['status'], 'complete')

    def test_incomplete_stage_is_reported_and_refused_downstream(self):
        self.fake_stage('screen', 'p2', 88830, lambda t: True, skip={3, 7})
        report = evaluate.collect('screen', 'p2', 88830)
        self.assertEqual((report['status'], report['missing_ordinals']), ('incomplete', [3, 7]))
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            evaluate.load_stage('screen', 'p2', 88830)

    def test_a_trial_that_does_not_match_the_plan_is_refused(self):
        directory = self.fake_stage('screen', 'p2', 88830, lambda t: True)
        trial = protocol.screen_trials()[0]
        path = protocol.trial_folder(directory, trial)/'episode.json'
        row = json.loads(path.read_text())
        row['seed'] += 1
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            evaluate.collect('screen', 'p2', 88830)

    def test_half_written_folder_is_incomplete_and_gets_quarantined_not_deleted(self):
        directory = self.fake_stage('screen', 'p2', 88830, lambda t: True)
        folder = protocol.trial_folder(directory, protocol.screen_trials()[4])
        (folder/'episode.json').unlink()
        self.assertFalse(evaluate.trial_is_complete(folder))
        evaluate.quarantine(directory, folder)
        self.assertFalse(folder.exists())
        self.assertTrue((directory/'quarantine'/f'{folder.name}.0').is_dir())
        self.assertEqual(evaluate.collect('screen', 'p2', 88830)['missing_ordinals'], [5])


class DecideTests(NeedsPlan):
    def test_screen_survivors_and_report_winner_follow_the_fixed_rules(self):
        scores = {50760: 5, 76140: 12, 88830: 18, 101520: 17, 114210: 15, 126900: 16}
        for c, s in scores.items():
            self.fake_stage('screen', 'p2', c, lambda t, s=s: t['ordinal'] <= s)
            evaluate.collect('screen', 'p2', c)
        decision = evaluate.decide('screen')
        self.assertEqual(decision['survivors'], [88830, 101520, 126900])
        for c, s in {88830: 80, 101520: 90, 126900: 90}.items():
            self.fake_stage('report', 'p2', c, lambda t, s=s: t['ordinal'] <= s)
            evaluate.collect('report', 'p2', c)
        self.assertEqual(evaluate.decide('report')['winner'], 126900, 'a tie goes to the later checkpoint')

    def test_decide_refuses_a_missing_or_incomplete_stage(self):
        with self.assertRaises(RuntimeError):
            evaluate.decide('screen')

    def test_final_pairs_the_winner_with_p1_on_the_same_trials(self):
        evaluate.DECISIONS.mkdir()
        evaluate.write_json(evaluate.DECISIONS/'report.json', dict(winner=101520))
        self.fake_stage('report', 'p2', 101520, lambda t: t['ordinal'] <= 105)
        self.fake_stage('report', 'p1', 126900, lambda t: t['ordinal'] <= 60)
        self.fake_stage('control', 'p2', 101520, lambda t: True)
        self.fake_stage('control', 'p1', 126900, lambda t: t['ordinal'] <= 10)
        self.fake_stage('confirm', 'p2', 101520, lambda t: t['ordinal'] <= 17)
        for mode, policy, c in (('report', 'p2', 101520), ('report', 'p1', 126900), ('control', 'p2', 101520),
                                ('control', 'p1', 126900), ('confirm', 'p2', 101520)):
            evaluate.collect(mode, policy, c)
        out = evaluate.final()
        self.assertEqual(out['criteria']['trials'], 120)
        self.assertTrue(out['criteria']['1_overall_at_least_70pct']['passed'])
        self.assertEqual(out['readouts']['p2_confirm']['overall']['count'], 17)
        self.assertIn('optimistic', out['caveat'])
        self.assertTrue((self.tmp/'final.json').is_file())


class PreparedTests(NeedsPlan):
    def test_changed_weights_are_detected(self):
        ck = self.tmp/'ck88830'
        ck.mkdir()
        (ck/'checkpoint_result.json').write_text(json.dumps(dict(model_sha256='sha88830')))
        (ck/'model.safetensors').write_bytes(b'x')
        prepared = evaluate.load_prepared()
        prepared['p2']['candidates']['88830'].update(model_bytes=1, model_mtime_ns=(ck/'model.safetensors').stat().st_mtime_ns)
        self.assertEqual(evaluate.check_prepared(prepared, 'p2', 88830)[1], 'sha88830')
        (ck/'model.safetensors').write_bytes(b'xy')
        with self.assertRaisesRegex(ValueError, 'changed since prepare'):
            evaluate.check_prepared(prepared, 'p2', 88830)

    def test_missing_prepare_and_feasibility_are_refused(self):
        evaluate.PREPARED.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Run prepare first'):
            evaluate.load_prepared()


class ThermalTests(unittest.TestCase):
    def test_cool_down_pauses_until_cool_and_ignores_missing_telemetry(self):
        temps = iter([85, 80, 74])
        state = lambda: dict(temperature_c=next(temps), utilization_pct=0, memory_mib=0)
        slept = []
        with mock.patch.object(evaluate, 'gpu_state', state):
            self.assertEqual(evaluate.cool_down(slept.append), 30)
        self.assertEqual(slept, [15, 15])
        with mock.patch.object(evaluate, 'gpu_state', lambda: dict(temperature_c=70, utilization_pct=0, memory_mib=0)):
            self.assertEqual(evaluate.cool_down(slept.append), 0)
        with mock.patch.object(evaluate, 'gpu_state', lambda: None):
            self.assertEqual(evaluate.cool_down(slept.append), 0)


if __name__ == '__main__':
    unittest.main()
