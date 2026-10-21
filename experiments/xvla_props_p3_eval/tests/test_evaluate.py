"""Collect / decide / final wiring, quarantine and the thermal guard, on synthetic folders. No simulator, no GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.xvla_props_p3_eval import analysis, evaluate, feasibility, protocol

BIN = {'can': 'orange', 'milk': 'orange', 'bottle': 'orange', 'bread': 'blue', 'lemon': 'blue', 'cereal': 'blue'}


class NeedsPlan(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P3 sealed parent not mounted')
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(protocol, 'EVAL_ROOT', self.tmp),
                        mock.patch.object(evaluate, 'PREPARED', self.tmp/'prepared.json'),
                        mock.patch.object(evaluate, 'DECISIONS', self.tmp/'decisions')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        facts = lambda c: dict(checkpoint=str(self.tmp/f'ck{c}'), model_sha256=f'sha{c}', model_bytes=1, model_mtime_ns=1)
        evaluate.write_json(evaluate.PREPARED, dict(
            excluded_layouts=[], p3=dict(parent_fingerprint='f3', candidates={str(c): facts(c) for c in protocol.CANDIDATES})))

    def fake_stage(self, mode, candidate, outcome_of, skip=()):
        """Write complete trial folders; `outcome_of(trial)` is 'correct' / 'wrong' / None (not in a bin)."""
        directory = protocol.output_dir(mode, candidate)
        for trial in protocol.select_trials(mode):
            if trial['ordinal'] in skip:
                continue
            folder = protocol.trial_folder(directory, trial)
            folder.mkdir(parents=True)
            outcome = outcome_of(trial)
            placed = {'correct': trial['bin'], 'wrong': 'blue' if trial['bin'] == 'orange' else 'orange', None: None}[outcome]
            cause = 'success' if outcome == 'correct' else 'wrong_bin' if outcome == 'wrong' else 'never_lifted'
            row = dict(trial, outcome='success' if outcome == 'correct' else 'timeout', task_success=outcome == 'correct',
                       placed_bin=placed, cause=cause, ticks=10, wall_seconds=1.0, final_gripper_open=True,
                       names=[trial['object']])
            (folder/'episode.json').write_text(json.dumps(row))
            for name in ('cameras.mp4', 'trace.npz'):
                (folder/name).write_bytes(b'x')
        return directory


class CollectTests(NeedsPlan):
    def test_complete_stage_collects_with_readouts_and_binding(self):
        self.fake_stage('screen', 88830, lambda t: 'correct' if t['ordinal'] <= 15 else None)
        report = evaluate.collect('screen', 88830)
        self.assertEqual((report['status'], report['planned_trials'], len(report['episodes'])), ('complete', 24, 24))
        self.assertEqual((report['model_sha256'], report['parent_fingerprint']), ('sha88830', 'f3'))
        self.assertEqual(report['readouts']['overall']['count'], 15)
        self.assertEqual(evaluate.load_stage('screen', 88830)['status'], 'complete')

    def test_incomplete_stage_is_reported_and_refused_downstream(self):
        self.fake_stage('screen', 88830, lambda t: 'correct', skip={3, 7})
        report = evaluate.collect('screen', 88830)
        self.assertEqual((report['status'], report['missing_ordinals']), ('incomplete', [3, 7]))
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            evaluate.load_stage('screen', 88830)
        with self.assertRaisesRegex(RuntimeError, 'is missing'):
            evaluate.load_stage('screen', 50760)

    def test_a_trial_that_does_not_match_the_plan_is_refused(self):
        directory = self.fake_stage('screen', 88830, lambda t: 'correct')
        path = protocol.trial_folder(directory, protocol.screen_trials()[0])/'episode.json'
        row = json.loads(path.read_text())
        row['seed'] += 1
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            evaluate.collect('screen', 88830)

    def test_a_nonfinite_action_trial_is_refused(self):
        directory = self.fake_stage('screen', 88830, lambda t: 'correct')
        path = protocol.trial_folder(directory, protocol.screen_trials()[2])/'episode.json'
        row = json.loads(path.read_text())
        row['outcome'] = 'nonfinite_action'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            evaluate.collect('screen', 88830)

    def test_half_written_folder_is_incomplete_and_gets_quarantined_not_deleted(self):
        directory = self.fake_stage('screen', 88830, lambda t: 'correct')
        folder = protocol.trial_folder(directory, protocol.screen_trials()[4])
        (folder/'episode.json').unlink()
        self.assertFalse(evaluate.trial_is_complete(folder))
        evaluate.quarantine(directory, folder)
        self.assertFalse(folder.exists())
        self.assertTrue((directory/'quarantine'/f'{folder.name}.0').is_dir())
        self.assertEqual(evaluate.collect('screen', 88830)['missing_ordinals'], [5])
        folder.mkdir()
        evaluate.quarantine(directory, folder)
        self.assertTrue((directory/'quarantine'/f'{folder.name}.1').is_dir(), 'a second quarantine never overwrites the first')

    def test_a_folder_missing_any_one_trial_file_is_incomplete_and_collected_as_missing(self):
        directory = self.fake_stage('screen', 88830, lambda t: 'correct')
        for ordinal, name in ((2, 'cameras.mp4'), (3, 'trace.npz'), (4, 'episode.json')):
            folder = protocol.trial_folder(directory, protocol.screen_trials()[ordinal - 1])
            (folder/name).unlink()
            self.assertFalse(evaluate.trial_is_complete(folder), name)
        self.assertEqual(evaluate.collect('screen', 88830)['missing_ordinals'], [2, 3, 4])

    def test_the_infeasible_layouts_of_prepare_are_dropped_from_the_readouts(self):
        prepared = evaluate.load_prepared()
        victim = protocol.screen_trials()[0]['layout_id']
        evaluate.write_json(evaluate.PREPARED, {**prepared, 'excluded_layouts': [victim]})
        self.fake_stage('screen', 88830, lambda t: 'correct')
        report = evaluate.collect('screen', 88830)
        self.assertEqual(report['readouts']['dropped_infeasible_trials'], 1)
        self.assertEqual(report['readouts']['overall']['n'], 23)
        self.assertEqual(report['excluded_layouts'], [victim])


class DecideTests(NeedsPlan):
    def test_screen_survivors_and_report_winner_follow_the_fixed_rules(self):
        scores = {50760: 5, 76140: 12, 88830: 18, 101520: 17, 114210: 15, 126900: 16}
        for c, s in scores.items():
            self.fake_stage('screen', c, lambda t, s=s: 'correct' if t['ordinal'] <= s else None)
            evaluate.collect('screen', c)
        decision = evaluate.decide('screen')
        survivors = decision['survivors']
        self.assertEqual(survivors, [88830, 101520, 126900])
        trained_ordinals = [x['ordinal'] for x in protocol.report_trials() if x['kind'] == 'trained']
        trained_successes = {88830: 60, 101520: 70, 126900: 70}
        for c, s in trained_successes.items():
            winners = set(trained_ordinals[:s])

            def outcome(t, winners=winners):
                return 'correct' if t['kind'] == 'held_out' or t['ordinal'] in winners else None
            self.fake_stage('report', c, outcome)
            evaluate.collect('report', c)
        report = evaluate.decide('report')
        self.assertEqual(report['winner'], 126900, 'a tie goes to the later checkpoint')
        self.assertEqual(report['counts'], {'88830': 60, '101520': 70, '126900': 70})

    def test_held_out_successes_never_decide_the_winner(self):
        survivors = (101520, 126900)
        evaluate.DECISIONS.mkdir()
        evaluate.write_json(evaluate.DECISIONS/'screen.json', dict(survivors=list(survivors)))
        # 101520: 50 trained successes and every held-out layout right; 126900: 51 trained and no held-out success.
        for c, trained, held in ((101520, 50, True), (126900, 51, False)):
            seen = {'n': 0}

            def outcome(t, trained=trained, held=held, seen=seen):
                if t['kind'] == 'held_out':
                    return 'correct' if held else None
                seen['n'] += 1
                return 'correct' if seen['n'] <= trained else None
            self.fake_stage('report', c, outcome)
            evaluate.collect('report', c)
        decision = evaluate.decide('report')
        self.assertEqual(decision['winner'], 126900)
        self.assertEqual(decision['counts'], {'101520': 50, '126900': 51})
        self.assertEqual(decision['all_layout_successes'], {'101520': 50 + 160, '126900': 51})

    def test_an_infeasible_layout_does_not_count_towards_the_screen(self):
        victim = protocol.screen_trials()[0]['layout_id']          # ordinal 1
        prepared = evaluate.load_prepared()
        evaluate.write_json(evaluate.PREPARED, {**prepared, 'excluded_layouts': [victim]})
        wins = {88830: set(range(2, 20)),                          # 18, none on the excluded layout
                101520: {1} | set(range(2, 17))}                   # 16 raw, 15 once layout 1 is dropped
        for c in protocol.CANDIDATES:
            ordinals = wins.get(c, set())
            self.fake_stage('screen', c, lambda t, o=ordinals: 'correct' if t['ordinal'] in o else None)
            evaluate.collect('screen', c)
        decision = evaluate.decide('screen')
        self.assertEqual(decision['counts']['101520'], 15)
        self.assertEqual(decision['survivors'], [88830], '15 is three behind 18: not a survivor (raw counts would keep it)')

    def test_decide_refuses_a_missing_or_incomplete_stage_and_an_unknown_stage(self):
        with self.assertRaises(RuntimeError):
            evaluate.decide('screen')
        with self.assertRaises(ValueError):
            evaluate.decide('confirm')

    def test_final_reads_the_winner_with_the_confirm_tier_and_every_survivor(self):
        evaluate.DECISIONS.mkdir()
        evaluate.write_json(evaluate.DECISIONS/'report.json', dict(winner=101520))
        evaluate.write_json(evaluate.DECISIONS/'screen.json', dict(survivors=[101520, 126900]))
        good = lambda t: 'correct'
        for mode, c, outcome in (('report', 101520, good), ('report', 126900, lambda t: 'wrong'), ('confirm', 101520, good)):
            self.fake_stage(mode, c, outcome)
            evaluate.collect(mode, c)
        out = evaluate.final()
        crit = out['criteria']
        self.assertEqual(crit['layouts'], 240)
        self.assertEqual(crit['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')
        self.assertEqual(crit['2_held_out_generalisation_exploratory']['verdict'], 'evidence')
        self.assertEqual(out['survivors']['126900']['1_trained_wrong_bin_at_most_5pct']['verdict'], 'fail')
        self.assertEqual(out['readouts']['confirm']['overall']['count'], 24)
        self.assertEqual(out['readouts']['report']['overall']['n'], 240)
        self.assertIn('optimistic', out['caveat'])
        self.assertTrue((self.tmp/'final.json').is_file())
        self.assertEqual(json.loads((self.tmp/'final.json').read_text())['winner'], 101520)


    def test_an_infeasible_layout_does_not_count_towards_the_report_winner(self):
        victim = next(t for t in protocol.report_trials() if t['kind'] == 'trained')['layout_id']
        prepared = evaluate.load_prepared()
        evaluate.write_json(evaluate.PREPARED, {**prepared, 'excluded_layouts': [victim]})
        evaluate.DECISIONS.mkdir()
        evaluate.write_json(evaluate.DECISIONS/'screen.json', dict(survivors=[101520, 126900]))
        # Raw, 101520 leads 80 to 79; with the infeasible layout dropped they tie and the later checkpoint wins.
        self.fake_stage('report', 101520, lambda t: 'correct' if t['kind'] == 'trained' else None)
        self.fake_stage('report', 126900, lambda t: None if t['layout_id'] == victim else
                        'correct' if t['kind'] == 'trained' else None)
        for c in (101520, 126900):
            evaluate.collect('report', c)
        decision = evaluate.decide('report')
        self.assertEqual(decision['counts'], {'101520': 79, '126900': 79})
        self.assertEqual(decision['winner'], 126900)

    def test_final_reads_whichever_winner_was_recorded(self):
        evaluate.DECISIONS.mkdir()
        evaluate.write_json(evaluate.DECISIONS/'report.json', dict(winner=126900))
        evaluate.write_json(evaluate.DECISIONS/'screen.json', dict(survivors=[101520, 126900]))
        for mode, c, outcome in (('report', 101520, lambda t: 'wrong'), ('report', 126900, lambda t: 'correct'),
                                 ('confirm', 126900, lambda t: 'correct')):
            self.fake_stage(mode, c, outcome)
            evaluate.collect(mode, c)
        out = evaluate.final()
        self.assertEqual(out['winner'], 126900)
        self.assertEqual(out['criteria']['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')
        self.assertEqual(out['survivors']['101520']['1_trained_wrong_bin_at_most_5pct']['verdict'], 'fail')


class PreparedTests(NeedsPlan):
    def test_changed_weights_are_detected(self):
        ck = self.tmp/'ck88830'
        ck.mkdir()
        (ck/'checkpoint_result.json').write_text(json.dumps(dict(model_sha256='sha88830')))
        (ck/'model.safetensors').write_bytes(b'x')
        prepared = evaluate.load_prepared()
        prepared['p3']['candidates']['88830'].update(model_bytes=1, model_mtime_ns=(ck/'model.safetensors').stat().st_mtime_ns)
        self.assertEqual(evaluate.check_prepared(prepared, 88830)[1], 'sha88830')
        (ck/'model.safetensors').write_bytes(b'xy')
        with self.assertRaisesRegex(ValueError, 'changed since prepare'):
            evaluate.check_prepared(prepared, 88830)
        (ck/'model.safetensors').write_bytes(b'x')
        prepared['p3']['candidates']['88830']['model_mtime_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'changed since prepare'):
            evaluate.check_prepared(prepared, 88830)
        prepared['p3']['candidates']['88830'].update(model_mtime_ns=(ck/'model.safetensors').stat().st_mtime_ns,
                                                      model_sha256='other')
        with self.assertRaisesRegex(ValueError, 'changed since prepare'):
            evaluate.check_prepared(prepared, 88830)

    def test_missing_prepare_and_feasibility_are_refused(self):
        evaluate.PREPARED.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Run prepare first'):
            evaluate.load_prepared()
        with mock.patch.object(evaluate, 'FEASIBILITY', self.tmp/'nope.json'):
            with self.assertRaisesRegex(RuntimeError, 'feasibility pass first'):
                evaluate.load_feasibility()
        partial = self.tmp/'feasibility.json'
        partial.write_text(json.dumps(dict(keys=120, succeeded=120, failed_layouts=[])))
        with mock.patch.object(evaluate, 'FEASIBILITY', partial):
            with self.assertRaisesRegex(ValueError, '288'):
                evaluate.load_feasibility()
        partial.write_text(json.dumps(dict(keys=288, succeeded=287, failed_layouts=['report_001'])))
        with mock.patch.object(evaluate, 'FEASIBILITY', partial):
            self.assertEqual(evaluate.load_feasibility()['failed_layouts'], ['report_001'])


class FeasibilitySummaryTests(NeedsPlan):
    def test_summary_counts_failures_by_object_and_shards_partition_the_layouts(self):
        hold = protocol._plan()['evaluation_holdout']
        results = {k: dict(success=not (k in ('report_001', 'report_002')), reason='x', frames=1, stable_ticks=0, seconds=0.0)
                   for k in hold}
        summary = feasibility.summarize(results, hold)
        self.assertEqual((summary['keys'], summary['succeeded']), (288, 286))
        self.assertEqual(summary['failed_layouts'], ['report_001', 'report_002'])
        self.assertEqual(sum(summary['failed_by_object'].values()), 2)
        self.assertEqual(set(summary['failed_by_object']), {'can', 'milk', 'bread', 'lemon', 'bottle', 'cereal'})
        self.assertNotIn('failed_by_object', feasibility.summarize(results))
        shards = [feasibility.keys_of_shard(k, 7, hold) for k in range(7)]
        self.assertEqual(sorted(x for s in shards for x in s), sorted(hold))
        self.assertEqual(feasibility.KEYS, 288)
        for bad in ((-1, 3), (3, 3), (0, 0), (0, 2.5)):
            with self.assertRaises(ValueError):
                feasibility.keys_of_shard(bad[0], bad[1], hold)


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
        hot = lambda t: dict(temperature_c=t, utilization_pct=0, memory_mib=0)
        with mock.patch.object(evaluate, 'gpu_state', side_effect=[hot(83), hot(75)]):
            self.assertEqual(evaluate.cool_down(slept.append), 15, 'at the hot threshold the pause starts; 75 is cool')
        with mock.patch.object(evaluate, 'gpu_state', side_effect=[hot(82.9)]):
            self.assertEqual(evaluate.cool_down(slept.append), 0, 'just below the hot threshold nothing pauses')
        with mock.patch.object(evaluate, 'gpu_state', lambda: None):
            self.assertEqual(evaluate.cool_down(slept.append), 0)


class EntryPointTests(unittest.TestCase):
    def test_the_module_calls_main_when_run(self):
        # FAILURES #36: a forked evaluator once lost its `if __name__ == '__main__'` guard and every stage "passed".
        source = Path(evaluate.__file__).read_text()
        self.assertRegex(source, r"if __name__ == '__main__':\n    sys\.exit\(main\(\)\)")
        source = Path(feasibility.__file__).read_text()
        self.assertRegex(source, r"if __name__ == '__main__':\n    main\(\)")

    def test_arguments_and_unknown_modes(self):
        args = evaluate.parse_args(['worker', '--mode', 'screen', '--candidates', '50760,76140', '--shard', '1', '--shards', '4'])
        self.assertEqual((args.mode, args.shard, args.shards, evaluate._ints(args.candidates)), ('screen', 1, 4, (50760, 76140)))
        for bad in (['worker', '--mode', 'control', '--candidates', '1', '--shard', '0', '--shards', '1'],
                    ['decide', 'confirm'], ['collect', '--mode', 'screen']):
            with self.assertRaises(SystemExit):
                evaluate.parse_args(bad)


if __name__ == '__main__':
    unittest.main()
