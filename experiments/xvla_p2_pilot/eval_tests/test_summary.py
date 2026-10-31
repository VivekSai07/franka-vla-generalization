"""The morning summary of a pilot night: the fixed decision rule, the arithmetic, and what is reported when a stage failed.

Fabricated results only, for arithmetic and structure: a number here is never evidence about a policy. Pure Python (no
simulator, no GPU, no torch).
"""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.xvla_p2_pilot import summary

CAUSES = ('success', 'never_lifted', 'wrong_object', 'wrong_bin', 'unstable_in_target', 'timeout_carrying', 'dropped_outside')


def causes(**counts):
    return {cause: counts.get(cause, 0) for cause in CAUSES}


def compare_result(new, control, layouts=192, new_only=None, control_only=None, new_causes=None, control_causes=None):
    new_only = new_only if new_only is not None else max(new-control, 0)
    control_only = control_only if control_only is not None else max(control-new, 0)
    return dict(arm='A1', checkpoint=10_000, role='primary and matched-phase', control_checkpoint=10_000, layouts=layouts,
                new_successes=new, control_successes=control, discordant=dict(new_only=new_only, control_only=control_only),
                mcnemar_exact_p=0.0123, effect=dict(estimate=(new-control)/layouts, lo=-0.01, hi=0.12), dropped_layouts=[],
                readouts=dict(new=dict(causes=new_causes or causes(success=new, never_lifted=layouts-new)),
                              control=dict(causes=control_causes or causes(success=control, never_lifted=layouts-control))))


def stage_result(successes, layouts=192):
    episodes = [dict(layout_id=f'l{i}', task_success=i < successes) for i in range(layouts)]
    return dict(status='complete', planned_trials=layouts, episodes=episodes,
                readouts=dict(causes=causes(success=successes, never_lifted=layouts-successes)))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_night(root, variants=(('force', 'A1'), ('box', 'A2')), control=120, context=110, layouts=192, results=None, gate=True,
               train=True, events=()):
    """A night root as the driver leaves it, with `results[name] = (new_successes, ...)` compare results."""
    root = Path(root)
    write(root/'night.json', dict(recipe_version='pilot_v1', updates=10_000, checkpoint=10_000, control_arm='A0', context_arm='A9',
                                  variants=[list(v) for v in variants]))
    write(root/'training/control/result.json', dict(status='complete', completed_updates=10_000, elapsed_seconds=6000.,
                                                    variant=dict(name='control', sha256='c'*64)))
    write(root/'arms/A0/comparison/step_010000/result.json', stage_result(control, layouts))
    if context is not None:
        write(root/'arms/A9/comparison/step_101520/result.json', stage_result(context, layouts))
    for name, arm in variants:
        if gate:
            write(root/f'training/{name}_zeroeq/result.json', dict(status='complete', metrics_equal=True, model_state_equal=True,
                                                                 optimizer_state_equal=True, updates=30))
        if train:
            write(root/f'training/{name}/result.json', dict(status='complete', completed_updates=10_000, elapsed_seconds=7000.,
                                                            variant=dict(name=name, sha256=name[0]*64)))
        if results and name in results:
            write(root/f'arms/{arm}/compare_step_010000.json', {**results[name], 'arm': arm})
    (root/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return root


class DecisionRuleTests(unittest.TestCase):
    def test_promote_iff_delta_is_at_least_five_points(self):
        # 1000 layouts: a difference of 49 is 4.9 points, 50 is 5.0.
        self.assertEqual(summary.decide(549, 500, 1000)['decision'], 'drop')
        self.assertEqual(summary.decide(550, 500, 1000)['decision'], 'promote')
        self.assertEqual(summary.decide(551, 500, 1000)['decision'], 'promote')
        self.assertEqual(summary.decide(500, 500, 1000)['decision'], 'drop')
        self.assertEqual(summary.decide(400, 500, 1000)['decision'], 'drop')

    def test_the_rule_at_the_real_layout_count_is_exact_arithmetic_not_a_rounded_float(self):
        # 192 layouts: 9/192 = 4.6875 points drops, 10/192 = 5.208 promotes; 5 points is 9.6 layouts, never reachable exactly.
        self.assertEqual(summary.decide(129, 120, 192)['decision'], 'drop')
        self.assertEqual(summary.decide(130, 120, 192)['decision'], 'promote')
        # An exact boundary at another count: 20 layouts, 1 more success is exactly 5 points.
        self.assertEqual(summary.decide(11, 10, 20)['decision'], 'promote')
        # Float error must not decide: 5 points of 1,000,000 layouts.
        self.assertEqual(summary.decide(500_050, 450_050, 1_000_000)['decision'], 'promote')
        self.assertEqual(summary.decide(500_049, 450_050, 1_000_000)['decision'], 'drop')

    def test_delta_is_in_points_of_the_paired_layouts(self):
        decision = summary.decide(130, 120, 192)
        self.assertAlmostEqual(decision['delta_points'], 100*10/192, places=9)
        self.assertEqual((decision['new_successes'], decision['control_successes'], decision['layouts']), (130, 120, 192))
        self.assertEqual(decision['rule'], 'promote iff delta >= +5 points')

    def test_bad_counts_are_refused(self):
        for args in ((-1, 0, 192), (10, 0, 0), (193, 0, 192), (10.5, 0, 192), (True, 0, 192)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                summary.decide(*args)


class BuildTests(unittest.TestCase):
    def build(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            make_night(tmp, **kwargs)
            return summary.build(tmp)

    def test_each_variant_has_delta_decision_p_discordance_and_mechanism_counts(self):
        out = self.build(results={'force': compare_result(130, 120, new_causes=causes(success=130, never_lifted=40, wrong_object=7, timeout_carrying=15),
                                                          control_causes=causes(success=120, never_lifted=50, wrong_object=7, timeout_carrying=15)),
                                  'box': compare_result(121, 120)})
        by = {v['name']: v for v in out['variants']}
        force = by['force']
        self.assertEqual((force['arm'], force['status'], force['decision']), ('A1', 'decided', 'promote'))
        self.assertAlmostEqual(force['delta_points'], 100*10/192, places=9)
        self.assertEqual((force['new_successes'], force['control_successes'], force['layouts']), (130, 120, 192))
        self.assertEqual(force['discordant'], dict(new_only=10, control_only=0))
        self.assertEqual(force['mcnemar_exact_p'], 0.0123)
        self.assertEqual(force['mechanism']['never_lifted'], dict(variant=40, control=50, change=-10))
        self.assertEqual(force['mechanism']['wrong_object'], dict(variant=7, control=7, change=0))
        self.assertEqual(set(force['mechanism']), set(CAUSES) - {'success'})
        self.assertEqual(by['box']['decision'], 'drop')
        self.assertEqual(force['variant_sha256'], 'f'*64)
        self.assertAlmostEqual(force['effect_points']['estimate'], 100*10/192, places=9)

    def test_the_control_and_the_untouched_checkpoint_are_context_not_a_decision(self):
        out = self.build(control=120, context=110, results={'force': compare_result(130, 120)})
        self.assertEqual((out['control']['arm'], out['control']['successes'], out['control']['layouts']), ('A0', 120, 192))
        self.assertEqual((out['context']['arm'], out['context']['successes']), ('A9', 110))
        self.assertAlmostEqual(out['context']['continuation_alone_points'], 100*10/192, places=9)
        self.assertNotIn('decision', out['control'])
        self.assertNotIn('decision', out['context'])

    def test_a_night_without_the_untouched_checkpoint_still_summarises(self):
        out = self.build(context=None, results={'force': compare_result(130, 120)})
        self.assertIsNone(out['context'])

    def test_a_variant_without_a_comparison_is_no_decision_with_the_reason(self):
        out = self.build(results={'force': compare_result(130, 120)})
        box = {v['name']: v for v in out['variants']}['box']
        self.assertEqual((box['status'], box['decision']), ('no_decision', None))
        self.assertIn('no comparison', box['reason'])
        self.assertIsNone(box['delta_points'])

    def test_a_failed_gate_or_training_is_reported_and_never_decided(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': compare_result(130, 120), 'box': compare_result(150, 120)})
            write(root/'training/force_zeroeq/result.json', dict(status='failed', error_type='ValueError', error='not byte-equal'))
            write(root/'training/box/result.json', dict(status='failed', error_type='RuntimeError', error='out of memory'))
            out = summary.build(root)
        by = {v['name']: v for v in out['variants']}
        self.assertEqual((by['force']['status'], by['force']['decision']), ('no_decision', None))
        self.assertIn('zero-equivalence gate', by['force']['reason'])
        self.assertEqual((by['box']['status'], by['box']['decision']), ('no_decision', None))
        self.assertIn('training', by['box']['reason'])
        self.assertEqual(len(out['failures']), 2)
        self.assertTrue(any('force' in f for f in out['failures']) and any('box' in f for f in out['failures']))

    def test_a_comparison_that_contradicts_its_own_counts_is_refused(self):
        bad = compare_result(130, 120)
        bad['discordant'] = dict(new_only=3, control_only=3)  # 130-120 = 10 cannot come from 3 and 3
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': bad})
            with self.assertRaises(ValueError):
                summary.build(root)

    def test_the_comparison_must_be_of_the_checkpoint_this_night_trained(self):
        wrong = compare_result(130, 120)
        wrong['checkpoint'] = 5000
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': wrong})
            with self.assertRaises(ValueError):
                summary.build(root)

    def test_wall_clock_comes_from_the_events_and_the_training_records(self):
        events = [dict(t=1000, event='start', phase='train', variant='force'), dict(t=8000, event='end', phase='train', variant='force', status='ok'),
                  dict(t=8100, event='start', phase='evaluate', variant='force'), dict(t=10600, event='end', phase='evaluate', variant='force', status='ok'),
                  dict(t=500, event='start', phase='zero_equivalence', variant='force'), dict(t=980, event='end', phase='zero_equivalence', variant='force', status='ok')]
        out = self.build(results={'force': compare_result(130, 120)}, events=events)
        clock = {v['name']: v for v in out['variants']}['force']['wall_clock_s']
        self.assertEqual((clock['zero_equivalence'], clock['train'], clock['evaluate']), (480, 7000, 2500))
        self.assertEqual(out['total_wall_clock_s'], 10600-500)

    def test_failure_events_are_reported(self):
        events = [dict(t=1, event='fail', phase='evaluate', variant='box', detail='stage incomplete')]
        out = self.build(results={'force': compare_result(130, 120), 'box': compare_result(121, 120)}, events=events)
        self.assertTrue(any('stage incomplete' in f for f in out['failures']))

    def test_the_promotion_threshold_and_the_calibration_state_are_stated(self):
        out = self.build(results={'force': compare_result(130, 120)})
        self.assertEqual((out['rule'], out['threshold_points']), ('promote iff delta >= +5 points', 5))
        self.assertIs(out['calibrated'], False)
        self.assertIn('provisional', out['calibration_note'])


class WriteTests(unittest.TestCase):
    def test_summary_json_and_a_short_markdown_are_written_and_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': compare_result(130, 120), 'box': compare_result(121, 120)})
            out = summary.write(root)
            data = json.loads((root/'summary.json').read_text())
            text = (root/'SUMMARY.md').read_text()
        self.assertEqual(data, json.loads(json.dumps(out)))
        for needle in ('force', 'box', 'promote', 'drop', '5.21', '0.0123', '130/192', '120/192', 'never_lifted', 'provisional',
                       'screen', 'McNemar'):
            self.assertIn(needle, text)
        self.assertLess(len(text.splitlines()), 60)

    def test_no_decision_rows_say_why_in_the_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': compare_result(130, 120)})
            summary.write(root)
            text = (root/'SUMMARY.md').read_text()
        self.assertIn('no decision', text)
        self.assertIn('no comparison', text)

    def test_regenerating_the_summary_from_the_same_inputs_gives_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_night(tmp, results={'force': compare_result(130, 120), 'box': compare_result(121, 120)})
            summary.write(root)
            first = (root/'summary.json').read_text()
            summary.write(root)
            self.assertEqual((root/'summary.json').read_text(), first)


if __name__ == '__main__':
    unittest.main()
