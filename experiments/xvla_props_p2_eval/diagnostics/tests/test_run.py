"""Robosuite image only: per-trial measures, Gates 0 and 1, and the analysis tables, on real fixture trials."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import mujoco
    MUJOCO = mujoco.__version__ == '3.4.0'
except ImportError:  # admission image: no simulator
    MUJOCO = False

EPISODES = Path('/vla/outputs/props_p2_data/props_p2_20261013/episodes')
from experiments.xvla_props_p2_eval import protocol
from experiments.xvla_props_p2_eval.diagnostics import run  # imports no simulator at module level

if MUJOCO:
    from experiments.xvla_props_p2_eval.diagnostics import reference
    AVAILABLE = EPISODES.is_dir() and (protocol.EVAL_ROOT/'report_p2_step_101520').is_dir()
else:
    AVAILABLE = False


def pick(predicate):
    root = protocol.EVAL_ROOT/'report_p2_step_101520'
    rows = json.loads((root/'result.json').read_text())['episodes']
    row = next(r for r in rows if predicate(r))
    return protocol.trial_folder(root, row), row


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the P2 evaluation outputs')
class MeasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calm = lambda r: r['task_success'] and max(r['distractor_displacement_m'].values()) < 1e-3
        cls.cases = {'success': pick(calm), 'never_lifted': pick(lambda r: r['cause'] == 'never_lifted'),
                     'wrong_object': pick(lambda r: r['cause'] == 'wrong_object')}

    def measure(self, key):
        return run.measure_trial(self.cases[key][0])

    def test_success_is_lifted_closed_and_held_with_raw_samples(self):
        m = self.measure('success')
        self.assertTrue(m['gate0'])
        self.assertTrue(m['lifted'] and m['closed'] and m['held'])
        self.assertLessEqual(m['grasp_closure'], m['lift_tick'])
        self.assertEqual(len(m['grasp_sample']), 4)
        self.assertEqual(len(m['closure_samples']), len(m['names']))
        self.assertTrue(all(len(v) == 4 for v in m['closure_samples']))

    def test_never_lifted_has_a_closest_approach_and_no_verdict_yet(self):
        m = self.measure('never_lifted')
        self.assertTrue(m['gate0'])
        self.assertFalse(m['lifted'])
        self.assertGreater(m['closest_approach_m'], 0.0)
        self.assertNotIn('class', m, 'the verdict needs an envelope and is applied by judge(), not stored')

    def test_wrong_object_names_the_lifted_distractor(self):
        m = self.measure('wrong_object')
        self.assertTrue(m['gate0'])
        self.assertEqual(m['first_lifted'], self.cases['wrong_object'][1]['first_lifted'])

    def test_measure_is_deterministic(self):
        self.assertEqual(self.measure('never_lifted'), self.measure('never_lifted'))


def synthetic(ticks=100, closed=(20, 60), lift_at=40, target_off=0.0, distractor_off=0.2, touch=(25, 60)):
    """A hand-built replay: identity hand at the origin, target on the hand axis, one distractor beside it."""
    import numpy as np
    rep = dict(names=['can', 'milk'], gate0=True, first_mismatch=None)
    rep['hand_pos'] = np.zeros((ticks, 3))
    rep['hand_rot'] = np.tile(np.eye(3).ravel(), (ticks, 1))
    rep['left_pos'] = np.tile([-0.04, 0, 0.05], (ticks, 1))
    rep['right_pos'] = np.tile([0.04, 0, 0.05], (ticks, 1))
    props = np.zeros((ticks, 2, 3))
    props[:, 0] = [target_off, 0, 0.15]
    props[:, 1] = [distractor_off, 0, 0.15]
    if lift_at is not None:
        props[lift_at:, 0, 2] += 0.05
    rep['props_xyz'] = props
    rep['props_yaw_deg'] = np.zeros((ticks, 2))
    command = np.full(ticks, 255.0)
    if closed is not None:
        command[closed[0]:closed[1]] = 0.0
    rep['command'] = command
    for key in ('left_touch', 'right_touch', 'touched'):
        rep[key] = np.zeros((ticks, 2), bool)
    if touch is not None:
        rep['left_touch'][touch[0]:touch[1], 0] = rep['right_touch'][touch[0]:touch[1], 0] = True
    return rep


REF = {'can': dict(lo=[-0.02, -0.02, 0.10, -20], hi=[0.02, 0.02, 0.20, 20], n=10),
       'milk': dict(lo=[-0.02, -0.02, 0.10, -20], hi=[0.02, 0.02, 0.20, 20], n=10)}


def fake_row(cause='never_lifted', success=False):
    return dict(cause=cause, task_success=success)


def judged(row, rep, ref):
    return run.judge(row, run.measures_from_replay(rep), ref)


class ReduceTests(unittest.TestCase):
    """Every field of the reduction and the judgement on synthetic replays with known answers (no simulator)."""

    def test_a_clean_grasp(self):
        m = run.measures_from_replay(synthetic())
        self.assertEqual((m['first_closure'], m['grasp_closure'], m['lift_tick']), (20, 20, 40))
        self.assertTrue(m['closed'] and m['lifted'] and m['held'])
        self.assertAlmostEqual(m['closest_approach_m'], 0.10, places=6)
        self.assertEqual(m['grasp_sample'][:3], [0.0, 0.0, 0.15])
        self.assertEqual(len(m['closure_samples']), 2)
        j = run.judge(fake_row('success', True), m, REF)
        self.assertTrue(j['target_in_env'] and j['grasp_inside'])
        self.assertEqual(j['distractor_in_env'], [False])
        self.assertEqual(j['class'], 'success')

    def test_target_off_the_envelope_is_an_approach_error(self):
        j = judged(fake_row(), synthetic(target_off=0.1, lift_at=None), REF)
        self.assertFalse(j['target_in_env'])
        self.assertEqual(j['class'], 'closed_off_target')

    def test_gripper_closed_on_the_distractor(self):
        j = judged(fake_row(), synthetic(target_off=0.1, distractor_off=0.0, lift_at=None), REF)
        self.assertEqual(j['distractor_in_env'], [True])
        self.assertEqual(j['class'], 'closed_on_distractor')

    def test_never_closed(self):
        m = run.measures_from_replay(synthetic(closed=None, lift_at=None, touch=None))
        self.assertFalse(m['closed'])
        self.assertIsNone(m['grasp_closure'])
        self.assertIsNone(m['grasp_sample'])
        self.assertEqual(run.judge(fake_row(), m, REF)['class'], 'never_closed')

    def test_chatter_is_not_a_closure(self):
        self.assertFalse(run.measures_from_replay(synthetic(closed=(30, 33), lift_at=None, touch=None))['closed'])

    def test_closed_in_envelope_but_not_held(self):
        j = judged(fake_row(), synthetic(lift_at=None, touch=(25, 30)), REF)
        self.assertTrue(j['target_in_env'] and not j['held'])
        self.assertEqual(j['class'], 'closed_not_held')

    def test_held_but_not_lifted(self):
        self.assertEqual(judged(fake_row(), synthetic(lift_at=None), REF)['class'], 'held_not_lifted')

    def test_held_is_measured_after_the_closure_not_before(self):
        self.assertFalse(run.measures_from_replay(synthetic(lift_at=None, touch=(2, 12)))['held'],
                         'contact before the first closure is not a hold')

    def test_grasp_closure_follows_the_run_that_lifts(self):
        rep = synthetic(closed=(20, 25), lift_at=None)
        rep['command'][50:90] = 0.0
        rep['props_xyz'][70:, 0, 2] += 0.05
        m = run.measures_from_replay(rep)
        self.assertEqual((m['first_closure'], m['grasp_closure']), (20, 50))
        self.assertEqual(run.judge(fake_row('timeout_carrying'), m, REF)['class'], 'lifted_then_failed')

    def test_a_distractor_is_judged_against_its_own_objects_envelope(self):
        ref = {'can': REF['can'], 'milk': dict(lo=[-0.05, -0.02, 0.10, -20], hi=[0.05, 0.02, 0.20, 20], n=10)}
        rep = synthetic(target_off=0.1, distractor_off=0.03, lift_at=None)
        j = run.judge(fake_row(), run.measures_from_replay(rep), ref)
        self.assertEqual(j['distractor_in_env'], [True])
        self.assertEqual(j['class'], 'closed_on_distractor')
        self.assertEqual(run.judge(fake_row(), run.measures_from_replay(rep), REF)['distractor_in_env'], [False])

    def test_the_same_measures_are_judged_differently_by_different_envelopes(self):
        m = run.measures_from_replay(synthetic(target_off=0.03, lift_at=None))
        wide = {'can': dict(lo=[-0.05, -0.05, 0.1, -45], hi=[0.05, 0.05, 0.2, 45], n=10), 'milk': REF['milk']}
        self.assertEqual(run.judge(fake_row(), m, REF)['class'], 'closed_off_target')
        self.assertEqual(run.judge(fake_row(), m, wide)['class'], 'held_not_lifted')

    def test_touched_distractors_are_listed(self):
        rep = synthetic()
        rep['touched'][30:40, 1] = True
        self.assertEqual(run.measures_from_replay(rep)['touched_distractors'], ['milk'])

    def test_missing_object_envelope_is_refused(self):
        m = run.measures_from_replay(synthetic())
        with self.assertRaises(ValueError):
            run.judge(fake_row(), m, {'can': REF['can']})


class SuccessEnvelopeTests(unittest.TestCase):
    def rows(self, obj, n, offset=0.0, success=True):
        out = []
        for i in range(n):
            out.append(dict(object=obj, task_success=success, grasp_sample=[offset + i * 0.001, 0.0, 0.15, float(i)]))
        return out

    def test_envelope_is_p1_to_p99_of_successes_only(self):
        rows = self.rows('can', 101) + self.rows('can', 50, offset=1.0, success=False)
        env = run.success_envelope(rows)
        self.assertEqual(env['can']['n'], 101)
        self.assertAlmostEqual(env['can']['lo'][0], 0.001, places=9)
        self.assertAlmostEqual(env['can']['hi'][0], 0.099, places=9)

    def test_too_few_successes_stop_the_analysis(self):
        with self.assertRaises(ValueError):
            run.success_envelope(self.rows('can', 9))

    def test_successes_without_a_grasp_sample_are_not_counted(self):
        rows = self.rows('can', 10)
        rows[0]['grasp_sample'] = None
        with self.assertRaises(ValueError):
            run.success_envelope(rows)

    def test_gate1_fraction_is_measured_against_the_other_policys_envelope(self):
        # theirs: x = 0.000 .. 0.099, so p1 to p99 = 0.001 .. 0.098; mine: x = 0.000 .. 0.019, yaw = 0 .. 19.
        # Only mine's first trial (x = 0.000) is below the lower bound, and yaw 0 .. 19 is inside theirs (1 .. 99)
        # except yaw 0 of that same trial.
        env = run.success_envelope(self.rows('can', 100))
        self.assertAlmostEqual(run.gate1_fraction(self.rows('can', 20), env), 19 / 20)
        self.assertEqual(run.gate1_fraction(self.rows('can', 20, offset=5.0), env), 0.0)

    def test_gate1_ignores_failures_and_trials_without_a_grasp(self):
        env = run.success_envelope(self.rows('can', 100))
        rows = self.rows('can', 20, offset=0.01) + self.rows('can', 10, offset=5.0, success=False)
        rows[0]['grasp_sample'] = None
        # 20 successes: rows[0] has no grasp sample and counts as outside (documented), the other 19 (yaw 1 .. 19 inside
        # 0.99 .. 98.01) are inside; the ten far-away failures are not successes and must not count: 19 / 20.
        self.assertEqual(run.gate1_fraction(rows, env), 0.95)


class GateTests(unittest.TestCase):
    def test_gate1_needs_both_directions(self):
        self.assertTrue(run.gate1_passes(0.8, 0.95))
        self.assertFalse(run.gate1_passes(0.79, 0.95))
        self.assertFalse(run.gate1_passes(0.95, 0.5))

    def test_gate0_summary_requires_every_trial(self):
        self.assertTrue(run.gate0_ok([dict(gate0=True), dict(gate0=True)]))
        self.assertFalse(run.gate0_ok([dict(gate0=True), dict(gate0=False)]))
        self.assertFalse(run.gate0_ok([]))


if __name__ == '__main__':
    unittest.main()
