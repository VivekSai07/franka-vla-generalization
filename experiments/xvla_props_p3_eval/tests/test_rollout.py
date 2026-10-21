"""Robosuite image only (MuJoCo 3.4.0): the real scene restore and the real rollout loop with stub policies."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import torch
    from experiments.xvla_props_p3_eval import analysis, feasibility, protocol, rollout, scenes
    AVAILABLE = mujoco.__version__ == '3.4.0' and protocol.PLAN_JSON.exists()
except Exception:  # pragma: no cover - admission image
    AVAILABLE = False


class Identity:
    def __call__(self, x):
        return x


class Parked:
    """Holds the arm where it is with the gripper open: lifts nothing."""
    def reset(self):
        pass

    def select_action(self, processed):
        state = processed['observation.state']
        return torch.cat([state[:7], torch.tensor([255.0])])[None]


class Replay:
    """Plays a recorded action sequence, one per tick, as the policy."""
    def __init__(self, actions):
        self.actions, self.tick = actions, 0

    def reset(self):
        self.tick = 0

    def select_action(self, processed):
        action = self.actions[min(self.tick, len(self.actions) - 1)]
        self.tick += 1
        return torch.tensor(action, dtype=torch.float32)[None]


def expert_actions(layout):
    """The scripted expert's own action sequence on `layout`, exactly as the collector would run it."""
    from experiments.props_p3_data import episode
    frames = []
    draw = {'seed': layout['seed'], 'home_pose_jitter_deg': list(layout['home_pose_jitter_deg'])}
    result = episode.run_attempt(layout, draw, lambda frame, trace: frames.append(frame['action'].copy()), render=False)
    return result, np.stack(frames)


def run(policy, trial, max_ticks=None, **overrides):
    folder = Path(tempfile.mkdtemp())
    model, data, names = scenes.restore(trial['layout_id'])
    old = rollout.MAX_TICKS
    rollout.MAX_TICKS = max_ticks or old
    try:
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            row = rollout.rollout(policy, Identity(), Identity(), model, data, names, renderer, {**trial, **overrides}, folder)
    finally:
        rollout.MAX_TICKS = old
    return row, folder


def trial_of(obj, tier):
    return next(t for t in protocol.select_trials(tier) if t['object'] == obj)


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P3 plan')
class SceneTests(unittest.TestCase):
    def test_every_reserved_layout_restores_within_tolerance_and_deterministically(self):
        holdout = protocol._plan()['evaluation_holdout']
        self.assertEqual(len(holdout), 288)
        # Grouped by (object, arrangement) so the bounded model cache is rebuilt at most twelve times.
        for key in sorted(holdout, key=lambda k: (holdout[k]['object'], holdout[k]['blue_slot'], k)):
            model, data, names = scenes.restore(key)
            layout = holdout[key]
            self.assertEqual(names, [layout['object']])
            self.assertLessEqual(len(scenes._KEYS), scenes.MAX_CACHED_MODELS)
            if key in ('screen_000', 'report_000', 'report_239'):
                again = scenes.restore(key)[1]
                self.assertTrue(np.array_equal(data.qpos, again.qpos))

    def test_unreserved_layout_is_refused(self):
        with self.assertRaises(ValueError):
            scenes.restore('train_000')
        with self.assertRaises(ValueError):
            scenes.restore('episode_000')

    def test_the_blue_bin_is_on_the_planned_side_for_the_restored_scene(self):
        holdout = protocol._plan()['evaluation_holdout']
        for slot, sign in (('left', 1), ('right', -1)):
            key = next(k for k, v in sorted(holdout.items()) if v['blue_slot'] == slot)
            model, data, _ = scenes.restore(key)
            self.assertGreater(sign * data.xpos[model.body('bin_blue').id][1], 0.3, slot)


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P3 plan')
class FeasibilityTests(unittest.TestCase):
    def test_one_expert_attempt_on_a_reserved_layout_runs_and_summarises(self):
        hold = protocol._plan()['evaluation_holdout']
        key = protocol.screen_trials()[0]['layout_id']
        result = feasibility.run_one(hold[key])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['frames'], 452)
        summary = feasibility.summarize({key: result}, hold)
        self.assertEqual((summary['keys'], summary['succeeded'], summary['failed_layouts']), (1, 1, []))


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P3 plan')
class RolloutTests(unittest.TestCase):
    def test_parked_policy_runs_the_real_loop_and_is_never_lifted(self):
        trial = protocol.screen_trials()[0]
        row, folder = run(Parked(), trial, max_ticks=20)
        for name in ('cameras.mp4', 'trace.npz', 'episode.json'):
            self.assertTrue((folder/name).stat().st_size > 0, name)
        self.assertEqual((row['outcome'], row['task_success'], row['cause'], row['ticks'], row['placed_bin']),
                         ('timeout', False, 'never_lifted', 20, None))
        self.assertEqual(json.loads((folder/'episode.json').read_text()), json.loads(json.dumps(row)))
        trace = np.load(folder/'trace.npz')
        self.assertEqual(trace['props_xyz'].shape, (20, 1, 3))
        self.assertEqual(row['names'], [trial['object']])
        self.assertEqual(row['instruction'], 'sort the item')

    def test_expert_actions_replayed_as_the_policy_succeed_for_every_object_group(self):
        # One drink and one food among the trained objects, and one of each among the held-out ones.
        seen = []
        for obj, tier in (('can', 'screen'), ('bread', 'screen'), ('bottle', 'report'), ('cereal', 'report')):
            trial = trial_of(obj, tier)
            layout = protocol._plan()['evaluation_holdout'][trial['layout_id']]
            result, actions = expert_actions(layout)
            self.assertTrue(result['success'], (obj, result['reason']))
            row, _ = run(Replay(actions), trial, max_ticks=len(actions) + 60)
            self.assertTrue(row['task_success'], (obj, row))
            self.assertEqual((row['outcome'], row['cause'], row['placed_bin']), ('success', 'success', trial['bin']), obj)
            seen.append(row)
        read = analysis.readouts(seen, [])
        self.assertEqual(read['success_without_a_correct_stable_placement'], 0)
        self.assertEqual(read['bin_choice']['trained']['correct'], 2)
        self.assertEqual(read['bin_choice']['held_out']['correct'], 2)

    def test_a_real_wrong_bin_event_is_classified_wrong_bin(self):
        # The expert carries the can to its own bin; the trial declares the other bin. Real physics, not a fabricated
        # trace: the can rests stably in a bin that is not the declared one.
        trial = trial_of('can', 'screen')
        layout = protocol._plan()['evaluation_holdout'][trial['layout_id']]
        result, actions = expert_actions(layout)
        self.assertTrue(result['success'], result['reason'])
        declared = 'blue' if trial['bin'] == 'orange' else 'orange'
        row, _ = run(Replay(actions), trial, max_ticks=len(actions) + 60, bin=declared)
        self.assertFalse(row['task_success'])
        self.assertEqual((row['placed_bin'], row['cause']), (trial['bin'], 'wrong_bin'))
        read = analysis.bin_choice([row])
        self.assertEqual((read['stable'], read['wrong']), (1, 1))

    def test_names_must_be_exactly_the_one_object(self):
        trial = protocol.screen_trials()[0]
        model, data, names = scenes.restore(trial['layout_id'])
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=64, width=64) as renderer:
            for bad in ([], [trial['object'], 'milk'], ['milk' if trial['object'] != 'milk' else 'can']):
                with self.assertRaises(ValueError):
                    rollout.rollout(Parked(), Identity(), Identity(), model, data, bad, renderer, trial, Path(tmp))


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P3 plan')
class RunJobTests(unittest.TestCase):
    """`evaluate.run_job` around a stubbed rollout: what it skips, quarantines and refuses."""

    def setUp(self):
        from experiments.xvla_props_p3_eval import evaluate
        self.evaluate = evaluate
        self.directory = Path(tempfile.mkdtemp())
        self.trial = protocol.screen_trials()[0]
        self.job = dict(candidate=88830, trial=self.trial)
        self.calls = []

    def run_job(self, outcome='timeout'):
        from unittest import mock

        def fake_rollout(policy, pre, post, model, data, names, renderer, trial, folder):
            self.calls.append(folder)
            for name in self.evaluate.TRIAL_FILES:
                (folder/name).write_bytes(b'x')
            return dict(trial, outcome=outcome)

        with mock.patch.object(scenes, 'restore', return_value=(object(), object(), [self.trial['object']])), \
                mock.patch.object(rollout, 'rollout', side_effect=fake_rollout), \
                mock.patch.object(mujoco, 'Renderer'), \
                mock.patch.object(self.evaluate, 'cool_down', return_value=0), \
                mock.patch.object(self.evaluate, 'gpu_state', return_value=None):
            return self.evaluate.run_job(None, None, None, self.job, self.directory)

    def test_a_fresh_trial_runs_once_and_writes_its_telemetry(self):
        row = self.run_job()
        folder = protocol.trial_folder(self.directory, self.trial)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(row['outcome'], 'timeout')
        self.assertTrue((folder/'telemetry.json').is_file())

    def test_a_complete_trial_is_never_rerun(self):
        self.run_job()
        self.calls.clear()
        self.assertIsNone(self.run_job())
        self.assertEqual(self.calls, [])

    def test_a_half_written_trial_is_quarantined_and_rerun(self):
        self.run_job()
        folder = protocol.trial_folder(self.directory, self.trial)
        (folder/'episode.json').unlink()
        self.calls.clear()
        self.assertIsNotNone(self.run_job())
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.directory/'quarantine'/f'{folder.name}.0').is_dir())
        self.assertTrue(self.evaluate.trial_is_complete(folder))

    def test_a_nonfinite_action_is_an_error_not_a_retry(self):
        with self.assertRaisesRegex(ValueError, 'no retry'):
            self.run_job(outcome='nonfinite_action')
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main()
