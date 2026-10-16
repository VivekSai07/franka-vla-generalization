"""Robosuite image only (MuJoCo 3.4.0): the real scene restore and the real rollout loop with stub policies."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import torch
    from experiments.xvla_props_p2_eval import protocol, rollout, scenes
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
    from experiments.props_p2_data import episode
    frames = []
    draw = {'seed': layout['seed'], 'home_pose_jitter_deg': list(layout['home_pose_jitter_deg'])}
    result = episode.run_attempt(layout, draw, lambda frame, trace: frames.append(frame['action'].copy()), render=False)
    return result, np.stack(frames)


def run(policy, trial, max_ticks=None):
    import mujoco
    folder = Path(tempfile.mkdtemp())
    model, data, names = scenes.restore(trial['layout_id'])
    old = rollout.MAX_TICKS
    rollout.MAX_TICKS = max_ticks or old
    try:
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            row = rollout.rollout(policy, Identity(), Identity(), model, data, names, renderer, trial, folder)
    finally:
        rollout.MAX_TICKS = old
    return row, folder


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P2 plan')
class SceneTests(unittest.TestCase):
    def test_every_reserved_layout_restores_within_tolerance_and_deterministically(self):
        holdout = protocol._plan()['evaluation_holdout']
        self.assertEqual(len(holdout), 120)
        for key in sorted(holdout):
            model, data, names = scenes.restore(key)
            layout = holdout[key]
            self.assertEqual(names[0], layout['object'])
            self.assertEqual(len(names), 1 + layout['distractor_count'])
            if key in ('screen_000', 'report_000'):
                again = scenes.restore(key)[1]
                self.assertTrue(np.array_equal(data.qpos, again.qpos))

    def test_unreserved_layout_is_refused(self):
        with self.assertRaises(ValueError):
            scenes.restore('episode_000')


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P2 plan')
class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = next(t for t in protocol.screen_trials() if t['distractor_count'] == 2)
        cls.layout = protocol._plan()['evaluation_holdout'][cls.trial['layout_id']]

    def test_parked_policy_runs_the_real_loop_and_is_never_lifted(self):
        row, folder = run(Parked(), self.trial, max_ticks=20)
        for name in rollout.TRIAL_FILES if hasattr(rollout, 'TRIAL_FILES') else ('cameras.mp4', 'trace.npz', 'episode.json'):
            self.assertTrue((folder/name).stat().st_size > 0, name)
        self.assertEqual((row['outcome'], row['task_success'], row['cause'], row['ticks']), ('timeout', False, 'never_lifted', 20))
        self.assertEqual(json.loads((folder/'episode.json').read_text()), json.loads(json.dumps(row)))
        trace = np.load(folder/'trace.npz')
        self.assertEqual(trace['props_xyz'].shape, (20, 3, 3))
        self.assertEqual(row['names'][0], self.trial['object'])
        self.assertIsNone(row['first_lifted'])
        self.assertFalse(row['wrong_object'])

    def test_expert_actions_replayed_as_the_policy_succeed(self):
        result, actions = expert_actions(self.layout)
        self.assertTrue(result['success'], result['reason'])
        row, folder = run(Replay(actions), self.trial, max_ticks=len(actions) + 60)
        self.assertTrue(row['task_success'], row)
        self.assertEqual((row['outcome'], row['cause']), ('success', 'success'))
        self.assertEqual(row['first_lifted'], self.trial['object'])
        self.assertTrue(all(v < 0.02 for v in row['distractor_displacement_m'].values()), row['distractor_displacement_m'])

    def test_a_real_wrong_object_event_is_classified_wrong_object(self):
        # The expert carries a distractor to the bin; the trial asks for the original target. This is the failure
        # the policy is being tested for, produced by physics rather than by a fabricated trace.
        # Same objects at the same poses as the restored scene, roles swapped: the first distractor becomes the
        # expert's target and the real target becomes its distractor.
        layout = self.layout
        swapped_target = layout['distractors'][0]['object']
        swapped = dict(layout, object=swapped_target, start=layout['distractors'][0]['pose'],
                       distractors=[dict(object=layout['object'], pose=layout['start']), *layout['distractors'][1:]])
        result, actions = expert_actions(swapped)
        self.assertTrue(result['success'], result['reason'])
        row, _ = run(Replay(actions), self.trial, max_ticks=len(actions) + 60)
        self.assertFalse(row['task_success'])
        self.assertEqual(row['first_lifted'], swapped_target)
        self.assertEqual(row['cause'], 'wrong_object')
        self.assertTrue(row['wrong_object'])


if __name__ == '__main__':
    unittest.main()
