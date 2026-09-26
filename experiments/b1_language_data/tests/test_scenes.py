"""B1 language scenes: cube XY + arm home-pose jitter, 361 unique bank (2 arrangements)."""
from contextlib import ExitStack
import copy
import importlib
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.fixed_layout.source import Expert, load_reset


class SceneTests(unittest.TestCase):
    def setUp(self):
        try:
            self.s = importlib.import_module('experiments.b1_language_data.scenes')
            self.p = importlib.import_module('experiments.b1_language_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing broad-workspace scenes: {exc}')
        guards = ExitStack(); self.addCleanup(guards.close)
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
            guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('Forbidden native execution')))
        guards.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('Forbidden expert execution')))
        self.expert = Expert()
        self.source = load_reset(self.expert)

    def test_derive_scene_cube_xy_and_arm_jitter_applied(self):
        p = self.p.make_plan(); layout = p['layouts'][0]
        jitter = dict(home_pose_jitter_deg=[0.5, -0.3, 0.2, -0.7, 0.1, -0.4, 0.9])
        arrays = self.s.derive_scene(self.expert.model, self.source, layout, jitter)
        for n, color in enumerate(('red', 'green')):
            from experiments.local_position_probe.scenes import cube_qpos
            start = cube_qpos(self.expert.model, color)
            self.assertAlmostEqual(arrays['qpos'][start], layout['cube_xy_m'][2*n], places=15)
            self.assertAlmostEqual(arrays['qpos'][start+1], layout['cube_xy_m'][2*n+1], places=15)
        expected_arm = self.source['qpos'][:7] + np.deg2rad(np.asarray(jitter['home_pose_jitter_deg']))
        np.testing.assert_array_equal(arrays['qpos'][:7], expected_arm)
        # initial_ctrl unaffected by qpos change; wrist-mounted gripper_cam may move.
        self.assertEqual(arrays['initial_ctrl'].tobytes(), self.source['initial_ctrl'].tobytes())

    def test_derive_scene_base_pose_zero_jitter_matches_source_arm(self):
        p = self.p.make_plan()
        arrays = self.s.derive_scene(self.expert.model, self.source, p['base_layout'],
                                     dict(home_pose_jitter_deg=[0.0]*7))
        np.testing.assert_array_equal(arrays['qpos'][:7], self.source['qpos'][:7])

    def test_bank_has_361_unique_digests(self):
        p = self.p.make_plan()
        bank = self.s.make_bank(self.expert.model, self.source, p)
        self.assertEqual(len(bank), 361)  # 200 episodes + 80 layouts x 2 arrangements + base
        digests = {self.s.scene_hash(a) for a in bank.values()}
        self.assertEqual(len(digests), 361)
        for e in p['episodes']:
            self.assertIn(f"episode_{e['episode_index']:03d}", bank)
        for lid in p['evaluation_layout_ids']:
            for arrangement in self.p.ARRANGEMENTS:
                self.assertIn(self.s.eval_bank_key(lid, arrangement), bank)
        self.assertIn('base', bank)

    def test_each_reserved_layout_is_banked_in_both_arrangements(self):
        """Every evaluation scene must be hash-proven and sealed, in both colour arrangements —
        the evaluator never constructs a scene itself."""
        p = self.p.make_plan()
        bank = self.s.make_bank(self.expert.model, self.source, p)
        for lid in p['evaluation_layout_ids']:
            keys = [self.s.eval_bank_key(lid, a) for a in self.p.ARRANGEMENTS]
            self.assertEqual(len(keys), 2)
            # the two arrangements must be genuinely different scenes
            self.assertNotEqual(self.s.scene_hash(bank[keys[0]]), self.s.scene_hash(bank[keys[1]]))
            placed = p['evaluation_holdout'][lid]['arrangements']
            self.assertEqual(placed['familiar'][:2], placed['swapped'][2:])
            self.assertEqual(placed['familiar'][2:], placed['swapped'][:2])

    def test_evaluation_layouts_disjoint_from_executed(self):
        p = self.p.make_plan()
        executed = {e['layout_id'] for e in p['episodes']}
        self.assertFalse(executed & set(p['evaluation_layout_ids']))

    def test_home_pose_jitter_out_of_range_rejected(self):
        p = self.p.make_plan(); layout = p['layouts'][0]
        for bad in (dict(home_pose_jitter_deg=[1.5]+[0.0]*6),
                    dict(home_pose_jitter_deg=[0.0]*6),
                    dict(home_pose_jitter_deg=[float('nan')]+[0.0]*6)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.s.derive_scene(self.expert.model, self.source, layout, bad)

    def test_cube_xy_out_of_envelope_rejected(self):
        p = self.p.make_plan(); layout = copy.deepcopy(p['layouts'][0])
        layout['cube_xy_m'][0] = 0.10
        with self.assertRaises(ValueError):
            self.s.derive_scene(self.expert.model, self.source, layout,
                                dict(home_pose_jitter_deg=[0.0]*7))


if __name__ == '__main__':
    unittest.main()
