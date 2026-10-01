"""Distractor axis: placement, parked-pose preservation, clearance and the disturbance gate.

These cover the behaviour B2 adds on top of the B1 fork. The exact-qpos gate in particular is
what would have rejected every distractor-present scene had it not been extended (the derived
scene is compared against an expectation rebuilt from the base source).
"""
import importlib
import unittest

import numpy as np


class DistractorPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = importlib.import_module('experiments.b2_distractor_data.plan')
        cls.plan = cls.p.make_plan()

    def test_arms_are_asymmetric_and_exact(self):
        counts = {}
        for episode in self.plan['episodes']:
            counts[episode['condition']] = counts.get(episode['condition'], 0) + 1
        self.assertEqual(counts, {'control': self.p.CONTROL_COUNT,
                                  'distractor': self.p.DISTRACTOR_COUNT})

    def test_cells_balanced_within_each_arm(self):
        for split in ('train', 'validation'):
            for condition in self.p.CONDITIONS:
                group = [e for e in self.plan['episodes']
                         if e['split'] == split and e['condition'] == condition]
                counts = {c: sum(e['cell'] == c for e in group) for c in self.p.CELL_IDS}
                self.assertEqual(set(counts.values()), {len(group)//len(self.p.CELL_IDS)},
                                 f'{split}/{condition} unbalanced: {counts}')

    def test_distractor_counts_and_colours_balanced(self):
        present = [e for e in self.plan['episodes'] if e['condition'] == 'distractor']
        sizes = {n: sum(len(e['distractors']) == n for e in present) for n in (1, 2)}
        self.assertEqual(sizes, {1: len(present)//2, 2: len(present)//2})
        colours = {c: sum(c in e['distractors'] for e in present) for c in self.p.DISTRACTOR_COLOURS}
        self.assertEqual(len(set(colours.values())), 1, f'colour imbalance: {colours}')

    def test_control_episodes_carry_no_distractors(self):
        for episode in self.plan['episodes']:
            if episode['condition'] == 'control':
                self.assertEqual(episode['distractors'], [])
                self.assertEqual(episode['distractor_xy_m'], {})

    def test_riskiest_profiles_run_first(self):
        """FAILURES #29: the opening minutes of the real run must themselves be the smoke test."""
        leading = [(e['condition'], len(e['distractors'])) for e in self.plan['episodes'][:4]]
        self.assertEqual(leading, [('distractor', 2), ('distractor', 1), ('control', 0), ('distractor', 2)])

    def test_every_distractor_clears_both_targets(self):
        for episode in self.plan['episodes']:
            layout = next(x for x in self.plan['layouts'] if x['layout_id'] == episode['layout_id'])
            cube = layout['cube_xy_m']
            points = [cube[:2], cube[2:]] + list(layout['distractor_xy_m'].values())
            for i in range(len(points)):
                for j in range(i):
                    gap = float(np.hypot(points[i][0]-points[j][0], points[i][1]-points[j][1]))
                    self.assertGreaterEqual(gap, self.p.MIN_CLEARANCE_M - 1e-9,
                                            f"{episode['layout_id']} pair {i},{j} gap {gap}")

    def test_distractors_span_the_full_y_range(self):
        """Confined to the middle band they would be predictable from position (FAILURES #27)."""
        ys = [xy[1] for e in self.plan['episodes'] for xy in e['distractor_xy_m'].values()]
        self.assertTrue(any(y < -0.20 for y in ys), 'no distractor in region A')
        self.assertTrue(any(y > 0.20 for y in ys), 'no distractor in region B')
        self.assertTrue(any(-0.20 <= y <= 0.20 for y in ys), 'no distractor in the middle band')

    def test_unplaceable_distractor_raises_rather_than_redrawing(self):
        seeds = {}
        original = self.p.MIN_CLEARANCE_M
        self.p.MIN_CLEARANCE_M = 10.0  # larger than the envelope, so no draw can ever clear
        try:
            with self.assertRaises(ValueError):
                self.p.draw_distractor_xy(seeds, 'impossible', ('blue',), [0.275, -0.3, 0.275, 0.3])
        finally:
            self.p.MIN_CLEARANCE_M = original


class DistractorSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = importlib.import_module('experiments.fixed_layout.source')
        cls.p = importlib.import_module('experiments.b2_distractor_data.plan')
        cls.s = importlib.import_module('experiments.b2_distractor_data.scenes')
        cls.expert = source.Expert()
        cls.base = source.load_reset(cls.expert)
        cls.model = cls.expert.model
        cls.plan = cls.p.make_plan()
        cls.layouts = {x['layout_id']: x for x in cls.plan['layouts']}

    def _adr(self, colour):
        return self.s.cube_qpos(self.model, colour)

    def _derive(self, episode):
        layout = self.layouts[episode['layout_id']]
        return self.s.derive_scene(self.model, self.base, layout,
                                   dict(home_pose_jitter_deg=episode['home_pose_jitter_deg']))

    def test_control_scene_keeps_both_distractors_bit_exactly_parked(self):
        episode = next(e for e in self.plan['episodes'] if e['condition'] == 'control')
        qpos = self._derive(episode)['qpos']
        for colour in self.p.DISTRACTOR_COLOURS:
            a = self._adr(colour)
            np.testing.assert_array_equal(qpos[a:a+7], self.base['qpos'][a:a+7])

    def test_active_distractors_sit_upright_at_target_rest_height(self):
        episode = next(e for e in self.plan['episodes'] if len(e['distractors']) == 2)
        qpos = self._derive(episode)['qpos']
        red = self._adr('red')
        for colour in episode['distractors']:
            a = self._adr(colour)
            self.assertAlmostEqual(qpos[a+2], self.base['qpos'][red+2], places=12)
            np.testing.assert_array_equal(qpos[a+3:a+7], self.base['qpos'][red+3:red+7])
            self.assertLess(abs(qpos[a]), 1.0, 'distractor still parked outside the workspace')

    def test_single_distractor_leaves_the_other_colour_parked(self):
        episode = next(e for e in self.plan['episodes'] if len(e['distractors']) == 1)
        qpos = self._derive(episode)['qpos']
        unused = next(c for c in self.p.DISTRACTOR_COLOURS if c not in episode['distractors'])
        a = self._adr(unused)
        np.testing.assert_array_equal(qpos[a:a+7], self.base['qpos'][a:a+7])

    def test_undeclared_distractor_movement_is_rejected(self):
        """The exact-qpos gate must still bind: a placement not in the layout is a mismatch."""
        episode = next(e for e in self.plan['episodes'] if e['condition'] == 'control')
        layout = dict(self.layouts[episode['layout_id']])
        jitter = dict(home_pose_jitter_deg=episode['home_pose_jitter_deg'])
        original = self.s._apply_distractors

        def sneak(model, qpos, base_source, _layout):
            a = self.s.cube_qpos(model, 'blue')
            qpos[a:a+2] = [0.30, 0.0]
            qpos[a+2] = base_source['qpos'][self.s.cube_qpos(model, 'red')+2]

        self.s._apply_distractors = lambda m, q, b, l: (
            sneak(m, q, b, l) if q is not None and l is layout else None)
        try:
            with self.assertRaises(ValueError):
                self.s.derive_scene(self.model, self.base, layout, jitter)
        finally:
            self.s._apply_distractors = original

    def test_clearance_violation_is_rejected(self):
        episode = next(e for e in self.plan['episodes'] if len(e['distractors']) == 1)
        layout = dict(self.layouts[episode['layout_id']])
        colour = episode['distractors'][0]
        layout['distractor_xy_m'] = {colour: [layout['cube_xy_m'][0], layout['cube_xy_m'][1]]}
        with self.assertRaises(ValueError):
            self.s.derive_scene(self.model, self.base, layout,
                                dict(home_pose_jitter_deg=episode['home_pose_jitter_deg']))

    def test_placed_distractors_start_at_rest(self):
        """The parked pose carries velocity, not just an odd position."""
        import mujoco
        from data_collection.benchmark import restore_state
        episode = next(e for e in self.plan['episodes'] if len(e['distractors']) == 2)
        arrays = self._derive(episode)
        data = mujoco.MjData(self.model)
        restore_state(self.model, data, arrays['initial_state'])
        for colour in episode['distractors']:
            dof = self.s._distractor_dof(self.model, colour)
            np.testing.assert_array_equal(data.qvel[dof:dof+6], np.zeros(6))

    def test_placed_distractors_stay_put_under_physics(self):
        """Without clearing the inherited velocity the cube is launched on the very first tick."""
        import mujoco
        from data_collection.benchmark import restore_state
        episode = next(e for e in self.plan['episodes'] if len(e['distractors']) == 2)
        arrays = self._derive(episode)
        data = mujoco.MjData(self.model)
        restore_state(self.model, data, arrays['initial_state'])
        before = {c: data.xpos[self.model.body(f'target_cube_{c}').id].copy()
                  for c in episode['distractors']}
        data.ctrl[:] = arrays['initial_ctrl']
        for _ in range(60 * 20):  # two seconds of control ticks at 20 substeps
            mujoco.mj_step(self.model, data)
        for colour, was in before.items():
            now = data.xpos[self.model.body(f'target_cube_{colour}').id]
            self.assertLess(float(np.linalg.norm(now - was)), 0.02,
                            f'{colour} did not stay where it was placed')

    def test_reserved_layouts_are_banked_in_both_conditions(self):
        holdout = self.plan['evaluation_holdout']
        self.assertEqual(len(holdout), self.p.EVAL_HOLDOUT_COUNT)
        for entry in holdout.values():
            self.assertEqual(entry['conditions']['control'], {})
            self.assertEqual(sorted(entry['conditions']['distractor']), sorted(entry['distractors']))

    def test_screen_and_report_tiers_are_disjoint(self):
        screen = set(self.plan['evaluation_screen_ids'])
        report = set(self.plan['evaluation_report_ids'])
        self.assertEqual(len(screen), self.p.EVAL_SCREEN_COUNT)
        self.assertEqual(len(report), self.p.EVAL_REPORT_COUNT)
        self.assertEqual(screen & report, set())


class DisturbanceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = importlib.import_module('experiments.b2_distractor_data.plan')
        cls.v = importlib.import_module('experiments.b2_distractor_data.validate')

    def test_gate_passes_below_threshold_and_fails_above(self):
        below = {'distractor_disturbance_m': {'blue': self.p.DISTRACTOR_DISTURBANCE_M - 1e-6}}
        self.v.require_distractors_undisturbed(below)
        above = {'distractor_disturbance_m': {'blue': self.p.DISTRACTOR_DISTURBANCE_M + 1e-6}}
        with self.assertRaises(ValueError) as caught:
            self.v.require_distractors_undisturbed(above)
        # record.py converts exactly this marker into a retryable rejection.
        self.assertIn('distractor disturbed:', str(caught.exception))

    def test_control_episode_has_nothing_to_disturb(self):
        self.v.require_distractors_undisturbed({'distractor_disturbance_m': {}})


if __name__ == '__main__':
    unittest.main()
