import unittest

import numpy as np

from experiments.xvla_props_p1_eval import protocol, scenes


class RestoreTests(unittest.TestCase):
    def setUp(self):
        if not protocol.PLAN_JSON.exists():
            self.skipTest('Props P1 parent not mounted')

    def _a_few_keys(self):
        plan = protocol._plan()
        return (plan['evaluation_screen_ids'][0], plan['evaluation_report_ids'][0],
                plan['evaluation_confirm_ids'][0])

    def test_restore_matches_the_plan_within_tolerance(self):
        plan = protocol._plan()
        for key in self._a_few_keys():
            layout = plan['evaluation_holdout'][key]
            model, data = scenes.restore(key)
            from experiments.props_scene.scene import object_pose, present_objects
            self.assertEqual(present_objects(model), [layout['object']])
            pose = object_pose(model, data, layout['object'])
            x, y, yaw = layout['start']
            self.assertLess(float(np.hypot(pose['xy'][0]-x, pose['xy'][1]-y)), scenes.POSITION_TOLERANCE_M, key)
            self.assertLess(scenes._yaw_difference_deg(pose['yaw_deg'], yaw), scenes.YAW_TOLERANCE_DEG, key)

    def test_restore_rejects_an_unknown_key(self):
        with self.assertRaises(ValueError):
            scenes.restore('screen_999')
        with self.assertRaises(ValueError):
            scenes.restore('not_a_real_tier_000')

    def test_restore_is_repeatable(self):
        key = self._a_few_keys()[0]
        _, first = scenes.restore(key)
        qpos_first = first.qpos.copy()
        first.qpos[:] = 0
        _, second = scenes.restore(key)
        np.testing.assert_array_equal(second.qpos, qpos_first)

    def test_every_reserved_layout_restores(self):
        """All 80 keys build and restore without raising; slow-ish, exercises every (object, slot)."""
        plan = protocol._plan()
        for key in plan['evaluation_holdout']:
            model, data = scenes.restore(key)
            self.assertIsNotNone(model)
            self.assertIsNotNone(data)


if __name__ == '__main__':
    unittest.main()
