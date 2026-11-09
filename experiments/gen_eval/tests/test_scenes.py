"""Scene restoration and one expert attempt per job; robosuite image. The texture folder and the sealed P2 data must be mounted."""
import unittest

import numpy as np

from experiments.gen_eval import feasibility, scenes
from experiments.gen_eval.plan import _with_conditions
from experiments.gen_scene import compose, levels
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.vla_inputs_eval import scenes as m0_scenes
from helpers import layouts, make_layout


class Restore(unittest.TestCase):
    def test_nominal_is_m0s_restore_byte_for_byte(self):
        layout = make_layout(1, "milk", ("can",))
        model, data, resolved = scenes.restore(layout, NOMINAL)
        m_model, m_data, names = m0_scenes.restore(layout)
        self.assertEqual(resolved["names"], names)
        np.testing.assert_array_equal(data.qpos, m_data.qpos)
        np.testing.assert_array_equal(data.qvel, m_data.qvel)
        np.testing.assert_array_equal(data.ctrl, m_data.ctrl)

    def test_names_put_the_target_first_then_the_layout_then_the_extras_then_the_primitives(self):
        layout = _with_conditions(layouts(8))[2]
        condition = Condition.from_dict(layout["axes"]["distractors"]["condition"])
        model, data, resolved = scenes.restore(layout, condition)
        self.assertEqual(resolved["names"], [layout["object"], *(d["object"] for d in layout["distractors"]),
                                             *(p[0] for p in condition.extra_props), *(f"{s}_{c}" for s, c, *_ in condition.primitives)])
        self.assertGreater(len(condition.extra_props) + len(condition.primitives), 0)

    def test_a_lemon_may_settle_a_few_mm_other_objects_may_not(self):
        self.assertEqual((scenes.POSITION_TOLERANCE_M, scenes.SETTLE_TOLERANCE_M), (1e-3, {"lemon": 5e-3}))
        layout = make_layout(2, "can", ("bread", "lemon"))
        for level in ("table_up5", "table_up10"):       # the lemon settles 1.7 mm off its plan pose at +10 cm
            scenes.restore(layout, levels.spatial(level, layout, "x", 0))
        original = scenes.SETTLE_TOLERANCE_M
        scenes.SETTLE_TOLERANCE_M = {}
        try:
            with self.assertRaises(ValueError):
                scenes.restore(layout, levels.spatial("table_up10", layout, "x", 0))
        finally:
            scenes.SETTLE_TOLERANCE_M = original

    def test_a_drifting_object_is_refused(self):
        layout = make_layout(1, "milk", ("can",))
        original = scenes.POSITION_TOLERANCE_M
        scenes.POSITION_TOLERANCE_M = -1.0
        try:
            with self.assertRaises(ValueError):
                scenes.restore(layout, NOMINAL)
        finally:
            scenes.POSITION_TOLERANCE_M = original
        original = scenes.YAW_TOLERANCE_DEG
        scenes.YAW_TOLERANCE_DEG = -1.0
        try:
            with self.assertRaises(ValueError):
                scenes.restore(layout, NOMINAL)
        finally:
            scenes.YAW_TOLERANCE_DEG = original

    def test_the_table_height_condition_is_restored_with_its_objects_on_the_table(self):
        layout = make_layout(1, "milk", ("can",))
        condition = levels.spatial("table_up10", layout, "x", 0)
        model, data, resolved = scenes.restore(layout, condition)
        self.assertAlmostEqual(compose.table_top_z(model), 0.10, places=12)
        self.assertGreater(float(data.xpos[model.body("prop_milk").id][2]), 0.10)

    def test_the_start_override_is_what_is_restored(self):
        layout = make_layout(3, "can", ("bread",))
        condition = levels.spatial("outer_start", layout, layout["identity"], 0)
        model, data, resolved = scenes.restore(layout, condition)
        self.assertLess(abs(float(data.xpos[model.body("prop_can").id][0]) - condition.start[0]), 2e-3)


class OneAttempt(unittest.TestCase):
    def test_the_nominal_job_of_a_layout_is_feasible_and_the_row_names_what_happened(self):
        layout = _with_conditions(layouts(8))[0]
        row = feasibility.run_one(layout, "nominal")
        self.assertEqual((row["feasible"], row["reason"], row["frames"]), (True, "success", 452))

    def test_every_job_kind_runs(self):
        planned = _with_conditions(layouts(8))
        layout = next(l for l in planned if l["axes"]["disturbance"]["level"] == "move")
        for job in feasibility.jobs_of(layout):
            row = feasibility.run_one(layout, job)
            self.assertIn("feasible", row, job)
            self.assertTrue(row["feasible"], (job, row))

    def test_an_unreachable_start_is_infeasible_and_says_so(self):
        layout = dict(make_layout(1, "milk", ("can",)), start=[0.95, 0.0, 0.0])
        row = feasibility.run_one(layout, "nominal")
        self.assertFalse(row["feasible"])
        self.assertNotEqual(row["reason"], "success")

    def test_the_move_job_runs_from_the_post_move_pose(self):
        planned = _with_conditions(layouts(8))
        layout = next(l for l in planned if l["axes"]["disturbance"]["level"] == "move")
        moved, _ = feasibility.job_scene(layout, "move")
        self.assertNotEqual(moved["start"][:2], layout["start"][:2])
        self.assertEqual(moved["start"][2], layout["start"][2])


if __name__ == "__main__":
    unittest.main()
