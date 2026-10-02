import unittest


class TrialPlanTests(unittest.TestCase):
    def test_trials_are_b2_control_report_trials_with_b2_seeds(self):
        from experiments.robosuite_sim2sim.starts import trials
        from experiments.xvla_b2_distractors_eval import evaluate as ev
        planned = trials()
        report = {t["ordinal"]: t for t in ev.evaluation_plan("report")}
        self.assertEqual(len(planned), 20)
        self.assertEqual(len({t["layout_id"] for t in planned}), 10)
        for trial in planned:
            self.assertEqual(trial["condition"], "control")
            self.assertEqual(report[trial["ordinal"]]["layout_id"], trial["layout_id"])
            self.assertEqual(trial["seed"], 20260928 * 10 + trial["ordinal"])
            self.assertIn(trial["layout_id"], ev.report_layout_ids()[:10])

    def test_fewer_layouts_gives_fewer_trials(self):
        from experiments.robosuite_sim2sim.starts import trials
        self.assertEqual(len(trials(3)), 6)


class SummaryTests(unittest.TestCase):
    def test_paired_table_and_b2_agreement(self):
        from experiments.robosuite_sim2sim.rollout import summarize
        rows = [dict(scene="mujoco", ordinal=1, task_success=True, b2_task_success=True),
                dict(scene="mujoco", ordinal=2, task_success=False, b2_task_success=True),
                dict(scene="robosuite", ordinal=1, task_success=False, b2_task_success=True),
                dict(scene="robosuite", ordinal=2, task_success=False, b2_task_success=True)]
        summary = summarize(rows)
        self.assertEqual(summary["mujoco"], dict(trials=2, successes=1))
        self.assertEqual(summary["robosuite"], dict(trials=2, successes=0))
        self.assertEqual(summary["paired"], dict(trials=2, both=0, mujoco_only=1, robosuite_only=0, neither=1))
        self.assertEqual(summary["mujoco_arm_agrees_with_b2_report"], 1)

    def test_one_arm_only_has_no_paired_table(self):
        from experiments.robosuite_sim2sim.rollout import summarize
        summary = summarize([dict(scene="robosuite", ordinal=1, task_success=True, b2_task_success=True)])
        self.assertNotIn("paired", summary)
        self.assertNotIn("mujoco", summary)


if __name__ == "__main__":
    unittest.main()
