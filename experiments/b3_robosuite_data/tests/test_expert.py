import unittest

import numpy as np

from experiments.b3_robosuite_data.plan import attempt_draw, make_plan


class ExpertEpisodeTests(unittest.TestCase):
    def test_riskiest_planned_episodes(self):
        from experiments.b3_robosuite_data.expert import run_attempt
        plan = make_plan()
        outcomes = []
        for episode in plan["episodes"][:4]:
            frames = []
            result = run_attempt(episode, attempt_draw(episode, 0),
                                 lambda frame, trace: frames.append(frame["action"].copy()))
            outcomes.append((episode["episode_index"], episode["profile"], episode["edge"],
                             result["success"], result["reason"], result["final_xy_m"], result["final_z_m"]))
            print("OUTCOME", outcomes[-1], "frames", result["frames"])
            self.assertEqual(len(frames), result["frames"])
            if not result["success"]:
                self.assertTrue(result["reason"])
            else:
                self.assertEqual(result["frames"], 528)
                self.assertLess(result["final_xy_m"], 0.02)
                self.assertTrue(0.035 < result["final_z_m"] < 0.045)
        control = next(o for o in outcomes if o[1] == [0, "none"])
        self.assertTrue(control[3], f"the control must succeed: {control}")


if __name__ == "__main__":
    unittest.main()
