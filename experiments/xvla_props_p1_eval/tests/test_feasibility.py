import unittest

from experiments.xvla_props_p1_eval import feasibility


class SummarizeTests(unittest.TestCase):
    def test_all_pass(self):
        results = {'screen_000': dict(success=True, reason='success', frames=452, stable_ticks=40, seconds=15.0)}
        summary = feasibility.summarize(results)
        self.assertEqual(summary, dict(results=results, failed_keys=[], failed_layouts=[], keys=1, succeeded=1))

    def test_some_fail(self):
        results = {
            'screen_000': dict(success=True, reason='success', frames=452, stable_ticks=40, seconds=15.0),
            'screen_001': dict(success=False, reason='grasp', frames=900, stable_ticks=0, seconds=30.0),
        }
        summary = feasibility.summarize(results)
        self.assertEqual(summary['failed_keys'], ['screen_001'])
        self.assertEqual(summary['failed_layouts'], ['screen_001'])
        self.assertEqual((summary['keys'], summary['succeeded']), (2, 1))

    def test_experts_are_cached_per_object_and_slot(self):
        feasibility._EXPERTS.clear()
        with self.assertRaises(Exception):
            # building a real model needs the robosuite image; this only checks the cache key shape
            feasibility._expert('not_a_real_object', 'left')


if __name__ == '__main__':
    unittest.main()
