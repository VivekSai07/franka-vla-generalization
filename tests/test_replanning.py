import unittest

import numpy as np

from validation.replanning_probe import transition_probes, expected_chunk_calls


class ReplanningTests(unittest.TestCase):
    def test_transition_probes_use_actual_commands_and_stay_inside_episode(self):
        actions = np.zeros((180, 8))
        actions[:40, 7] = 255
        actions[100:, 7] = 255
        probes = transition_probes(actions, 50)
        self.assertEqual(sorted({p["transition_index"] for p in probes}), [40, 100])
        self.assertEqual({p["transition"] for p in probes}, {"close", "release"})
        self.assertTrue(all(0 <= p["frame_index"] <= 130 for p in probes))
        self.assertTrue(any(p["frame_index"] == 40 for p in probes))

    def test_chunk_counts_cover_partial_final_chunk(self):
        self.assertEqual(expected_chunk_calls(600, 10), 60)
        self.assertEqual(expected_chunk_calls(600, 1), 600)
        self.assertEqual(expected_chunk_calls(31, 10), 4)


if __name__ == "__main__":
    unittest.main()