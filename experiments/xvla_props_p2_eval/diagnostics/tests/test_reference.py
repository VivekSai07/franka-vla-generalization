"""Robosuite image only: the expert grasp envelope by forward kinematics over the sealed P2 episodes."""
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    MUJOCO = mujoco.__version__ == '3.4.0'
except ImportError:  # admission image: no simulator
    MUJOCO = False

EPISODES = Path('/vla/outputs/props_p2_data/props_p2_20261013/episodes')
if MUJOCO:
    from experiments.xvla_props_p2_eval.diagnostics import reference
    AVAILABLE = EPISODES.is_dir()
else:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the sealed P2 episodes')
class ReferenceTests(unittest.TestCase):
    def test_sample_is_finite_deterministic_and_on_the_hand_axis(self):
        a = reference.expert_grasp_sample(EPISODES/'episode_000.npz')
        b = reference.expert_grasp_sample(EPISODES/'episode_000.npz')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4,))
        self.assertTrue(np.isfinite(a).all())
        # measured 2026-10-19 on one episode per object: the target lies 0.11-0.17 m along the hand axis, within 1.5 cm
        # of it sideways; a closure tick that violates this is not a grasp.
        self.assertTrue(0.05 < a[2] < 0.25, a)
        self.assertLess(np.hypot(a[0], a[1]), 0.03)

    def test_the_closure_tick_is_the_grasp_on_every_object(self):
        # Independent evidence for the closure definition: after the closure the target keeps its place in the hand
        # frame (it is held) and then leaves the table (it is carried).
        seen = {}
        for path in sorted(EPISODES.glob('episode_*.npz')):
            name = str(np.load(path)['object'])
            if name not in seen:
                seen[name] = reference.expert_hold_check(path)
            if len(seen) == 6:
                break
        self.assertEqual(len(seen), 6)
        for name, check in seen.items():
            self.assertLess(check['drift_m'], 0.01, f'{name}: target moves {check["drift_m"]:.3f} m in the hand after closing')
            self.assertGreater(check['rise_m'], 0.10, f'{name}: target rises only {check["rise_m"]:.3f} m after closing')

    def test_building_over_many_layouts_keeps_the_compiled_model_cache_bounded(self):
        # Each compiled scene is ~390 MB; the 540 episodes use many object combinations.
        from experiments.props_scene import scene as props_scene
        from experiments.xvla_props_p2_eval import scenes
        files = sorted(EPISODES.glob('episode_*.npz'))[:40]
        reference.build_reference(files)
        self.assertLessEqual(props_scene._built.cache_info().currsize, scenes.MAX_CACHED_MODELS)

    def test_build_reference_counts_and_orders(self):
        files = sorted(EPISODES.glob('episode_*.npz'))[:24]
        ref = reference.build_reference(files)
        self.assertEqual(sum(v['n'] for v in ref.values()), 24)
        for name, v in ref.items():
            self.assertEqual(len(v['lo']), 4)
            self.assertTrue(all(lo <= hi for lo, hi in zip(v['lo'], v['hi'])), name)


if __name__ == '__main__':
    unittest.main()
