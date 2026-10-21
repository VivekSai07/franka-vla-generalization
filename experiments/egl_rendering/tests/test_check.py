import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.egl_rendering import check

OK_PAIR = dict(images=904, images_differing=0, pixels_differing=0, max_abs=0)


class RendererGuardTests(unittest.TestCase):
    """A run labelled one renderer must really use it: a mislabelled run would make the whole comparison meaningless."""

    def test_matching_environment_is_accepted(self):
        check.require_renderer('egl', {'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'})
        check.require_renderer('osmesa', {'MUJOCO_GL': 'osmesa'})

    def test_wrong_or_missing_renderer_is_refused(self):
        with self.assertRaises(SystemExit):
            check.require_renderer('egl', {'MUJOCO_GL': 'osmesa'})
        with self.assertRaises(SystemExit):
            check.require_renderer('osmesa', {})

    def test_pyopengl_platform_is_not_required(self):
        # measured 2026-10-19: rendering and Renderer.close() work with and without it; only the label must match
        check.require_renderer('egl', {'MUJOCO_GL': 'egl'})

    def test_unknown_renderer_name_is_refused(self):
        with self.assertRaises(SystemExit):
            check.require_renderer('glfw', {'MUJOCO_GL': 'glfw'})


class VerdictTests(unittest.TestCase):
    def verdict(self, **over):
        args = dict(osmesa_pair=OK_PAIR, physics_differences=[], egl_pairs=[OK_PAIR] * 3, layers_added=1)
        args.update(over)
        return check.verdict(**args)

    def test_all_three_pass(self):
        v = self.verdict()
        self.assertEqual((v['C1'], v['C2'], v['C3'], v['admitted']), (True, True, True, True))

    def test_c1_needs_byte_identical_osmesa_and_exactly_one_layer(self):
        self.assertFalse(self.verdict(osmesa_pair={**OK_PAIR, 'max_abs': 1, 'images_differing': 1})['C1'])
        self.assertFalse(self.verdict(layers_added=2)['C1'])
        self.assertFalse(self.verdict(layers_added=0)['C1'])

    def test_c2_any_physics_difference_fails_and_names_nothing_away(self):
        v = self.verdict(physics_differences=['qpos'])
        self.assertFalse(v['C2'])
        self.assertFalse(v['admitted'])

    def test_c3_follows_the_repeatability_limits(self):
        self.assertFalse(self.verdict(egl_pairs=[OK_PAIR, {**OK_PAIR, 'max_abs': 3, 'images_differing': 1}])['C3'])
        self.assertTrue(self.verdict(egl_pairs=[{**OK_PAIR, 'max_abs': 1, 'images_differing': 6}] * 3)['C3'])

    def test_admission_needs_all_three_and_c4_c5_gate_nothing(self):
        self.assertFalse(self.verdict(egl_pairs=[{**OK_PAIR, 'max_abs': 50, 'images_differing': 900}])['admitted'])
        self.assertTrue(self.verdict()['admitted'])


if __name__ == '__main__':
    unittest.main()
