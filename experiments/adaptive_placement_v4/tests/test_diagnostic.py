"""Saved-state reconstruction only: never call the real path planner in tests."""
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np


def diagnostic():
    path = Path(__file__).resolve().parents[1]/'diagnostic.py'
    assert path.is_file(), 'saved-state numerical diagnostic is not implemented'
    return importlib.import_module('experiments.adaptive_placement_v4.diagnostic')


class DiagnosticTests(unittest.TestCase):
    def test_exact_saved_lift_and_seeded_endpoints_without_route_attempt(self):
        d = diagnostic()
        ctx = d.reconstruct()
        from data_collection.benchmark import capture_state
        np.testing.assert_array_equal(capture_state(ctx.model, ctx.data), ctx.arrays['final_state'])
        np.testing.assert_array_equal(ctx.measured_q, ctx.saved['placement']['start_q'])
        np.testing.assert_array_equal(ctx.previous_ctrl, ctx.saved['placement']['previous_target'])
        np.testing.assert_array_equal(ctx.checker.relative.homogeneous, ctx.saved['payload_hand_relative'])
        for pose, expected in zip(ctx.endpoints, ctx.saved['placement']['endpoints']):
            np.testing.assert_array_equal(pose.homogeneous, expected)
        self.assertEqual(ctx.checker.checked, 0)
        self.assertEqual(ctx.original['rejected_waypoint_index'], 71)
        self.assertAlmostEqual(ctx.original['joint_step_rad'], .0697633209375439, places=15)
        self.assertEqual(ctx.entry['seed'], 488840035)

    def test_exclusive_marker_prevents_second_call_and_preserves_failed_attempt(self):
        d = diagnostic()
        from experiments.bounded_placement_v3.solver import PlanningFailure
        called = []
        def fail(journal):
            called.append(1)
            journal['attempts'] = [{'status': 'failed', 'q': np.array([np.nan])}]
            raise PlanningFailure('synthetic_numeric_failure', used=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = d.run_once(root, fail, {'synthetic': True})
            self.assertFalse(result['numerical_pass'])
            self.assertEqual(result['first_failed_gate'], 'synthetic_numeric_failure')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(FileExistsError):
                d.run_once(root, fail, {'synthetic': True})
            self.assertEqual(len(called), 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            self.assertIn('NaN', (root/'plan.json').read_text())

    def test_exclusive_success_is_numeric_only_not_physics_authorization(self):
        d = diagnostic()
        with tempfile.TemporaryDirectory() as temp:
            result = d.run_once(Path(temp), lambda journal: journal.update(success=True), {})
            self.assertTrue(result['numerical_pass'])
            self.assertFalse(result['physics_authorized'])
            self.assertEqual(result['actual_expert_invocations'], 0)
            self.assertEqual(result['physics_substeps'], 0)