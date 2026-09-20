"""Saved-observation/contract integration only; never invoke an actual expert."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import pinocchio as pin

from test_bounded_placement_v3 import candidate

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / 'outputs/paired_demonstrations/paired_20260917'
V2 = ROOT / 'outputs/diagnostics/pose_snapshot_v2_20260917'


class PrefixTests(unittest.TestCase):
    def test_all_197_saved_v2_prefix_commands_phases_targets_rng(self):
        p = candidate('prefix')
        from experiments.paired_demonstrations.expert_v1 import Expert
        from data_collection.benchmark import restore_state, synchronize_observation
        expert = Expert(); data = mujoco.MjData(expert.model)
        episode = json.loads((PAIRED/'attempts.jsonl').read_text().splitlines()[4])
        with np.load(V2/'corrected_trace.npz', allow_pickle=False) as trace:
            restore_state(expert.model, data, trace['initial_state'])
            prefix = p.Prefix(expert, data, episode)
            for tick in range(197):
                data.qpos[:] = trace['qpos'][tick]
                data.time = trace['sim_time'][tick]
                synchronize_observation(expert.model, data)
                row = prefix.command()
                for key in ('ctrl', 'requested_joint_velocity', 'applied_joint_velocity',
                            'command_saturation', 'target_pos', 'phase_index', 'phase_time'):
                    np.testing.assert_array_equal(row[key], trace[key][tick], err_msg=f'{key} tick {tick}')
            self.assertEqual(prefix.phase, 4)
            self.assertEqual(prefix.frames, 197)
            with self.assertRaises(candidate('solver').PlanningFailure):
                prefix.command()
            base = data.xpos[expert.model.body('target_cube_blue').id].copy()
            endpoints = p.placement_endpoints(prefix, data)
            rng = np.random.RandomState(488840035)
            rng.normal(0, .005, 3); rng.normal(0, .005, 3)
            expected = [base+[0, 0, .305]+rng.normal(0, .005, 3),
                        base+[0, 0, .155]+rng.normal(0, .005, 3), base+[0, 0, .155]]
            for endpoint, want in zip(endpoints, expected):
                np.testing.assert_array_equal(endpoint.translation, want)
            self.assertEqual(prefix.rng.get_state()[2], rng.get_state()[2])
            np.testing.assert_array_equal(prefix.rng.get_state()[1], rng.get_state()[1])


class ExecutionTests(unittest.TestCase):
    def measurement(self, **overrides):
        return dict(grasped=True, xy=.001, z=.06, hand_error=.001, rotation_error=.001,
                    max_speed=.01, unexpected=False, **overrides)

    def test_settling_failure_never_opens_and_stops_at_60(self):
        g = candidate('gates'); controller = g.PlacementExecution([(2, np.zeros(7))], 197)
        row = self.measurement(); row['xy'] = .02
        self.assertEqual(controller.command(np.zeros(7), row)[1], 0)
        for _ in range(60):
            _, gripper, _ = controller.command(np.zeros(7), row)
            self.assertEqual(gripper, 0)
        with self.assertRaises(candidate('solver').PlanningFailure) as error:
            controller.command(np.zeros(7), row)
        self.assertEqual(error.exception.gate, 'placement_not_ready')
        self.assertFalse(controller.guard.opened)

    def test_only_tenth_ready_arrival_sample_opens_and_tracking_holds(self):
        g = candidate('gates'); target = np.full(7, .01)
        controller = g.PlacementExecution([(0, np.zeros(7)), (2, target)], 197)
        controller.command(np.zeros(7), self.measurement())
        held, grip, stage = controller.command(np.full(7, .06), self.measurement())
        np.testing.assert_array_equal(held, np.zeros(7)); self.assertEqual(grip, 0)
        controller.command(np.zeros(7), self.measurement())
        for _ in range(9):
            self.assertEqual(controller.command(target, self.measurement())[1], 0)
        self.assertEqual(controller.command(target, self.measurement())[1], 255)

    def test_diagnostic_order_and_review_block_before_any_trial(self):
        d = candidate('diagnostic')
        self.assertEqual([e['episode_index'] for e in d.saved_cases()], [4, 0, 1, 2, 3])
        self.assertEqual(d.saved_cases()[0]['seed'], 488840035)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(candidate('solver').PlanningFailure) as error:
                d.require_review(Path(directory), {'file': 'hash'})
            self.assertEqual(error.exception.gate, 'independent_review_required')

    def test_failure_artifact_preserves_empty_plan_and_no_invocation(self):
        d = candidate('diagnostic'); failure = candidate('solver').PlanningFailure('release_height_contract_conflict', z=.12)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            d.save_failure(out, failure, {'waypoints': []}, 0)
            saved = json.loads((out/'failure.json').read_text())
            self.assertEqual(saved['first_failed_gate'], 'release_height_contract_conflict')
            self.assertEqual(saved['actual_expert_invocations'], 0)
            self.assertEqual(saved['plan'], {'waypoints': []})
            with self.assertRaises(FileExistsError):
                d.save_failure(out, failure, {}, 0)

    def test_expert_configuration_is_explicit_version_only(self):
        e = candidate('expert')
        from experiments.pose_snapshot_v2.expert import Expert as V2Expert
        old, new = V2Expert(), e.Expert()
        self.assertEqual(new.config, dict(old.config, expert_version=e.EXPERT_VERSION))
        np.testing.assert_array_equal(new.bounds, old.bounds)

    def test_lossless_diagnostic_video_and_durable_raw_tick(self):
        a = candidate('artifacts')
        from experiments.paired_demonstrations.media import CAMERAS, pixels_digest
        import av
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            sink = a.TraceSink(out, 'test instruction')
            image = np.zeros((512, 512, 3), dtype=np.uint8); image[:, 100:300, 0] = 200
            frame = {key: image for key in CAMERAS}
            frame.update(action=np.zeros(8, np.float32), **{'observation.state': np.zeros(9, np.float32)}, task='test instruction')
            sink(frame, dict(sim_time=.2, substeps=20))
            sink.finish()
            with np.load(out/'ticks/tick_000000.npz', allow_pickle=False) as row:
                np.testing.assert_array_equal(row['action'], np.zeros(8))
                self.assertEqual(row['sim_time'], .2)
            for camera in CAMERAS.values():
                with av.open(str(out/f'{camera}.mkv')) as video:
                    decoded = list(video.decode(video=0))
                    self.assertEqual(len(decoded), 1)
                    self.assertEqual(pixels_digest(decoded[0].to_ndarray(format='rgb24')), pixels_digest(image))

    def test_saved_control_reference_replay_detects_state_control_and_clock_corruption(self):
        r = candidate('replay')
        from experiments.paired_demonstrations.expert_v1 import Expert
        from data_collection.benchmark import capture_state, restore_state, step_control
        expert = Expert(); data = mujoco.MjData(expert.model)
        episode = json.loads((PAIRED/'attempts.jsonl').read_text().splitlines()[4])
        rows = []
        with np.load(V2/'corrected_trace.npz', allow_pickle=False) as trace:
            initial = trace['initial_state'].copy()
            restore_state(expert.model, data, initial)
            for tick in range(3):
                row = {key: trace[key][tick].copy() for key in (
                    'qpos', 'ctrl', 'sim_time', 'image_time', 'body_xpos', 'camera_xpos', 'camera_xmat',
                    'action', 'observation_state', 'phase_index', 'phase_time', 'substeps')}
                row.update(qvel=data.qvel.copy(), integration_state=capture_state(expert.model, data))
                data.ctrl[:] = row['ctrl']
                step_control(expert.model, data, 1/30, 7)
                row.update(post_state=capture_state(expert.model, data), released_stack=False)
                rows.append(row)
        result = dict(frames=3, final_qpos=data.qpos.copy(), final_state=capture_state(expert.model, data),
                      final_ctrl=data.ctrl.copy(), success=False, stable_ticks=0, actual_substeps=60)
        self.assertEqual(r.verify_reference(expert, initial, episode, rows, result, {})['qpos_max_error'], 0.)
        for key in ('qpos', 'ctrl', 'sim_time'):
            corrupted = [{k: np.copy(v) for k, v in row.items()} for row in rows]
            corrupted[1][key] += .01
            with self.assertRaises((ValueError, AssertionError)):
                r.verify_reference(expert, initial, episode, corrupted, result, {})

    def test_fixed_trial_loop_stops_on_first_task_or_integrity_failure(self):
        d = candidate('diagnostic')
        self.assertTrue(hasattr(d, 'run_fixed_cases'), 'Missing stop-first-failure orchestration')
        for failure_field in ('success', 'integrity_passed'):
            invoked = []
            def invoke(entry):
                invoked.append(entry['episode_index'])
                return dict(success=len(invoked) != 2 if failure_field == 'success' else True,
                            integrity_passed=len(invoked) != 2 if failure_field == 'integrity_passed' else True)
            results = d.run_fixed_cases(invoke)
            self.assertEqual(invoked, [4, 0])
            self.assertEqual(len(results), 2)

    def test_no_open_when_remaining_episode_cannot_fit_release_and_retreat_minima(self):
        g = candidate('gates')
        controller = g.PlacementExecution([(2, np.zeros(7))], 1189)
        controller.command(np.zeros(7), self.measurement())
        for _ in range(9):
            controller.command(np.zeros(7), self.measurement())
        with self.assertRaises(candidate('solver').PlanningFailure) as error:
            controller.command(np.zeros(7), self.measurement())
        self.assertEqual(error.exception.gate, 'post_release_deadline')
        self.assertFalse(controller.guard.opened)


if __name__ == '__main__':
    unittest.main()