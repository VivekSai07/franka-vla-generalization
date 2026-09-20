"""Real saved MuJoCo states; mj_step/renderer/expert execution forbidden."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from data_collection.benchmark import capture_state, restore_state, synchronize_observation
from experiments.local_position_probe import contract as c, scenes as s


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
            self.guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No physics or renderer')))
        self.model, self.manifest, self.source = s.load_source()

    def test_actual_17_variants_exactly_one_qpos_scalar_and_no_other_integration_change(self):
        self.assertEqual(self.model.nq, 37)
        self.assertEqual(self.source['initial_state'].size, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        hashes = set()
        for condition in c.make_plan()['conditions']:
            with self.subTest(condition=condition['condition_id']):
                derived = s.derive_scene(self.model, self.source, condition)
                a = derived['arrays']; hashes.add(derived['scene_hash'])
                expected = self.source['qpos'].copy()
                if condition['object']:
                    # Independently inspected source freejoint positions: red9, green16.
                    index = {'red': 9, 'green': 16}[condition['object']] + condition['axis']
                    expected[index] += condition['delta_m']
                self.assertEqual(a['qpos'].tobytes(), expected.tobytes())
                changed = np.flatnonzero(a['initial_state'].view('u8') != self.source['initial_state'].view('u8'))
                self.assertEqual(changed.tolist(), [] if not condition['object'] else [index+1])
                for key in ('initial_ctrl', 'camera_xpos', 'camera_xmat'):
                    self.assertEqual(a[key].tobytes(), self.source[key].tobytes())
                d = mujoco.MjData(self.model); restore_state(self.model, d, a['initial_state'])
                before = capture_state(self.model, d).tobytes(); synchronize_observation(self.model, d)
                self.assertEqual(capture_state(self.model, d).tobytes(), before)
        self.assertEqual(len(hashes), 17)

    def test_control_all_saved_arrays_bit_exact(self):
        a = s.derive_scene(self.model, self.source, c.make_plan()['conditions'][0])['arrays']
        self.assertEqual(set(a), set(self.source))
        for key in a: self.assertEqual(a[key].tobytes(), self.source[key].tobytes())

    def test_wrong_axis_body_quaternion_warmstart_ctrl_time_rejected(self):
        condition = c.make_plan()['conditions'][1]
        valid = s.derive_scene(self.model, self.source, condition)['arrays']
        # Mutate real MjData then recapture all arrays so stale redundant arrays cannot explain rejection.
        for field, index in (('qpos', 10), ('qpos', 16), ('qpos', 12), ('qacc_warmstart', 0), ('ctrl', 0), ('qvel', 0), ('time', None)):
            d = mujoco.MjData(self.model); restore_state(self.model, d, valid['initial_state'])
            if index is None: d.time += .001
            else: getattr(d, field)[index] += .001
            synchronize_observation(self.model, d)
            with self.subTest(field=field, index=index), self.assertRaises(ValueError):
                s.validate_scene(self.model, self.source, condition, s.arrays_from_data(self.model, d))

    def test_nonfinite_and_inconsistent_saved_geometry_rejected(self):
        condition = c.make_plan()['conditions'][0]
        for key in self.source:
            bad = {k: v.copy() for k, v in self.source.items()}; bad[key].flat[0] = np.nan
            with self.subTest(key=key), self.assertRaises(ValueError): s.validate_scene(self.model, self.source, condition, bad)
        bad = {k: v.copy() for k, v in self.source.items()}; bad['camera_xmat'].flat[0] += .001
        with self.assertRaises(ValueError): s.validate_scene(self.model, self.source, condition, bad)

    def test_workspace_clearance_height_and_robot_contact_rejected(self):
        d = mujoco.MjData(self.model)
        for x, y, z in ((.19, 0., .021), (.3, -.41, .021), (.3, 0., .08),
                        tuple(self.source['body_xpos'][self.model.body('target_cube_green').id])):
            restore_state(self.model, d, self.source['initial_state']); d.qpos[9:12] = [x, y, z]
            synchronize_observation(self.model, d)
            with self.subTest(position=(x, y, z)), self.assertRaises(ValueError): s.validate_geometry(self.model, d, self.source)
        # Place a real cube at the real finger body; forward only, no contact evolution.
        restore_state(self.model, d, self.source['initial_state'])
        d.qpos[9:12] = d.xpos[self.model.body('left_finger').id]
        synchronize_observation(self.model, d)
        with self.assertRaisesRegex(ValueError, 'robot.*cube|cube.*robot'): s.require_no_robot_cube_contacts(self.model, d)

    def test_nonfree_cube_joint_rejected(self):
        joint = self.model.body_jntadr[self.model.body('target_cube_red').id]
        self.model.jnt_type[joint] = mujoco.mjtJoint.mjJNT_BALL
        with self.assertRaises(ValueError): s.derive_scene(self.model, self.source, c.make_plan()['conditions'][1])


if __name__ == '__main__': unittest.main()