"""The colour-pair overrides must be indistinguishable from the shared originals on red-on-green.

`fixed_layout.validate` is sealed inside `broad_workspace_data`'s provenance, so B2 rebinds its
cube-dependent globals instead of editing it (FAILURES #28/#29). These tests pin the rebound
versions to the originals wherever the originals are defined, so the only behavioural difference
is the one B2 needs: green-on-red episodes measure the green cube.
"""
import unittest

import mujoco
import numpy as np

from experiments.fixed_layout import validate as generic
from experiments.fixed_layout.source import Expert, load_reset
from experiments.b2_distractor_data import validate as v
from experiments.b2_distractor_data import plan as p


N = p.EXPECTED_FRAMES


def _passing_post(n=N):
    """Geometry that clears every gate: grasp, lift clear of the table, then a stacked release."""
    pick = np.zeros((n, 3)); pick[:, 2] = .02
    place = np.zeros((n, 3)); place[:, 0] = .30; place[:, 2] = .02
    place[100:, 0] = 0.
    pick[40:100, 2] = .12
    pick[100:, 2] = place[100:, 2] + .04
    hand = pick + .01
    left = np.zeros(n, bool); left[40:120] = True
    return {'pick_xyz': pick, 'place_xyz': place, 'hand_xyz': hand,
            'left_contact': left.copy(), 'right_contact': left.copy()}


def _actions(n=N):
    a = np.zeros((n, 8), np.float32)
    a[40:400, 7] = 100.
    a[400:, 7] = 200.
    return a


class GatesEquivalenceTests(unittest.TestCase):
    def test_gates_match_the_original_on_red_on_green(self):
        post, actions, z = _passing_post(), _actions(), .02
        mine = v.gates_for_pair(actions, post, z)
        theirs = generic.contact_gates(actions, post, z, 'red', 'green')
        self.assertEqual(mine, theirs)

    def test_gates_reject_whatever_the_original_rejects(self):
        actions, z = _actions(), .02
        cases = {
            'never closed': {'left_contact': np.zeros(N, bool), 'right_contact': np.zeros(N, bool)},
            'no lift': {'pick_xyz': np.full((N, 3), .02)},
        }
        for name, override in cases.items():
            with self.subTest(case=name):
                post = {**_passing_post(), **override}
                with self.assertRaises(ValueError):
                    generic.contact_gates(actions, post, z, 'red', 'green')
                with self.assertRaises(ValueError):
                    v.gates_for_pair(actions, post, z)

    def test_gates_reject_malformed_geometry_like_the_original(self):
        post = {**_passing_post(), 'hand_xyz': np.full((N, 3), np.nan)}
        with self.assertRaises(ValueError):
            v.gates_for_pair(_actions(), post, .02)

    def test_only_the_colour_guard_is_dropped(self):
        """The original refuses green-on-red outright; the copy is what makes B2 possible."""
        post, actions, z = _passing_post(), _actions(), .02
        with self.assertRaises(ValueError):
            generic.contact_gates(actions, post, z, 'green', 'red')
        self.assertEqual(v.gates_for_pair(actions, post, z), generic.contact_gates(actions, post, z, 'red', 'green'))


class ObservationEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert = Expert()
        cls.source = load_reset(cls.expert)

    def _data(self):
        data = mujoco.MjData(self.expert.model)
        data.qpos[:] = self.source['qpos']
        generic.synchronize_observation(self.expert.model, data)
        return data

    def _same(self, a, b):
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)

    def test_observation_matches_the_original_on_red_on_green(self):
        data = self._data()
        self._same(v.observation_for_pair(self.expert.model, data, 'red', 'green'),
                   generic._observation(self.expert.model, data))

    def test_swapping_the_pair_swaps_exactly_the_cube_bindings(self):
        data = self._data()
        forward = v.observation_for_pair(self.expert.model, data, 'red', 'green')
        reverse = v.observation_for_pair(self.expert.model, data, 'green', 'red')
        np.testing.assert_array_equal(reverse['pick_xyz'], forward['place_xyz'])
        np.testing.assert_array_equal(reverse['place_xyz'], forward['pick_xyz'])
        for key in ('qpos', 'body_xpos', 'camera_xpos', 'camera_xmat', 'hand_xyz'):
            np.testing.assert_array_equal(np.asarray(reverse[key]), np.asarray(forward[key]))

    def test_the_two_cubes_are_actually_distinguishable(self):
        """Guards against the pair swap being a silent no-op."""
        data = self._data()
        forward = v.observation_for_pair(self.expert.model, data, 'red', 'green')
        self.assertFalse(np.array_equal(forward['pick_xyz'], forward['place_xyz']))


class TaskAwareRestoreTests(unittest.TestCase):
    NAMES = ('read_rows', 'TASK', '_observation', 'contact_gates', 'stack_metrics')

    @classmethod
    def setUpClass(cls):
        cls.expert = Expert()
        cls.reset = {'body_xpos': np.zeros((cls.expert.model.nbody, 3))}

    def _enter(self):
        return v.task_aware(p.TASK_RED_ON_GREEN, 0, 'red', 'green', self.expert, self.reset)

    def test_every_patched_global_is_restored(self):
        before = {n: getattr(generic, n) for n in self.NAMES}
        with self._enter():
            self.assertNotEqual(getattr(generic, '_observation'), before['_observation'])
        for name in self.NAMES:
            self.assertIs(getattr(generic, name), before[name], name)

    def test_globals_are_restored_after_an_exception(self):
        before = {n: getattr(generic, n) for n in self.NAMES}
        with self.assertRaises(RuntimeError):
            with self._enter():
                raise RuntimeError('boom')
        for name in self.NAMES:
            self.assertIs(getattr(generic, name), before[name], name)

    def test_patched_stack_metrics_ignores_the_hardcoded_literals(self):
        seen = []
        original = generic.stack_metrics
        generic.stack_metrics = lambda model, data, pick, place: seen.append((pick, place))
        try:
            with v.task_aware(p.TASK_GREEN_ON_RED, 1, 'green', 'red', self.expert, self.reset):
                generic.stack_metrics(None, None, 'red', 'green')
        finally:
            generic.stack_metrics = original
        self.assertEqual(seen, [('green', 'red')])


if __name__ == '__main__':
    unittest.main()
