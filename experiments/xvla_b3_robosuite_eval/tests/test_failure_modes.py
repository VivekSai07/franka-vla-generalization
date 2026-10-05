import json
import unittest
from pathlib import Path

import numpy as np

from experiments.xvla_b3_robosuite_eval import failure_modes as fm
from experiments.xvla_b3_robosuite_eval import protocol

Z0 = 0.021
COLOURS = ['blue', 'green', 'red']          # pick blue, place green, distractor red
TICKS = 600


def trace(lift=None, gripper=None, xy=0.6, z=0.0, blue_xy=(0.0, 0.0), green_xy=(0.6, 0.0), red_xy=(-0.3, 0.3)):
    """cube_xyz for blue/green/red; `lift` is the blue cube's per-tick z offset."""
    cube = np.zeros((TICKS, 3, 3))
    cube[:, :, 2] = Z0
    cube[:, 0, :2], cube[:, 1, :2], cube[:, 2, :2] = blue_xy, green_xy, red_xy
    if lift is not None:
        cube[:, 0, 2] += lift
    action = np.zeros((TICKS, 8))
    action[:, 7] = 255 if gripper is None else gripper
    broadcast = lambda v: np.full(TICKS, v) if np.isscalar(v) else np.asarray(v, dtype=float)
    return dict(cube_xyz=cube, colours=np.asarray(COLOURS), applied_action=action, xy_m=broadcast(xy), z_m=broadcast(z))


def span(start, stop, value, base=0.0):
    out = np.full(TICKS, base, dtype=float)
    out[start:stop] = value
    return out


class ModeTests(unittest.TestCase):
    def mode(self, **kwargs):
        return fm.failure_mode(trace(**kwargs), 'blue', 'green')

    def test_never_started_has_a_gripper_that_never_closes(self):
        self.assertEqual(self.mode()[0], 'never_started')

    def test_grasp_failed_closes_without_lifting(self):
        self.assertEqual(self.mode(gripper=span(100, 140, 0, 255))[0], 'grasp_failed')

    def test_cube_lost_ends_below_the_table(self):
        self.assertEqual(self.mode(lift=span(100, 400, 0.15) - np.linspace(0, 0.2, TICKS) * (np.arange(TICKS) > 400))[0],
                         'cube_lost')

    def test_wrong_cube_lifted(self):
        t = trace()
        t['cube_xyz'][100:300, 2, 2] += 0.1
        self.assertEqual(fm.failure_mode(t, 'blue', 'green')[0], 'wrong_cube_lifted')

    def test_stall_above_the_target_holds_closed_and_high(self):
        lift = span(100, TICKS, 0.15)
        mode, info = self.mode(lift=lift, gripper=span(100, TICKS, 0, 255), xy=0.02, z=0.13)
        self.assertEqual(mode, 'stall_above_target')
        self.assertGreaterEqual(info['hover_ticks'], 300)

    def test_a_late_release_after_a_long_hold_is_still_a_stall(self):
        xy = span(0, TICKS, 0.02)
        xy[450:] = 0.13
        mode, _ = self.mode(lift=span(100, 450, 0.2), gripper=span(100, 450, 0, 255), xy=xy, z=0.2)
        self.assertEqual(mode, 'stall_above_target')

    def test_near_miss_records_the_signed_offset(self):
        lift = span(100, 300, 0.2)
        mode, info = self.mode(lift=lift, gripper=span(100, 300, 0, 255), xy=0.02, z=0.0,
                               blue_xy=(0.622, 0.0))
        self.assertEqual(mode, 'near_miss_toppled')
        self.assertEqual(info['release_tick'], 300)
        self.assertAlmostEqual(info['release_offset_xy_m'][0], 0.022, places=6)

    def test_release_far_from_the_target(self):
        mode, info = self.mode(lift=span(100, 300, 0.25), gripper=span(100, 300, 0, 255), xy=0.2, z=0.25)
        self.assertEqual(mode, 'released_away_from_target')
        self.assertEqual(info['release_tick'], 300)

    def test_cube_resting_on_a_distractor(self):
        t = trace(lift=span(100, 300, 0.2), gripper=span(100, 300, 0, 255))
        t['cube_xyz'][300:, 0, :2] = t['cube_xyz'][300:, 2, :2] + 0.01
        t['cube_xyz'][300:, 0, 2] = Z0 + 0.04
        mode, info = fm.failure_mode(t, 'blue', 'green')
        self.assertEqual((mode, info['resting_on']), ('stacked_on_wrong_cube', 'red'))

    def test_the_place_cube_is_not_a_wrong_cube(self):
        t = trace(lift=span(100, 300, 0.2), gripper=span(100, 300, 0, 255), xy=0.02, z=0.04)
        t['cube_xyz'][300:, 0, :2] = t['cube_xyz'][300:, 1, :2]
        t['cube_xyz'][300:, 0, 2] = Z0 + 0.04
        self.assertNotEqual(fm.failure_mode(t, 'blue', 'green')[0], 'stacked_on_wrong_cube')

    def test_carry_aborted_stays_near_the_start(self):
        lift = span(100, 300, 0.07)
        mode, _ = self.mode(lift=lift, gripper=span(100, 300, 0, 255), xy=0.6, blue_xy=(0.05, 0.0))
        self.assertEqual(mode, 'carry_aborted_near_pick')

    def test_every_documented_mode_is_reachable(self):
        self.assertEqual(len(set(fm.MODES)), len(fm.MODES))
        self.assertIn('other', fm.MODES)


class ScreenValidation(unittest.TestCase):
    """The classifier reproduces the 29 reviewed screen labels; skipped where the screens are not mounted."""

    def test_reproduces_every_reviewed_label(self):
        labels = json.loads((Path(fm.__file__).parent/'screen_labels.json').read_text())
        found = {}
        for checkpoint in protocol.CANDIDATES:
            directory = protocol.output_dir('screen', checkpoint)
            if not (directory/'result.json').is_file():
                self.skipTest('screen outputs not mounted')
            for row in fm.review(directory).values():
                found[f'{checkpoint:06d}:{row["ordinal"]}'] = row['mode']
        self.assertEqual(found, labels)


if __name__ == '__main__':
    unittest.main()
