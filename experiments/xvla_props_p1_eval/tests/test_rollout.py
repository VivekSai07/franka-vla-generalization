import random
import unittest

import numpy as np
import torch

from experiments.props_scene import constants as c
from experiments.xvla_props_p1_eval import rollout


class SeedTrialTests(unittest.TestCase):
    def test_seed_trial_is_reproducible_across_python_numpy_and_torch(self):
        rollout.seed_trial(123)
        a = (random.random(), np.random.rand(), torch.rand(1).item())
        rollout.seed_trial(123)
        b = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(a, b)


class ConstantsTests(unittest.TestCase):
    def test_max_ticks_matches_the_scene_constant_and_the_plan(self):
        self.assertEqual(rollout.MAX_TICKS, c.MAX_TICKS)
        self.assertEqual(rollout.MAX_TICKS, 900)

    def test_image_keys_match_the_shared_camera_map(self):
        self.assertEqual(rollout.IMAGE_KEYS, ('observation.images.image', 'observation.images.image2'))
        self.assertEqual(rollout.CAMERA_NAMES, ('front_cam', 'gripper_cam'))

    def test_trace_keys_cover_every_field_rollout_writes(self):
        self.assertEqual(set(rollout.TRACE_KEYS), {'state', 'raw_action', 'applied_action', 'prop_xyz',
                                                    'xy_margin_m', 'z_m', 'lowest_z_m', 'gripper_open',
                                                    'finger_contact'})


if __name__ == '__main__':
    unittest.main()
