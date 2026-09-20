"""The probe must use real observations, never actions or simulator ground truth."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class VerificationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('publication.verify_baselines'),
                             'Independent verification implementation is missing')
        from publication import verify_baselines
        return verify_baselines

    def test_observation_reads_only_measured_state_views_and_task(self):
        verify = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tick.npz'
            image = np.full((512, 512, 3), 128, dtype=np.uint8)
            np.savez(path, **{'observation.state': np.arange(9, dtype=np.float32),
                              'observation.images.image': image,
                              'observation.images.image2': image,
                              'task': 'recorded task', 'action': np.full(8, 999., dtype=np.float32),
                              'trace/qpos': np.full(37, -999.)})
            obs = verify.read_observation(path)
            self.assertEqual(set(obs), {'observation.state', 'observation.images.image',
                                       'observation.images.image2', 'task'})
            self.assertTrue(torch.equal(obs['observation.state'], torch.arange(9, dtype=torch.float32)))
            self.assertEqual(obs['observation.images.image'].shape, (3, 512, 512))
            self.assertEqual(obs['observation.images.image'][0, 0, 0].item(),
                             torch.tensor(128 / 255, dtype=torch.float32).item())
            self.assertEqual(obs['task'], 'recorded task')

    def test_prediction_rejects_targets_and_oracle_inputs(self):
        from publication.load_baseline import predict
        with self.assertRaises(ValueError):
            predict(None, None, None, {'action': torch.zeros(8)})

    def test_empty_processor_action_is_removed_but_real_target_rejected(self):
        from publication import load_baseline
        strip = getattr(load_baseline, 'without_targets', None)
        self.assertIsNotNone(strip, 'Native converter empty action entries need explicit handling')
        batch = {'observation.state': torch.zeros(1, 9), 'action': None, 'action_is_pad': None}
        self.assertEqual(set(strip(batch)), {'observation.state'})
        self.assertIn('action', batch)
        with self.assertRaises(ValueError):
            strip(dict(batch, action=torch.zeros(1, 8)))

    def test_release_inventory_rejects_added_pickle_and_changed_metadata(self):
        verify = self.module()
        from publication.package_baselines import inventory, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            write_json(root / 'inventory.json', inventory(root))
            verify.check_inventory(root)
            (root / 'optimizer.pt').write_bytes(b'not allowed')
            with self.assertRaises(ValueError):
                verify.check_inventory(root)
            (root / 'optimizer.pt').unlink()
            (root / 'config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify.check_inventory(root)


if __name__ == '__main__':
    unittest.main()