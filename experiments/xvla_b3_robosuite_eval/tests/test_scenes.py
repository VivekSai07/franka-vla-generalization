import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.xvla_b3_robosuite_eval import protocol, scenes


def keys():
    screen = protocol.screen_trials()[0]
    report = protocol.report_trials()
    control = next(t for t in report if t['condition'] == 'control' and t['kind'] == 'trained')
    held_out = next(t for t in report if t['condition'] == 'distractor' and t['kind'] == 'held_out')
    return screen, control, held_out


class RestoreTests(unittest.TestCase):
    def setUp(self):
        if not protocol.BANK_NPZ.exists():
            self.skipTest('B3 parent not mounted')

    def test_three_keys_restore_exactly(self):
        bank = np.load(protocol.BANK_NPZ)
        manifest = json.loads(protocol.BANK_JSON.read_text())
        for trial in keys():
            key = trial['bank_key']
            model, data = scenes.restore(key)
            np.testing.assert_array_equal(data.qpos, bank[f'{key}:qpos'])
            self.assertEqual(scenes.present_colours(model), trial['colours'])
            self.assertEqual(scenes.present_colours(model), manifest[key]['colours'])
            for colour, xy in manifest[key]['cube_xy_m'].items():
                position = data.xpos[model.body(f'target_cube_{colour}').id]
                self.assertLess(float(np.hypot(*(position[:2] - np.asarray(xy)))), 1e-3, (key, colour))

    def test_every_banked_key_matches_the_protocol(self):
        manifest = json.loads(protocol.BANK_JSON.read_text())
        wanted = {t['bank_key'] for t in protocol.screen_trials() + protocol.report_trials()}
        self.assertEqual(wanted, set(manifest))
        self.assertEqual(len(manifest), 100)

    def test_restore_is_repeatable_and_independent(self):
        key = keys()[0]['bank_key']
        _, first = scenes.restore(key)
        first.qpos[:] = 0
        _, second = scenes.restore(key)
        self.assertFalse(np.array_equal(first.qpos, second.qpos))
        _, third = scenes.restore(key)
        np.testing.assert_array_equal(second.qpos, third.qpos)

    def test_a_changed_bank_array_is_rejected_by_its_manifest_hash(self):
        key = keys()[0]['bank_key']
        with tempfile.TemporaryDirectory() as tmp:
            arrays = dict(np.load(protocol.BANK_NPZ))
            arrays[f'{key}:qpos'] = arrays[f'{key}:qpos'].copy()
            arrays[f'{key}:qpos'][0] += 1e-9
            copy = Path(tmp)/'bank.npz'
            np.savez(copy, **arrays)
            with patch.object(protocol, 'BANK_NPZ', copy):
                with self.assertRaisesRegex(ValueError, 'bank'):
                    scenes.restore(key)
            # an untouched key in the same tampered copy still restores
            other = keys()[1]['bank_key']
            with patch.object(protocol, 'BANK_NPZ', copy):
                scenes.restore(other)

    def test_a_changed_manifest_hash_is_rejected(self):
        key = keys()[0]['bank_key']
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(protocol.BANK_JSON.read_text())
            manifest[key]['scene_sha256'] = '0'*64
            copy = Path(tmp)/'bank.json'
            copy.write_text(json.dumps(manifest))
            with patch.object(protocol, 'BANK_JSON', copy):
                with self.assertRaises(ValueError):
                    scenes.restore(key)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            scenes.restore('evaluation_999:control')


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        if not protocol.BANK_NPZ.exists():
            self.skipTest('B3 parent not mounted')

    def test_the_expert_solves_one_screen_scene(self):
        from experiments.xvla_b3_robosuite_eval import feasibility
        result = feasibility.run_one(keys()[0]['bank_key'])
        print('FEASIBILITY_PROBE', result)
        self.assertTrue(result['success'], result)
        self.assertLess(result['final_xy_m'], 0.02)
        self.assertTrue(0.035 < result['final_z_m'] < 0.045)
        self.assertEqual(result['frames'], 528)

    def test_summary_names_failed_layouts_once(self):
        from experiments.xvla_b3_robosuite_eval import feasibility
        ok = dict(success=True, reason='success', final_xy_m=0.001, final_z_m=0.04, frames=528, seconds=1)
        bad = dict(ok, success=False, reason='grasp')
        summary = feasibility.summarize({'evaluation_001:control': ok, 'evaluation_001:distractor': bad,
                                         'evaluation_002:distractor': bad, 'evaluation_003:control': ok})
        self.assertEqual(summary['failed_keys'], ['evaluation_001:distractor', 'evaluation_002:distractor'])
        self.assertEqual(summary['failed_layouts'], ['evaluation_001', 'evaluation_002'])
        self.assertEqual((summary['keys'], summary['succeeded']), (4, 2))


if __name__ == '__main__':
    unittest.main()
