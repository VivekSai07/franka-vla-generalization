"""Actual full-state initialization with native stepping/rendering forbidden."""
from contextlib import ExitStack
import copy
import importlib
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.fixed_layout.source import Expert, load_reset


class SceneTests(unittest.TestCase):
    def setUp(self):
        try:
            self.s = importlib.import_module('experiments.local_varied_data.scenes')
            self.p = importlib.import_module('experiments.local_varied_data.plan')
        except ModuleNotFoundError as exc:
            self.fail(f'Missing local-varied scenes: {exc}')
        guards = ExitStack(); self.addCleanup(guards.close)
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
            guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('Forbidden native execution')))
        guards.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('Forbidden expert execution')))
        self.expert = Expert()
        self.source = load_reset(self.expert)

    def test_all_77_real_layouts_only_four_xy_state_values_change(self):
        plan = self.p.make_plan()
        bank = self.s.make_bank(self.expert.model, self.source, plan)
        self.assertEqual(len(bank), 77)
        self.assertEqual(len({self.s.scene_hash(a) for a in bank.values()}), 77)
        for layout in plan['layouts']:
            a = bank[layout['layout_id']]
            expected = self.source['qpos'].copy()
            expected[[9, 10, 16, 17]] += layout['offsets_m']
            self.assertEqual(a['qpos'].tobytes(), expected.tobytes())
            changed = np.flatnonzero(a['initial_state'].view('u8') != self.source['initial_state'].view('u8')).tolist()
            self.assertEqual(changed, [] if layout['layout_id'] == 'center' else [10, 11, 17, 18])
            for key in ('initial_ctrl', 'camera_xpos', 'camera_xmat'):
                self.assertEqual(a[key].tobytes(), self.source[key].tobytes())
        for e in plan['episodes']: self.assertIn(e['layout_id'], bank)
        self.assertFalse(set(e['layout_id'] for e in plan['episodes']) & set(plan['evaluation_layout_ids']))

    def test_wrong_non_xy_state_and_derived_geometry_rejected(self):
        layout = self.p.make_plan()['layouts'][1]
        a = self.s.derive_scene(self.expert.model, self.source, layout)
        for key, index in (('initial_state', 0), ('initial_state', 50), ('qpos', 11), ('camera_xmat', 0)):
            bad = copy.deepcopy(a); bad[key].flat[index] += .001
            with self.subTest(key=key, index=index), self.assertRaises(ValueError):
                self.s.validate_scene(self.expert.model, self.source, layout, bad)

    def test_old_17_probe_overlap_and_duplicates_rejected_without_redraw(self):
        plan = self.p.make_plan()
        bad = copy.deepcopy(plan['layouts'])
        bad[1]['offsets_m'] = [.005, 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'probe'): self.s.validate_layouts(bad)
        bad = copy.deepcopy(plan['layouts']); bad[2]['offsets_m'] = bad[1]['offsets_m'].copy()
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.s.validate_layouts(bad)


if __name__ == '__main__': unittest.main()