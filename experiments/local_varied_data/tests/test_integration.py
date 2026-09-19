"""Two real LeRobot/AV1 episodes, numeric simulation only; distinct reset injection.

The simulator double replaces native operations, not persistence, contacts,
commands, media, replay evidence or per-episode gates. No expert.run is invoked.
"""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.fixed_layout import validate as generic
from experiments.fixed_layout.tests.test_validate import SyntheticSimulator
from experiments.fixed_layout.record import write_json, file_hash, TickSink
from experiments.fixed_layout.source import Expert
from experiments.paired_demonstrations.media import CAMERAS
from experiments.local_varied_data import plan, record, validate

EVIDENCE = Path('/vla/outputs/implementation/local_varied_data_20260919')


class IntegrationTests(unittest.TestCase):
    def test_two_real_encodes_continue_after_read_and_distinct_reset_replay(self):
        root = Path(tempfile.mkdtemp(prefix='two-episode-', dir=EVIDENCE))
        (root / 'attempts').mkdir()
        dataset = record.create_dataset(root, lambda: None)
        slots = plan.make_plan()['episodes']
        sims = [SyntheticSimulator(.001), SyntheticSimulator(.0011)]
        # Four XY changes in this numeric-only double's simplified qpos mapping.
        # Episode indices are contiguous for the real LeRobot writer. Distinct
        # geometry is a validator fixture, not a claim this is canonical slot 1.
        sims[1].qposes[:, 9:11] += [.004, -.002]
        sims[1].qposes[:, 12:14] += [-.003, .001]
        sims[1].qposes[350:, 9:11] = sims[1].qposes[350:, 12:14]
        first_files = {}; resets = []; total_steps = 0
        guards = ExitStack()
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
            guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No native execution')))
        guards.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('No real expert')))
        try:
            with guards:
                for i, sim in enumerate(sims):
                    attempt = root / 'attempts' / f'{i:06d}'
                    images = {key: np.full((512, 512, 3), 32+64*n, np.uint8) for n, key in enumerate(CAMERAS)}
                    with sim.patches(generic):
                        reset, trace = sim.fixture(attempt, generic, episode_index=i, images=images)
                        # Fixture generated the old slot and an unused empty
                        # replay directory; retain it, use the shared primitives
                        # directly below rather than deleting fixture evidence.
                        (attempt / 'slot.json').write_text(json.dumps(slots[i]))
                        resets.append(reset)
                        for tick in range(528):
                            raw = generic._npz(attempt / 'ticks' / f'{tick:06d}.npz')
                            frame = {key: raw[key] for key in (*CAMERAS, 'action', 'observation.state')}
                            frame['task'] = raw['task'].item(); dataset.add_frame(frame)
                        dataset.save_episode(); dataset.finalize(); dataset._writer_closed_for_reading = True
                        info, entries = validate.metadata(root, i+1)
                        a, s, actual, gates, media = generic._episode_inputs(root, slots[i], sim, reset, lambda: None, info, entries)
                        self.assertEqual(media['decoded_frames'], 1056)
                        self.assertGreaterEqual(gates['stable_ticks'], 30)
                        for mode in generic.MODES:
                            result = generic.replay_controls(attempt / 'replays', sim, reset, actual, a, mode, lambda: None)
                            self.assertTrue(result['passed'])
                            self.assertEqual(generic.audit_replay(attempt / 'replays' / mode, sim, reset, actual, a, mode, lambda: None), result)
                            with self.assertRaises(FileExistsError):
                                generic.replay_controls(attempt / 'replays', sim, reset, actual, a, mode, lambda: None)
                        if i:
                            with self.assertRaisesRegex(ValueError, 'reset'):
                                generic.load_attempt(attempt, slots[i], a, s, np.arange(528, dtype=np.float32)/30, sim, resets[0], lambda: None)
                            with self.assertRaisesRegex(ValueError, 'reset'):
                                generic.audit_replay(attempt / 'replays/reference', sim, resets[0], actual, a, 'reference', lambda: None)
                            for path, digest in first_files.items(): self.assertEqual(file_hash(path), digest)
                        else:
                            first_files = {x: file_hash(x) for directory in ('dataset/data', 'dataset/videos', 'dataset/meta/episodes')
                                           for x in (root / directory).rglob('*') if x.is_file()}
                        total_steps += sim.steps
                        self.assertEqual(sim.run_calls, 0)
                self.assertEqual(total_steps, 2112)
                self.assertFalse(np.array_equal(resets[0]['qpos'], resets[1]['qpos']))
                # Fresh disk read after both serial writes; no writer buffer use.
                info, entries = validate.metadata(root, 2)
                self.assertEqual(info['total_frames'], 1056)
                self.assertEqual(len(entries), 2)
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                reopened = LeRobotDataset(repo_id='local_varied_data', root=root / 'dataset', video_backend='pyav')
                self.assertEqual(len(reopened), 1056)
                for index in (0, 528):
                    sample = reopened[index]
                    self.assertEqual(int(sample['episode_index']), index//528)
                    for key in CAMERAS: self.assertEqual(tuple(sample[key].shape), (3, 512, 512))
                reopened.finalize()
                print(f'REAL_LEROBOT_2EP root={root} rows=1056 decoded_frames=2112 replay_modes=4 numeric_fake_steps={total_steps} native_steps=0 expert_calls=0', flush=True)
        finally:
            dataset.finalize()


if __name__ == '__main__': unittest.main()