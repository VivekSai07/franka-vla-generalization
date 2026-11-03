"""Train/rollout parity of the drawn box, on replayed sealed episodes. Robosuite image, CPU, OSMesa rendering.

The sealed episodes are replayed exactly (S0's `replay.frames`, which raises on any difference from the sealed `qpos` and prop
poses). At every frame the adapter's state builder runs at the instant the M0 rollout calls it; at four frames per episode the
two cameras are also rendered (512 x 512, as M0's rollout does) and the adapter's `act` runs with the real X-VLA processors.
What is proven:

* the live box (`signals.target_box`, the function that wrote the sidecar) equals the sidecar's row byte for byte at every
  frame of every replayed episode, and the state builder returns M2's training row byte for byte (M2's parity, re-run);
* the frames the processors receive differ from the undrawn render exactly on the band computed from the sidecar box, where
  they are exactly magenta, front camera box on the front frame and gripper camera box on the gripper frame;
* the processed `image_input` (and `proprio`) the policy receives equals what the training pipeline produces from the same
  drawn frames (the loader's `draw_frame`, `collate_rows`, the same processors).

The codec difference between decoded training video and raw renders is pre-existing and shared by every arm; the comparison
here is on the same render, drawn two ways. `M3_SAMPLES_DIR` (optional) writes the user-approval sheets of the rollout frames.
"""
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

EPISODES = (0, 18, 271, 539)  # first, one with clipped force frames, a middle one, the last
FRAMES = 452
TICKS = (0, 120, 260, 451)


def modules():
    from experiments.vla_inputs_signals import replay, signals
    from experiments.xvla_p2_force import forces
    from experiments.xvla_p2_force_box import boxes, contract, eval_adapter, overlay
    return replay, signals, forces, boxes, contract, eval_adapter, overlay


def mounted():
    replay, signals, forces, boxes, contract, eval_adapter, overlay = modules()
    return all(path.exists() for path in (forces.SIDECAR/'seal.json', contract.PARENT/'seal.json'))


def have_bundle():
    return (modules()[4].CACHE/'tokenizer').exists()


def build_adapter(real_pre=None):
    """The adapter with its checkpoint loading stubbed; `real_pre` (if any) is wrapped so what it is given is recorded."""
    replay, signals, forces, boxes, contract, eval_adapter, overlay = modules()
    seen = {}

    class Policy:
        config = SimpleNamespace()

        def reset(self):
            pass

        def select_action(self, processed):
            seen['processed'] = processed
            return torch.zeros(1, 8)

    def pre(observation):
        seen['observation'] = observation
        return observation if real_pre is None else real_pre(observation)

    with patch('experiments.vla_inputs_arm_xvla.adapter.load_policy', return_value=Policy()), \
            patch('lerobot.policies.factory.make_pre_post_processors', return_value=(pre, lambda action: action)):
        adapter = eval_adapter.XVLAForceBoxAdapter(Path('/nonexistent/checkpoint'))
    return adapter, seen


def replay_episode(index, adapter, seen, sample_ticks=TICKS):
    """One exact replay: the builder's state and stored box at every frame, and at `sample_ticks` the rendered frames and one `act`."""
    import mujoco
    from experiments.vla_inputs_eval.rollout import CAMERA_NAMES
    replay, signals, forces, boxes, contract, eval_adapter, overlay = modules()
    sealed = replay.load_episode(index)
    target, instruction = sealed['episode']['object'], contract.planned_instructions()[index]
    out = dict(sealed=sealed, states={}, boxes={}, live={}, images={}, observation={}, processed={})
    holder = {}

    def on_frame(tick, model, data):
        out['live'][tick] = signals.target_box(model, data, target)
        out['states'][tick] = adapter.state_builder(model, data, target)
        out['boxes'][tick] = adapter._box.copy()
        if tick in sample_ticks:
            if 'renderer' not in holder:
                holder['renderer'] = mujoco.Renderer(model, height=512, width=512)
            frames = []
            for camera in CAMERA_NAMES:
                holder['renderer'].update_scene(data, camera=camera)
                frames.append(holder['renderer'].render().copy())
            out['images'][tick] = frames
            adapter.act(dict(images=frames, state=out['states'][tick], instruction=instruction))
            out['observation'][tick], out['processed'][tick] = seen['observation'], seen['processed']

    try:
        replay.frames(index, on_frame)
    finally:
        if 'renderer' in holder:
            holder['renderer'].close()
    return out


class BoxParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest('P2 parent or S0 sidecar not mounted')
        (cls.replay, cls.signals, cls.forces, cls.boxes, cls.c, cls.adapter_module, cls.overlay) = modules()
        cls.adapter, cls.seen = build_adapter()
        cls.cache = {index: replay_episode(index, cls.adapter, cls.seen) for index in EPISODES}

    def test_the_live_box_equals_the_sidecar_row_byte_for_byte_at_every_frame(self):
        for index, got in self.cache.items():
            self.assertEqual(sorted(got['live']), list(range(FRAMES)))
            for tick in range(FRAMES):
                row = self.boxes.table()[index, tick]
                self.assertEqual(np.asarray(got['live'][tick]).tobytes(), row.tobytes(), (index, tick))

    def test_the_state_builder_stores_that_box_and_returns_m2s_training_row_at_every_frame(self):
        for index, got in self.cache.items():
            for tick in range(FRAMES):
                self.assertEqual(got['boxes'][tick].tobytes(), self.boxes.table()[index, tick].tobytes(), (index, tick))
                row = self.c.joined_state(torch.from_numpy(got['sealed']['state'][tick]), index, tick).numpy()
                live = np.asarray(got['states'][tick], dtype=np.float32)
                self.assertEqual(live.shape, (17,))
                self.assertEqual(live.tobytes(), row.tobytes(), (index, tick))

    def test_the_live_target_is_in_frame_in_both_cameras_at_every_replayed_frame_as_in_training(self):
        for index, got in self.cache.items():
            self.assertTrue(all(float(got['live'][t][:, 4].min()) == 1.0 for t in range(FRAMES)), index)

    def test_building_the_state_and_the_box_does_not_write_the_simulation(self):
        index, before, after = EPISODES[1], {}, {}
        adapter, _ = build_adapter()
        target = self.replay.load_episode(index)['episode']['object']
        names = ('qpos', 'qvel', 'qacc', 'ctrl', 'act', 'xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'cam_xpos', 'cam_xmat',
                 'cfrc_int', 'qfrc_constraint', 'efc_force')

        def on_frame(tick, model, data):
            if tick in (0, 150, 300):
                before[tick] = {n: getattr(data, n).copy() for n in names}
                adapter.state_builder(model, data, target)
                after[tick] = {n: getattr(data, n).copy() for n in names}

        self.replay.frames(index, on_frame)
        self.assertEqual(sorted(before), [0, 150, 300])
        for tick in before:
            for name in names:
                np.testing.assert_array_equal(before[tick][name], after[tick][name], (tick, name))

    def test_the_frames_the_processors_receive_are_the_render_drawn_with_the_sidecar_box(self):
        c, o = self.c, self.overlay
        for index, got in self.cache.items():
            for tick in TICKS:
                undrawn = {key: torch.from_numpy(image).permute(2, 0, 1).float()/255
                           for key, image in zip(c.CAMERAS, got['images'][tick])}
                box = self.boxes.table()[index, tick]
                seen = got['observation'][tick]
                for camera, key in enumerate(c.CAMERAS):
                    band = torch.from_numpy(o.band(box[camera]))
                    frame = seen[key]
                    self.assertEqual((frame.dtype, tuple(frame.shape)), (torch.float32, (3, 512, 512)))
                    self.assertGreater(int(band.sum()), 0, (index, tick, key))
                    self.assertTrue(torch.equal(frame[0][band], torch.ones(int(band.sum()))), (index, tick, key))
                    self.assertTrue(torch.equal(frame[1][band], torch.zeros(int(band.sum()))), (index, tick, key))
                    self.assertTrue(torch.equal(frame[2][band], torch.ones(int(band.sum()))), (index, tick, key))
                    self.assertTrue(torch.equal(frame[:, ~band], undrawn[key][:, ~band]), (index, tick, key))
                    self.assertTrue(torch.equal(frame, o.draw(undrawn[key], box[camera])), (index, tick, key))
                self.assertFalse(torch.equal(seen[c.CAMERAS[0]], o.draw(undrawn[c.CAMERAS[0]], box[1])))  # not the gripper box

    def test_every_replayed_render_is_free_of_the_band_colour_before_drawing(self):
        """The colour check on rollout frames: nothing in the raw render is already magenta, so every magenta pixel is a band."""
        c, o = self.c, self.overlay
        for index, got in self.cache.items():
            for tick in TICKS:
                for image in got['images'][tick]:
                    frame = torch.from_numpy(image).permute(2, 0, 1).float()/255
                    self.assertEqual(o.near_magenta(frame, 0.0), 0, (index, tick))


@unittest.skipUnless(mounted() and have_bundle(), 'needs the sidecar, the P2 parent and the base bundle')
class ProcessedInputParityTests(unittest.TestCase):
    """The tensors the policy receives, not only the drawn frames: the rollout path and the training path through the same processors."""

    @classmethod
    def setUpClass(cls):
        (cls.replay, cls.signals, cls.forces, cls.boxes, cls.c, cls.adapter_module, cls.overlay) = modules()
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        config = cls.c.make_config('cpu')
        cls.policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(cls.policy)
        cls.policy.config, cls.policy.model = config, SimpleNamespace(dim_proprio=20, dim_action=20)
        cls.pre, _ = make_xvla_pre_post_processors(config, {'action': {'mean': torch.zeros(8), 'std': torch.ones(8)}})
        cls.cache = {}
        for index in EPISODES[:2]:
            adapter, seen = build_adapter(real_pre=cls.pre)
            cls.cache[index] = replay_episode(index, adapter, seen)

    def training_inputs(self, index, got):
        c = self.c
        rows = []
        for tick in TICKS:
            undrawn = {key: torch.from_numpy(image).permute(2, 0, 1).float()/255
                       for key, image in zip(c.CAMERAS, got['images'][tick])}
            row = {'episode_index': torch.tensor(index), 'task': c.planned_instructions()[index],
                   'observation.state': c.joined_state(torch.from_numpy(got['sealed']['state'][tick]), index, tick),
                   **undrawn, 'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}
            rows.append(c.OVERLAY(row, self.boxes.target_box(index, tick)))  # the loader's own hook, the real overlay
        return self.policy._build_model_inputs(self.pre(c.collate_rows(rows)))

    def test_the_processed_image_input_and_proprio_are_identical_in_training_and_rollout(self):
        for index, got in self.cache.items():
            trained = self.training_inputs(index, got)
            self.assertEqual(tuple(trained['image_input'].shape), (4, 3, 3, 224, 224))
            for k, tick in enumerate(TICKS):
                live = self.policy._build_model_inputs(got['processed'][tick])
                self.assertEqual(live['image_input'].numpy().tobytes(), trained['image_input'][k:k+1].numpy().tobytes(), (index, tick))
                self.assertEqual(live['proprio'].numpy().tobytes(), trained['proprio'][k:k+1].numpy().tobytes(), (index, tick))
                self.assertEqual(live['image_mask'].tolist(), trained['image_mask'][k:k+1].tolist())
                self.assertEqual(torch.count_nonzero(live['image_input'][:, 2]).item(), 0)  # the third view slot stays empty

    def test_the_band_survives_the_resize_to_224_as_a_visible_magenta_ridge(self):
        """Not parity, a sanity check on the design: after the processor's 512 -> 224 resize the band is still clearly magenta."""
        for index, got in self.cache.items():
            trained = self.training_inputs(index, got)['image_input']
            for k in range(4):
                for view in (0, 1):
                    image = trained[k, view]  # (3, 224, 224)
                    magenta = (image[0] > 0.8) & (image[1] < 0.2) & (image[2] > 0.8)
                    self.assertGreater(int(magenta.sum()), 100, (index, k, view))


def write_samples(directory):
    """The rollout sheets for the user's approval: both cameras, drawn exactly as the policy receives them."""
    from PIL import Image
    replay, signals, forces, boxes, c, adapter_module, overlay = modules()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter, seen = build_adapter()
    index, written = EPISODES[0], []
    got = replay_episode(index, adapter, seen)
    sheet = np.zeros((1024, len(TICKS)*512, 3), dtype=np.uint8)
    for column, tick in enumerate(TICKS):
        for line, key in enumerate(c.CAMERAS):
            panel = (got['observation'][tick][key].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            sheet[line*512:(line+1)*512, column*512:(column+1)*512] = panel
    name = f'rollout_replay_episode_{index:03d}.png'
    Image.fromarray(sheet).save(directory/name)
    (directory/'rollout_index.json').write_text(json.dumps(dict(file=name, episode=index, ticks=list(TICKS),
                                                                rows=['front camera', 'gripper camera'])))
    return name


class SampleSheetTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('M3_SAMPLES_DIR') and mounted(), 'set M3_SAMPLES_DIR to write the rollout sheets')
    def test_write_the_rollout_sheets(self):
        name = write_samples(os.environ['M3_SAMPLES_DIR'])
        self.assertTrue((Path(os.environ['M3_SAMPLES_DIR'])/name).is_file())


if __name__ == '__main__':
    unittest.main()
