"""Inference boundaries of the 17-value policy: replanning happens once every ten ticks, and what happens in between does not matter.

Needs a force checkpoint (the preflight run's `step_000002` is enough: the architecture and the saved processors are what is under
test, not what was learned) and the GPU. Robosuite image; `FORCE_CHECKPOINT` names the checkpoint directory (default `/ckpt`).
The ticks are the M0 rollout's: one observation and one action per tick at 30 Hz; the policy queues ten actions per inference
(`n_action_steps` 10), so it replans at 3 Hz. This is instantaneous conditioning at chunk boundaries, not force feedback.
"""
import json
import os
import unittest
from pathlib import Path

import numpy as np

CHECKPOINT = Path(os.environ.get('FORCE_CHECKPOINT', '/ckpt'))
TICKS, CHUNK = 30, 10

try:
    import torch
    HAVE_GPU = torch.cuda.is_available()
except ImportError:
    HAVE_GPU = False


def images(fill):
    return [np.full((512, 512, 3), fill, np.uint8), np.full((512, 512, 3), 255-fill, np.uint8)]


@unittest.skipUnless(HAVE_GPU and (CHECKPOINT/'model.safetensors').is_file(), 'needs the GPU and a force checkpoint')
class InferenceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.xvla_p2_force import contract, eval_adapter, forces
        cls.contract, cls.forces = contract, forces
        cls.adapter = eval_adapter.XVLAForceAdapter(CHECKPOINT)
        cls.instruction = contract.planned_instructions()[0]
        cls.base = np.linspace(-0.5, 0.5, 9)

    def observation(self, tick, extras=None, fill=0):
        extras = self.forces.extra_state(0, tick).astype(np.float64) if extras is None else extras
        return dict(images=images(fill), state=np.concatenate([self.base+1e-3*tick, extras]), instruction=self.instruction)

    def run_ticks(self, observation_at):
        """Actions and the ticks at which the policy ran, for 30 ticks; `observation_at(tick)` gives each observation."""
        adapter, inferences = self.adapter, []
        original = adapter.policy._get_action_chunk
        tick = {'now': 0}

        def counted(batch, *args, **kwargs):
            inferences.append(tick['now'])
            torch.manual_seed(7+tick['now'])  # the chunk's noise depends on the tick it is drawn at, not on history
            return original(batch, *args, **kwargs)

        adapter.policy._get_action_chunk = counted
        try:
            adapter.reset()
            actions = []
            for tick['now'] in range(TICKS):
                actions.append(adapter.act(observation_at(tick['now'])))
        finally:
            del adapter.policy._get_action_chunk
        return np.stack(actions), inferences

    def test_the_policy_replans_at_three_hertz_and_queues_ten_actions_per_inference(self):
        actions, inferences = self.run_ticks(self.observation)
        self.assertEqual(inferences, [0, 10, 20])
        self.assertEqual(actions.shape, (TICKS, 8))
        self.assertTrue(np.isfinite(actions).all())
        self.assertEqual(self.adapter.policy.config.n_action_steps, CHUNK)
        print('INFERENCE_BOUNDARIES', json.dumps(dict(ticks=TICKS, inference_ticks=inferences, n_action_steps=CHUNK,
                                                      hz=30, replanning_hz=30/CHUNK)))

    def test_what_is_observed_between_inferences_does_not_change_the_queued_actions(self):
        boundary = {0, 10, 20}
        reference, _ = self.run_ticks(self.observation)
        rng = np.random.default_rng(1)

        def perturbed(tick):
            if tick in boundary:
                return self.observation(tick)
            return self.observation(tick, extras=rng.normal(size=8)*5., fill=int(rng.integers(0, 255)))

        other, inferences = self.run_ticks(perturbed)
        self.assertEqual(inferences, [0, 10, 20])
        self.assertEqual(other.tobytes(), reference.tobytes())

    def test_what_is_observed_at_an_inference_tick_does_change_the_next_chunk(self):
        reference, _ = self.run_ticks(self.observation)
        forced = lambda tick: self.observation(tick, extras=np.full(8, 5.)) if tick == 10 else self.observation(tick)
        other, _ = self.run_ticks(forced)
        self.assertEqual(other[:10].tobytes(), reference[:10].tobytes())  # the first chunk was planned before tick 10
        self.assertNotEqual(other[10:20].tobytes(), reference[10:20].tobytes())

    def test_the_checkpoints_saved_processor_gives_the_training_processors_proprio(self):
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        from types import SimpleNamespace
        c = self.contract
        config = c.make_config('cpu')
        config.device = 'cuda'
        trained, _ = make_xvla_pre_post_processors(config, {'action': {'mean': torch.zeros(8), 'std': torch.ones(8)}})
        stub = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(stub)
        stub.config, stub.model = config, SimpleNamespace(dim_proprio=20, dim_action=20)
        rows = [self.observation(t) for t in (0, 100, 451)]

        def proprio(pre, row):
            frame = {'observation.state': torch.tensor(row['state'], dtype=torch.float32), 'task': row['instruction'],
                     **{key: torch.from_numpy(image).permute(2, 0, 1).float()/255
                        for key, image in zip(c.CAMERAS, row['images'])}}
            from experiments.xvla_props_p2_eval import protocol
            with protocol.eval_instruction(row['instruction']):
                return stub._build_model_inputs(pre(c.observation(frame)))['proprio'].cpu().numpy()

        for row in rows:
            self.assertEqual(proprio(self.adapter.pre, row).tobytes(), proprio(trained, row).tobytes())
            self.assertEqual(proprio(self.adapter.pre, row).shape, (1, 20))


if __name__ == '__main__':
    unittest.main()
