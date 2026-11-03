"""The rollout side of M3: M2's 17-value state, and the target's live box drawn on both camera frames (arm A3).

Not listed in `contract.SOURCE_FILES`: fixing an adapter after training must not invalidate the training binding. M0's arm
manifest binds it (through `experiments/vla_inputs_arm_xvla_force_box`, the module M0 loads by identifier, which only re-exports
this file), together with every repository module it loads and the inputs it reads (`INPUT_DIRS`).

M0 calls the arm's state builder and then `act` at every tick, on the same simulation state, before anything steps
(`vla_inputs_eval/rollout.py`). The state builder here returns exactly what M2's returns and also stores the live box of this
instant, `signals.target_box(model, data, target)`, the very function that wrote the sidecar; `act` takes that box, draws it with
`overlay.draw_frame` on the frames after the division by 255 (the same float32 frames, the same colour and the same band as
the training loader) and then does exactly what M2's `act` does. A box is used once: `act` without a box stored since the last
`act`, or since `reset`, raises, so a stale box can never be drawn. `eval_tests/test_box_parity.py` proves the live box equals
the sidecar row byte for byte on replayed sealed episodes and that the frames the policy receives are drawn as in training.
"""
import numpy as np
import torch

from experiments.vla_inputs_arm_xvla.adapter import IMAGE_KEYS
from experiments.vla_inputs_eval.adapters import check_action, check_observation
from experiments.vla_inputs_signals import signals
from experiments.xvla_p2_force.eval_adapter import INPUT_DIRS, XVLAForceAdapter, state_force
from experiments.xvla_p2_force_box import contract, overlay
from experiments.xvla_props_p2_eval import protocol

__all__ = ["INPUT_DIRS", "XVLAForceBoxAdapter", "state_force"]


class XVLAForceBoxAdapter(XVLAForceAdapter):
    """M2's adapter with the live target box drawn on both frames: loading, processors and action handling are M2's (A0's)."""
    name = "xvla_force_box"

    def __init__(self, checkpoint, state_builder=state_force):
        self._box = None
        self.inner_state_builder = state_builder
        super().__init__(checkpoint, state_builder=self.state_and_box)

    def state_and_box(self, model, data, target):
        """M2's state at this instant; the same instant's box is kept for the `act` that follows. Reads `data`, never writes it."""
        state = self.inner_state_builder(model, data, target)
        self._box = np.asarray(signals.target_box(model, data, target), dtype=np.float64)
        return state

    def reset(self):
        self._box = None
        super().reset()

    def act(self, obs):
        check_observation(obs)
        if self._box is None:
            raise ValueError("act() needs the box the state builder stored at this tick; none is stored")
        box, self._box = self._box, None
        frame = {"observation.state": torch.tensor(obs["state"], dtype=torch.float32), "task": obs["instruction"]}
        for key, image in zip(IMAGE_KEYS, obs["images"]):
            frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        frame = overlay.draw_frame(frame, box)
        with protocol.eval_instruction(obs["instruction"]):
            processed = self.pre(contract.observation(frame))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.post(self.policy.select_action(processed)).squeeze(0).float().cpu().numpy()
        check_action(action)
        return action
