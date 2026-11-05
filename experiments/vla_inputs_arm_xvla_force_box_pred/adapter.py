"""Arm A5 of the M0 evaluator: A3's frozen policy fed a box predicted from the camera images (M3b, spec D6). No policy is trained.

M0 loads an arm by `module:Class` from a module named `experiments.vla_inputs_arm_*`. The code is `adapter.py` here; the class is A3's
`XVLAForceBoxAdapter` (loading, M2's 17-value force state, the processors, the policy call, the overlay) with one change: at every `act` the box that
is drawn is the one `experiments.vla_inputs_box_detector.predictor.Predictor` returns for the raw renders and the instruction, never the oracle.
The oracle box the state builder computes (the very `signals.target_box` of A3) is moved at once into `oracle_box`, a diagnostic that `act` never
reads and that is only written to the log; `tests/test_adapter.py` proves the drawn frames do not change when it does, and that the same check
fails for an adapter mutated to draw it.

Per `act` one JSON line goes to the log directory (default `/out/arms/A5/predicted_box_logs`, the campaign's own folder in the container):
`{"trial": "p<pid>-r<reset count>", "tick", "predicted": [[5], [5]], "oracle": [[5], [5]]}`; `experiments/vla_inputs_m3b_analysis` matches a trial's lines to
its trace by the oracle sequence. The forces in the state stay the exact simulator values, as in A3.

`INPUT_DIRS` adds the trained detector's final directory (`detector.pt`, `detector.json`) to A3's, so M0's `prepare` hashes every file and
every worker and `compare` re-verifies it.
"""
import json
import os
from pathlib import Path

import numpy as np

from experiments.vla_inputs_arm_xvla_force_box.adapter import INPUT_DIRS as A3_INPUT_DIRS
from experiments.vla_inputs_arm_xvla_force_box.adapter import XVLAForceBoxAdapter, state_force
from experiments.vla_inputs_eval.adapters import check_observation
from experiments.vla_inputs_box_detector import predictor

DETECTOR_DIRECTORY = "outputs/vla_inputs_box_detector/run_v1/final"
INPUT_DIRS = (*A3_INPUT_DIRS, DETECTOR_DIRECTORY)
LOG_DIRECTORY = Path("/out/arms/A5/predicted_box_logs")
ROOT = Path(__file__).resolve().parents[2]

__all__ = ["DETECTOR_DIRECTORY", "INPUT_DIRS", "XVLAPredictedBoxAdapter", "state_force"]


class XVLAPredictedBoxAdapter(XVLAForceBoxAdapter):
    name = "xvla_force_box_predicted"

    def __init__(self, checkpoint, state_builder=state_force, detector=None, log_dir=LOG_DIRECTORY):
        self.oracle_box, self._resets, self._tick, self._log_file = None, 0, 0, None
        self._log_dir = None if log_dir is None else Path(log_dir)
        if detector is None:
            import torch
            detector = predictor.Predictor.load(ROOT / DETECTOR_DIRECTORY, "cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector
        super().__init__(checkpoint, state_builder=state_builder)

    def reset(self):
        self._resets += 1
        self._tick, self.oracle_box = 0, None
        super().reset()

    def state_and_box(self, model, data, target):
        state = super().state_and_box(model, data, target)
        self.oracle_box, self._box = self._box, None  # the privileged box leaves the drawing path here
        return state

    def act(self, obs):
        check_observation(obs)
        self._box = np.asarray(self.detector.predict(obs["images"], obs["instruction"]), dtype=np.float64)
        self._write_log(self._box)
        return super().act(obs)

    def _write_log(self, predicted):
        if self._log_dir is not None:
            if self._log_file is None:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                self._log_file = (self._log_dir / f"worker_{os.getpid()}.jsonl").open("a")
            oracle = None if self.oracle_box is None else self.oracle_box.tolist()
            self._log_file.write(json.dumps(dict(trial=f"p{os.getpid()}-r{self._resets}", tick=self._tick, predicted=predicted.tolist(), oracle=oracle)) + "\n")
            self._log_file.flush()
        self._tick += 1
