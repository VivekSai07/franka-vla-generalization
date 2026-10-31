"""The equivalence gate: the M0 rollout + X-VLA adapter reproduce recorded P2 report trials of checkpoint 101,520.

Robosuite image, GPU, the P2 bundle at /bundle. The four trials are fixed by a rule written before any M0 rollout ran: the two
lowest-ordinal successes and the two lowest-ordinal failures of the recorded report tier. Every P2 trace array and the
episode.json outcome fields must be byte-equal; the three added signal arrays are only checked for shape and finiteness.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    raise unittest.SkipTest("needs a GPU")

import mujoco

from experiments.vla_inputs_arm_xvla.adapter import XVLAAdapter
from experiments.vla_inputs_eval import adapters, rollout, scenes
from experiments.xvla_props_p2_eval import protocol
from experiments.xvla_props_p2_eval import rollout as p2_rollout

BUNDLE = Path(os.environ.get("BUNDLE", "/bundle"))
CHECKPOINT = BUNDLE / "checkpoints/step_101520"
RECORDED = BUNDLE / "evaluation/report_p2_step_101520"
GATE_ORDINALS = (1, 2, 14, 15)  # two lowest-ordinal successes, two lowest-ordinal failures
FIELDS = ("outcome", "cause", "ticks", "placed_bin", "first_lifted", "task_success", "final_gripper_open", "wrong_object",
          "names", "distractor_displacement_m")


def recorded_rows():
    return sorted((json.loads(p.read_text()) for p in RECORDED.glob("trial_*/episode.json")), key=lambda r: r["ordinal"])


class TheRuleIsAsWritten(unittest.TestCase):
    def test_the_four_ordinals_are_the_two_lowest_successes_and_failures(self):
        rows = recorded_rows()
        successes = [r["ordinal"] for r in rows if r["task_success"]][:2]
        failures = [r["ordinal"] for r in rows if not r["task_success"]][:2]
        self.assertEqual(tuple(successes + failures), GATE_ORDINALS)


class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.use_deterministic_algorithms(True)
        cls.adapter = XVLAAdapter(CHECKPOINT)
        cls.plan = json.loads((BUNDLE / "dataset_provenance/plan.json").read_text())["evaluation_holdout"]
        cls.trials = {t["ordinal"]: t for t in protocol.report_trials()}

    def run_ordinal(self, ordinal):
        trial = self.trials[ordinal]
        model, data, names = scenes.restore(self.plan[trial["layout_id"]])
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
            row = rollout.run(self.adapter, self.adapter.state_builder, model, data, names, renderer, trial, Path(tmp))
            with np.load(Path(tmp) / "trace.npz") as stored:
                arrays = {k: stored[k] for k in stored.files}
        folder = protocol.trial_folder(RECORDED, trial)
        with np.load(folder / "trace.npz") as stored:
            recorded = {k: stored[k] for k in stored.files}
        return row, arrays, json.loads((folder / "episode.json").read_text()), recorded

    def test_the_gate_trials_reproduce_the_recorded_ones(self):
        for ordinal in GATE_ORDINALS:
            row, arrays, episode, recorded = self.run_ordinal(ordinal)
            for field in FIELDS:
                self.assertEqual(row[field], episode[field], (ordinal, field))
            self.assertEqual(set(recorded), set(rollout.TRACE_KEYS))
            for key in rollout.TRACE_KEYS:
                self.assertEqual(arrays[key].dtype, recorded[key].dtype, (ordinal, key))
                self.assertTrue(np.array_equal(arrays[key], recorded[key]), (ordinal, key))
            ticks = row["ticks"]
            self.assertEqual(arrays["wrist_wrench"].shape, (ticks, 6))
            self.assertEqual(arrays["finger_force"].shape, (ticks, 2))
            self.assertEqual(arrays["target_box"].shape, (ticks, 2, 5))
            self.assertTrue(all(np.isfinite(arrays[k]).all() for k in rollout.SIGNAL_KEYS))
            print(f"GATE ordinal {ordinal}: {row['outcome']} {row['cause']} ticks {ticks} equal", flush=True)

    def test_p2s_own_rollout_function_agrees_on_a_short_prefix_too(self):
        """Independent of the recorded files: P2's rollout and this one, same policy, same seed, same first ticks."""
        trial = self.trials[1]
        outputs = []
        for run in ("p2", "m0"):
            model, data, names = scenes.restore(self.plan[trial["layout_id"]])
            with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
                if run == "m0":
                    rollout.run(self.adapter, self.adapter.state_builder, model, data, names, renderer, trial, Path(tmp), max_ticks=25)
                else:
                    p2_rollout.MAX_TICKS = 25
                    try:
                        p2_rollout.rollout(self.adapter.policy, self.adapter.pre, self.adapter.post, model, data, names,
                                           renderer, trial, Path(tmp))
                    finally:
                        p2_rollout.MAX_TICKS = rollout.MAX_TICKS
                with np.load(Path(tmp) / "trace.npz") as stored:
                    outputs.append({k: stored[k] for k in rollout.TRACE_KEYS})
        for key in rollout.TRACE_KEYS:
            self.assertTrue(np.array_equal(outputs[0][key], outputs[1][key]), key)


if __name__ == "__main__":
    unittest.main()
