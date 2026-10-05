"""Replay one gate attempt without rendering and print hand, target and joint-limit margin per phase.

    python -m experiments.props_scene.diagnose_attempt <attempt id> '<json constant overrides>'
"""
import sys
import json
import numpy as np
import mujoco
from experiments.props_scene import constants as c
for key, value in json.loads(sys.argv[2]).items():
    setattr(c, key, value)
from experiments.props_scene import benchmark as b
from experiments.props_scene.validate import execute

ident = sys.argv[1]
attempt = next(a for a in b.build_plan() if a["id"] == ident)
print(ident, [round(v, 3) for v in attempt["pose"]])
result, actions, traces, check, digest, err = execute(attempt, render=False)
print("result", result["success"], result["reason"])
model = result["model"]
data = mujoco.MjData(model)
lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
last = None
for i, t in enumerate(traces):
    p = t["phase_index"]
    if p == last and i % 10:
        continue
    last = p
    data.qpos[:] = t["qpos"]
    mujoco.mj_forward(model, data)
    hand = data.xpos[model.body("robot0_right_hand").id]
    margin = np.minimum(t["qpos"][:7] - lo, hi - t["qpos"][:7])
    print(f"tick {i:3d} {c.PHASE_NAMES[p]:8s} hand {np.round(hand,3)} target {np.round(t['target_pos'],3)} "
          f"err {np.linalg.norm(hand-t['target_pos']):.3f} q4 {t['qpos'][3]:.3f} min-margin {margin.min():.3f} (j{margin.argmin()+1}) q1 {t['qpos'][0]:.2f} q7 {t['qpos'][6]:.2f}")
