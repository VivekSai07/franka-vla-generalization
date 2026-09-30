"""Validation of the measurement + deep-dive on the odd failures."""
import json, glob
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH
from experiments.local_position_probe.scenes import cube_qpos

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
d = mujoco.MjData(model)
R = json.load(open('/out/trials_classified.json'))


def load(ck, ordinal):
    root = f'{RUN}/evaluation_report_step_{ck:06d}'
    ep = glob.glob(f'{root}/trial_{ordinal:03d}_*/episode.json')[0]
    return json.load(open(ep)), np.load(ep.replace('episode.json', 'trace.npz')), root


print('=== 1. POSITIVE CONTROL for distractor-displacement measure ===')
# On a successful distractor trial the pick cube MUST have moved ~0.7m and a distractor should have ~0.
ep, t, root = load(50000, 2)
scene = np.load(f"{root}/{ep['layout_id']}_{ep['condition']}.npz")
q0, q1 = scene['qpos'], t['terminal_qpos']
print('trial', ep['layout_id'], ep['condition'], 'distractors', ep['distractors'], 'success', ep['task_success'])
for colour in ('red', 'green', 'blue', 'yellow'):
    s = cube_qpos(model, colour)
    print(f'  {colour:7s} qpos_start={s:2d}  initial xyz={np.round(q0[s:s+3], 4)}  terminal xyz={np.round(q1[s:s+3], 4)}  xy-displacement={np.linalg.norm(q1[s:s+2]-q0[s:s+2]):.4f} m')
print('  requested distractor_xy_m:', ep['distractor_xy_m'])
print('  (scene distractor initial xy should equal requested)')
# scan ALL trials: max displacement of the two distractor-colour cubes, and the two targets
mx = {c: 0.0 for c in ('blue', 'yellow')}
mxd = {c: 0.0 for c in ('blue', 'yellow')}
for r in R:
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep, t, _ = load(r['ckpt'], r['ordinal'])
    scene = np.load(f"{root}/{ep['layout_id']}_{ep['condition']}.npz")
    for c in ('blue', 'yellow'):
        s = cube_qpos(model, c)
        disp = float(np.linalg.norm(t['terminal_qpos'][s:s+3] - scene['qpos'][s:s+3]))
        (mxd if ep['condition'] == 'distractor' and c in ep['distractors'] else mx)[c] = max((mxd if ep['condition'] == 'distractor' and c in ep['distractors'] else mx)[c], disp)
print('max 3D displacement of ACTIVE distractors over all 160 distractor trials x 2 ckpts:', {k: f'{v*1000:.3f} mm' for k, v in mxd.items()})
print('max displacement of PARKED (inactive) distractor cubes:', {k: f'{v*1000:.3f} mm' for k, v in mx.items()})

print('\n=== 2. ODD FAILURES: which cube actually moved / got lifted? ===')
def fingertip(state):
    d.qpos[:9] = state; mujoco.mj_forward(model, d)
    return 0.5 * (d.xpos[model.body('left_finger').id] + d.xpos[model.body('right_finger').id])

def describe(ck, ordinal):
    ep, t, _ = load(ck, ordinal)
    pk, pl = t['pick_position'], t['place_position']
    n = len(pk)
    dpk = np.linalg.norm(pk[:, :3] - pk[0, :3], axis=1); dpl = np.linalg.norm(pl[:, :3] - pl[0, :3], axis=1)
    st = t['state']
    tips = np.array([fingertip(s) for s in st[::5]])
    # distance of fingertip midpoint to each cube over time (sampled every 5 ticks)
    dist_pick = np.linalg.norm(tips - pk[::5], axis=1); dist_place = np.linalg.norm(tips - pl[::5], axis=1)
    closed = t['applied_action'][:, 7] <= 127
    print(f"--- ckpt {ck} ord {ordinal} {ep['layout_id']} {ep['condition']} cell {ep['cell']} pick={ep['pick']} place={ep['place']} success={ep['task_success']} ticks={n}")
    print(f"    TARGET(pick) cube: moved {100*dpk.max():.1f} cm max, final z {pk[-1,2]:.3f} | OTHER(place) cube: moved {100*dpl.max():.1f} cm max, final z {pl[-1,2]:.3f}, max z {pl[:,2].max():.3f}")
    print(f"    gripper closest approach: to PICK cube {100*dist_pick.min():.1f} cm (at tick {5*int(dist_pick.argmin())}), to PLACE cube {100*dist_place.min():.1f} cm (at tick {5*int(dist_place.argmin())})")
    print(f"    ticks with gripper closed & within 6cm of PICK: {int(((dist_pick<0.06)&closed[::5]).sum()*5)}; of PLACE: {int(((dist_place<0.06)&closed[::5]).sum()*5)}")
    print(f"    final: xy={100*t['terminal_xy_m']:.1f}cm z={100*t['terminal_z_m']:.1f}cm  (place-cube z relative: {pl[-1,2]-pk[-1,2]:+.3f})  clipped_ticks={ep['joint_clipped_ticks']}")
    return ep, t

for ck, o in ((50000, 137), (50000, 138), (50000, 139), (50000, 140), (33400, 139), (33400, 71), (33400, 72), (33400, 155)):
    describe(ck, o)
