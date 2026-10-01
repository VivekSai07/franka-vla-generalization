import glob, json
import numpy as np
RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
for ck, o in ((33400, 140), (33400, 71), (50000, 159)):
    root = f'{RUN}/evaluation_report_step_{ck:06d}'
    ep = glob.glob(f'{root}/trial_{o:03d}_*/episode.json')[0]
    d = json.load(open(ep)); t = np.load(ep.replace('episode.json', 'trace.npz'))
    pk, pl = t['pick_position'], t['place_position']; grip = t['applied_action'][:, 7] <= 127
    bil = np.asarray(t['bilateral_target_contact'], bool)
    print(f"=== ck {ck} ord {o} {d['layout_id']} {d['condition']} pick={d['pick']} place={d['place']} ({d['outcome']}) ===")
    print(' tick : pickZ_above_table  baseZ_above_table  gripper  bilateral(on pick)  pick_xy_from_start  base_xy_from_start')
    for tk in range(0, 1200, 60):
        print(f' {tk:4d} : {100*(pk[tk,2]-pk[0,2]):6.1f} cm        {100*(pl[tk,2]-pl[0,2]):6.1f} cm        {"closed" if grip[tk] else "open  "}   {str(bool(bil[tk])):5s}            {100*np.linalg.norm(pk[tk,:2]-pk[0,:2]):6.1f}             {100*np.linalg.norm(pl[tk,:2]-pl[0,:2]):6.1f}')
