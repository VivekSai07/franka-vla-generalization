"""Stage 7: all-four-cube final poses for the weird trials + video contact sheets."""
import glob, json, sys
import numpy as np
import av
from PIL import Image, ImageDraw
import mujoco
from src.robots.franka_panda.config import SCENE_PATH
from experiments.local_position_probe.scenes import cube_qpos

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
TRIALS = [(33400, 71), (33400, 139), (50000, 137), (50000, 116), (50000, 157), (33400, 22), (33400, 155)]


def paths(ck, o):
    root = f'{RUN}/evaluation_report_step_{ck:06d}'
    ep = glob.glob(f'{root}/trial_{o:03d}_*/episode.json')[0]
    return root, ep


print('=== final pose of ALL FOUR cubes (xyz, metres) ===')
for ck, o in TRIALS:
    root, ep = paths(ck, o)
    d = json.load(open(ep)); t = np.load(ep.replace('episode.json', 'trace.npz'))
    scene = np.load(f"{root}/{d['layout_id']}_{d['condition']}.npz")
    print(f"--- ck {ck} ord {o} {d['layout_id']} {d['condition']} pick={d['pick']} place={d['place']} distractors={d['distractors']} ticks={d['ticks']}")
    for c in ('red', 'green', 'blue', 'yellow'):
        s = cube_qpos(model, c)
        a, b = scene['qpos'][s:s + 3], t['terminal_qpos'][s:s + 3]
        tag = ' (PICK)' if c == d['pick'] else ' (BASE)' if c == d['place'] else ' (distractor)' if c in d['distractors'] else ' (parked)'
        print(f"    {c:7s}{tag:14s} start {np.round(a, 3)} -> end {np.round(b, 3)}  moved {100*np.linalg.norm(b[:2]-a[:2]):5.1f} cm xy, dz {100*(b[2]-a[2]):+5.1f} cm")
    arm = t['terminal_state'][:9]
    print('    terminal arm joints:', np.round(arm[:7], 2), ' gripper width-ish:', np.round(arm[7:9], 3))


def sheet(ck, o, ticks):
    root, ep = paths(ck, o)
    d = json.load(open(ep)); mp4 = ep.replace('episode.json', 'cameras.mp4')
    frames = {}
    with av.open(mp4) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in ticks: frames[i] = fr.to_image()
    tiles = []
    for tk in ticks:
        if tk not in frames: continue
        im = frames[tk].resize((640, 320))
        dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 235, 16], fill=(0, 0, 0))
        dr.text((3, 2), f"ck{ck} o{o} {d['layout_id'][-3:]} t={tk} ok={d['task_success']}", fill=(255, 255, 0))
        tiles.append(im)
    W = 2; H = (len(tiles) + 1) // 2
    out = Image.new('RGB', (640 * W, 320 * H))
    for k, im in enumerate(tiles): out.paste(im, (640 * (k % W), 320 * (k // W)))
    out.save(f'/out/sheet_ck{ck}_o{o}.png')
    print('wrote', f'/out/sheet_ck{ck}_o{o}.png', 'frames', sorted(frames))


sheet(50000, 137, [0, 100, 250, 600, 900, 1199])
sheet(33400, 139, [0, 150, 300, 450, 700, 1199])
sheet(33400, 71, [0, 200, 400, 700, 1000, 1199])
sheet(50000, 116, [0, 100, 200, 500, 900, 1199])
sheet(50000, 157, [0, 250, 300, 320, 400, 1199])
