import json, glob, numpy as np
import sys
run=sys.argv[1]
rows=[]
for d in sorted(glob.glob(run+"/rollouts/*")):
    e=json.load(open(d+"/episode.json")); t=np.load(d+"/trace.npz")
    lift=t["pick_lift_m"]; grip=t["applied_action"][:,7]; xy=t["xy_m"]; z=t["z_m"]
    closed=grip<127
    first_close=int(np.argmax(closed)) if closed.any() else -1
    lifted=lift>0.03
    first_lift=int(np.argmax(lifted)) if lifted.any() else -1
    reopen=np.flatnonzero((~closed[1:])&closed[:-1])+1
    released_after_lift=[int(i) for i in reopen if first_lift>=0 and i>first_lift]
    start_xy=float(xy[0]); min_xy=float(xy.min()); 
    # horizontal travel of the pick cube from its start
    pp=t["pick_position"]; travel=float(np.linalg.norm(pp[:,:2]-pp[0,:2],axis=1).max())
    if e["task_success"]: cls="success"
    elif first_lift<0: cls="acquisition (never lifted)"
    elif not released_after_lift: cls="carried, never released"
    else: cls="released off target"
    rows.append(dict(scene=e["scene"], key=d.split("/")[-1], pick=e["pick"], arr=e["arrangement"], cls=cls, ticks=e["ticks"],
        first_close=first_close, first_lift=first_lift, release=released_after_lift[:1], n_close=int((closed[1:]&~closed[:-1]).sum()),
        start_xy=round(start_xy,3), min_xy=round(min_xy,3), final_xy=round(float(xy[-1]),3), final_z=round(float(z[-1]),3), travel=round(travel,3), maxlift=round(float(lift.max()),3)))
for s in ("mujoco","robosuite"):
    print("=====",s)
    for r in rows:
        if r["scene"]==s: print(f"{r['key'][len(s)+1:]:22s} {r['pick']:5s} {r['arr']:8s} {r['cls']:26s} close@{r['first_close']:4d} lift@{r['first_lift']:4d} rel{r['release']} ncl={r['n_close']} startxy={r['start_xy']} minxy={r['min_xy']} finxy={r['final_xy']} finz={r['final_z']} travel={r['travel']} maxlift={r['maxlift']}")
    from collections import Counter
    print(Counter(r["cls"] for r in rows if r["scene"]==s))
    for k in ("pick","arr"):
        print(k, {v:(sum(r["cls"]=="success" for r in rows if r["scene"]==s and r[k]==v), sum(1 for r in rows if r["scene"]==s and r[k]==v)) for v in sorted(set(r[k] for r in rows))})
json.dump(rows, open(run+"/classified.json","w"), indent=1)
