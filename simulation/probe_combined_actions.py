"""Probe coupled servo commands under temporary support; exploratory only."""
import itertools, json
import numpy as np
from supported_stepping_env import SupportedSteppingEnv

rows=[]
for leg, (a,b), waist in itertools.product((0,1), ((1,1),(1,-1),(-1,1),(-1,-1)), (-1,0,1)):
    e=SupportedSteppingEnv(episode_seconds=1.5, disturbance=0., support_gain=3.)
    e.reset(seed=0); action=np.zeros(5); action[2*leg:2*leg+2]=[a,b]; action[4]=waist
    peak=0.; supported=0; max_tilt=0.
    for k in range(75):
        _,_,term,trunc,info=e.step(action)
        peak=max(peak,info['foot_clearance_m'][leg]); supported+=int(info['valid_swing'])
        max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
        if term or trunc: break
    row=dict(leg=leg,action=[a,b],waist=waist,peak_mm=peak*1000,supported_frames=supported,
             max_tilt_deg=max_tilt,fell=bool(term)); rows.append(row); e.close()
    print(json.dumps(row),flush=True)
with open('combined_action_probe.json','w',encoding='utf8') as f: json.dump(rows,f,indent=2)
