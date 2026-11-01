"""Probe coordinating both legs to lift one foot without a virtual harness."""
import itertools,json
import numpy as np
from stepping_env import SteppingEnv
rows=[]
for leg,lift,stance in itertools.product((0,1),(.5,1.),(-1.,-.5,0.,.5,1.)):
 e=SteppingEnv(episode_seconds=2.,disturbance=0.);e.reset(seed=0);a=np.zeros(5)
 for j,f in ((leg,lift),(1-leg,stance)):
  a[2*j:2*j+2]=np.array([-1.,1.] if j==0 else [1.,-1.])*f
 peak=0.;frames=0;max_tilt=0.
 for k in range(100):
  _,_,term,trunc,info=e.step(a);peak=max(peak,info['foot_clearance_m'][leg])
  frames+=int(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg])
  max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
  if term or trunc:break
 row=dict(leg=leg,lift=lift,stance=stance,peak_mm=peak*1000,supported_frames=frames,fell=bool(term),seconds=(k+1)*.02,max_tilt_deg=max_tilt)
 rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('weight_shift_probe.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
