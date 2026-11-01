"""Test a virtual body-weight support harness; for training exploration only."""
import json
import numpy as np
from supported_stepping_env import SupportedSteppingEnv
rows=[]
for fraction in (0., .5, .8):
 for leg in (0,1):
  e=SupportedSteppingEnv(episode_seconds=2.,disturbance=0.,support_gain=3.,weight_support=fraction)
  e.reset(seed=0);action=np.zeros(5);action[2*leg:2*leg+2]=([-1,1] if leg==0 else [1,-1])
  peak=0.; valid=0
  for k in range(100):
   _,_,term,trunc,info=e.step(action);peak=max(peak,info['foot_clearance_m'][leg]);valid+=int(info['valid_swing'])
   if term or trunc:break
  row=dict(weight_support=fraction,leg=leg,lift_mm=peak*1000,supported_frames=valid,fell=bool(term),seconds=(k+1)*.02)
  rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('weight_support_probe.json','w',encoding='utf8') as f: json.dump(rows,f,indent=2)
