"""Compare hand-designed cyclic commands, with and without virtual support."""
import itertools,json
import numpy as np
from supported_stepping_env import SupportedSteppingEnv
rows=[]
for support,amp,period in itertools.product((.0,.8),(.4,.7,1.),(1.,2.)):
 e=SupportedSteppingEnv(episode_seconds=4.,disturbance=0.,support_gain=3. if support else 0.,weight_support=support)
 e.reset(seed=0);x0=e.data.xipos[e.body,0];frames=[0,0];peak=[0.,0.];tilt=0.
 for k in range(200):
  phase=2*np.pi*k*.02/period;s0=max(0.,np.sin(phase));s1=max(0.,-np.sin(phase))
  action=amp*np.array([-s0,s0,s1,-s1,0.]);_,_,term,trunc,info=e.step(action)
  for leg in (0,1):
   peak[leg]=max(peak[leg],info['foot_clearance_m'][leg]);frames[leg]+=int(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg])
  tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
  if term or trunc:break
 row=dict(weight_support=support,amplitude=amp,period=period,seconds=(k+1)*.02,fell=bool(term),
  forward_m=float(e.data.xipos[e.body,0]-x0),peak_clearance_m=peak,supported_frames=frames,max_tilt_deg=tilt)
 rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('alternating_commands_probe.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
