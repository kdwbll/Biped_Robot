"""Measure how assisted imitation behaves as the virtual harness is reduced."""
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from supported_stepping_env import SupportedSteppingEnv
torch.set_num_threads(1);model=PPO.load('assisted_imitation_start.zip',device='cpu');rows=[]
for fraction in (1.,.8,.6,.4,.2,0.):
 e=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3*fraction,weight_support=.8*fraction)
 obs,_=e.reset(seed=9001);x0=e.data.xipos[e.body,0];frames=[0,0];peak=[0.,0.]
 for k in range(200):
  a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
  for leg in (0,1):
   peak[leg]=max(peak[leg],info['foot_clearance_m'][leg]);frames[leg]+=int(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg])
  if term or trunc:break
 row=dict(support_fraction=fraction,weight_fraction=.8*fraction,seconds=(k+1)*.02,fell=bool(term),
          forward_m=float(e.data.xipos[e.body,0]-x0),peak_clearance_m=peak,supported_frames=frames)
 rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('imitation_support_levels.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
