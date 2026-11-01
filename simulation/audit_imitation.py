"""Evaluate imitation with and without the simulated harness."""
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from supported_stepping_env import SupportedSteppingEnv
torch.set_num_threads(1)
model=PPO.load('assisted_imitation_start.zip',device='cpu')
rows=[]
for support in (.8,0.):
 for seed in (9001,9002,9003):
  e=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3. if support else 0.,weight_support=support)
  obs,_=e.reset(seed=seed);x0=e.data.xipos[e.body,0];peak=[0.,0.];frames=[0,0];events=[];prev=[False,False]
  for k in range(200):
   a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
   for leg in (0,1):
    peak[leg]=max(peak[leg],info['foot_clearance_m'][leg])
    swing=bool(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg]);frames[leg]+=int(swing)
    if swing and not prev[leg]:events.append(leg)
    prev[leg]=swing
   if term or trunc:break
  row=dict(support=support,seed=seed,seconds=(k+1)*.02,fell=bool(term),forward_m=float(e.data.xipos[e.body,0]-x0),peak_clearance_m=peak,supported_frames=frames,event_order=events)
  rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('assisted_imitation_audit.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
