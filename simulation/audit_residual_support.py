"""Evaluate a residual policy at several harness levels and without support."""
import json,sys
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from residual_gait_env import ResidualGaitEnv
torch.set_num_threads(1);path=Path(sys.argv[1]);model=PPO.load(str(path),device='cpu');rows=[]
weights=(float(sys.argv[2]),0.) if len(sys.argv)>2 else (.64,0.)
for weight in weights:
 for seed in (9001,9002,9003):
  e=ResidualGaitEnv(support_gain=3.75*weight,weight_support=weight)
  obs,_=e.reset(seed=seed);x0=e.data.xipos[e.body,0];frames=[0,0];peak=[0.,0.]
  for k in range(200):
   a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
   for leg in (0,1):
    peak[leg]=max(peak[leg],info['foot_clearance_m'][leg]);frames[leg]+=int(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg])
   if term or trunc:break
  row=dict(support=weight,seed=seed,seconds=(k+1)*.02,fell=bool(term),forward_m=float(e.data.xipos[e.body,0]-x0),peak_clearance_m=peak,supported_frames=frames)
  rows.append(row);e.close();print(json.dumps(row),flush=True)
path.with_suffix(f'.support_{int(weights[0]*100)}_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
