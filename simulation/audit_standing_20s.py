"""Longer no-support evaluation of the most stable candidate found so far."""
import json,numpy as np,torch
from stable_baselines3 import PPO
from stepping_env import SteppingEnv
torch.set_num_threads(1)
model=PPO.load('training_checkpoints/continued_gpu_49152_steps.zip',device='cpu');rows=[]
for seed in (9001,9002,9003):
 e=SteppingEnv(episode_seconds=20.,disturbance=.02);obs,_=e.reset(seed=seed);tilt=0.;x0=e.data.xipos[e.body,0]
 for k in range(1000):
  a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
  tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
  if term or trunc:break
 row=dict(seed=seed,seconds=(k+1)*.02,fell=bool(term),max_tilt_deg=tilt,forward_m=float(e.data.xipos[e.body,0]-x0))
 rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('standing_20s_audit.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
