"""Audit sustained supported lifts and stance-foot drift for a residual policy."""
import json,sys
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from residual_gait_env import ResidualGaitEnv
torch.set_num_threads(1);path=Path(sys.argv[1]);weight=float(sys.argv[2]);model=PPO.load(str(path),device='cpu');rows=[]
for seed in (9001,9002,9003):
 e=ResidualGaitEnv(episode_seconds=4.,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
 obs,_=e.reset(seed=seed);x0=e.data.xipos[e.body,0];run=[0,0];events=[];stance_drift=0.;last_pos=np.array([e.data.geom_xpos[g,:2].copy() for g in e.foot_geom]);last_contact=[False,False]
 for k in range(200):
  a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
  pos=np.array([e.data.geom_xpos[g,:2].copy() for g in e.foot_geom]);contacts=info['foot_contacts']
  for leg in (0,1):
   if contacts[leg] and last_contact[leg]:stance_drift+=float(np.linalg.norm(pos[leg]-last_pos[leg]))
   lifted=info['foot_clearance_m'][leg]>.003 and contacts[1-leg]
   run[leg]=run[leg]+1 if lifted else 0
   if run[leg]==3:events.append(dict(leg=leg,seconds=(k-1)*.02))
  last_pos=pos;last_contact=contacts
  if term or trunc:break
 row=dict(seed=seed,weight_support=weight,fell=bool(term),seconds=(k+1)*.02,forward_m=float(e.data.xipos[e.body,0]-x0),sustained_lift_events=events,stance_drift_m=stance_drift)
 rows.append(row);e.close();print(json.dumps(row),flush=True)
path.with_suffix(f'.sequence_{int(weight*100)}_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
