"""Cross-check the GPU trial on the GPU; compare separately with CPU audit."""
import json,numpy as np,torch
from stable_baselines3 import PPO
from warp_vec_env import WarpVecEnv
torch.set_num_threads(1);model=PPO.load('warp_trial_32768.zip',device='cuda');rows=[]
for weight in (.5,0.):
 for seed in (9101,9102,9103):
  e=WarpVecEnv(nworld=1,weight=weight,seconds=4.,seed=seed);obs=e.reset();x0=e._arrays()['body'][0,0];peak=np.zeros(2);run=np.zeros(2,int);order=[]
  for k in range(200):
   a,_=model.predict(obs,deterministic=True);obs,reward,done,infos=e.step(a);info=infos[0]
   peak=np.maximum(peak,info['foot_clearance_m'])
   for leg in (0,1):
    swing=info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg]
    run[leg]=run[leg]+1 if swing else 0
    if run[leg]==3:order.append(leg)
   if done[0]:break
  # VecEnv auto-resets on timeout, so displacement at the terminal step is not available here.
  row=dict(weight=weight,seed=seed,seconds=(k+1)*.02,fell=bool(done[0] and not info['TimeLimit_truncated']),
   peak_clearance_m=peak.tolist(),events=order,overflow=info['overflow'])
  rows.append(row);e.close();print(json.dumps(row),flush=True)
open('warp_trial_gpu_evaluation.json','w',encoding='utf8').write(json.dumps(rows,indent=2))
