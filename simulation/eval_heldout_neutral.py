import numpy as np
from safer_clocked_env import SaferClockedEnv
for seed in (9001,9002,9003,9004,9005):
 e=SaferClockedEnv(episode_seconds=4,disturbance=.02);o,_=e.reset(seed=seed);x=float(e.data.xipos[e.body,0]);roll=0
 for k in range(200):
  o,r,term,tr,info=e.step(np.zeros(5));roll=max(roll,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
  if term or tr:break
 print(seed,'distance mm',round((float(e.data.xipos[e.body,0])-x)*1000,3),'tilt',round(roll,2),'fell',term,flush=True)
