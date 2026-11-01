import math,numpy as np
from larger_action_env import LargerActionBipedEnv
for s in (-1,1):
 for k in (-1,0,1):
  for phase in (0.,np.pi/2):
   e=LargerActionBipedEnv(episode_seconds=3);obs,_=e.reset(seed=7001);x0=e.data.xipos[e.body,0];mx=0
   for i in range(150):
    t=i*.02;u=.6*math.sin(2*math.pi*t);v=.6*math.sin(2*math.pi*t+phase);action=np.array([u,k*v,s*u,s*k*v,0.]);obs,r,term,tr,info=e.step(action);mx=max(mx,np.degrees(np.arccos(np.clip(info['upright'],-1,1))))
    if term or tr:break
   print('s',s,'k',k,'phase',round(phase,2),'time',round((i+1)*.02,2),'distance mm',round((e.data.xipos[e.body,0]-x0)*1000,2),'tilt',round(mx,1),'switches',info.get('support_switches'),flush=True)
