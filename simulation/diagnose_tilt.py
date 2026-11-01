import mujoco,numpy as np
from biped_env import BipedEnv
from stable_baselines3 import PPO
for kind in ['neutral','policy']:
 e=BipedEnv(episode_seconds=4);o,_=e.reset(); pol=PPO.load(r'C:\Practice_project\biped_sim\ppo_experiment_4096.zip',device='cpu') if kind=='policy' else None
 print('\n',kind,'initial q',np.round(e.data.qpos[3:7],5),'contacts',e.data.ncon)
 for step in range(200):
  act=np.zeros(5) if pol is None else pol.predict(o,deterministic=True)[0];o,r,term,trunc,info=e.step(act)
  if step in (0,24,49,99,199) or term:
   R=e.data.xmat[e.body].reshape(3,3); roll=np.degrees(np.arctan2(R[2,1],R[2,2]));pitch=np.degrees(np.arcsin(np.clip(-R[2,0],-1,1)))
   print('t',round((step+1)*.02,2),'roll',round(roll,2),'pitch',round(pitch,2),'y',round(e.data.xipos[e.body,1],4),'upright',round(info['upright'],4),'action',np.round(act,2),'contacts',e.data.ncon)
  if term or trunc:break
