import numpy as np
from gymnasium.utils.env_checker import check_env
from biped_env import BipedEnv
e=BipedEnv(episode_seconds=1.0);check_env(e,skip_render_check=True)
for mode in ('neutral','random'):
 obs,_=e.reset(seed=42); total=0
 for i in range(50):
  action=np.zeros(5,dtype=np.float32) if mode=='neutral' else e.action_space.sample()
  obs,r,term,trunc,info=e.step(action);total+=r
  if term or trunc:break
 print(mode,'steps',i+1,'terminated',term,'truncated',trunc,'upright',round(info['upright'],3),'reward',round(total,3),'obs finite',np.isfinite(obs).all())
