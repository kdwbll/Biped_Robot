import numpy as np
from biped_env import BipedEnv
for seed in range(5):
 e=BipedEnv(episode_seconds=3);e.action_space.seed(seed);o,_=e.reset(seed=seed);good=True
 for i in range(150):
  o,r,term,trunc,info=e.step(e.action_space.sample());good &= bool(np.isfinite(o).all() and np.isfinite(r))
  if term or trunc:break
 print('seed',seed,'seconds',round((i+1)*.02,2),'reason','fallen' if term else 'timeout' if trunc else 'other','finite',good,'upright',round(info['upright'],3))
