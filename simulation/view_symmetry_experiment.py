"""View mirrored upper-part hypothesis; simulation only."""
import time
import numpy as np
import mujoco.viewer
from biped_env import BipedEnv
e=BipedEnv(episode_seconds=8,model_path='biped_mirrored_part_centered_experiment.xml');obs,_=e.reset()
with mujoco.viewer.launch_passive(e.model,e.data) as v:
 while v.is_running():
  start=time.perf_counter();obs,r,done,trunc,info=e.step(np.zeros(5))
  if done or trunc:obs,_=e.reset()
  v.sync();delay=.02-(time.perf_counter()-start)
  if delay>0:time.sleep(delay)
