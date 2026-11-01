import numpy as np
from gymnasium.utils.env_checker import check_env
from alternating_biped_env import AlternatingBipedEnv
e=AlternatingBipedEnv(episode_seconds=.5);check_env(e,skip_render_check=True)
a,_=e.reset(seed=7)
for i in range(25):
 o,r,t,tr,info=e.step(np.zeros(5,dtype=np.float32))
 if t or tr:break
print('success',i+1,'support switches',info.get('support_switches'),'contact flags',info.get('foot_contacts'))
