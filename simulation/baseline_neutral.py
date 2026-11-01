import numpy as np
from biped_env import BipedEnv
e=BipedEnv();obs,_=e.reset(seed=123);total=0.;distance=0
for i in range(400):
 obs,r,done,trunc,info=e.step(np.zeros(5,dtype=np.float32));total+=r;distance+=info['x_displacement']
 if done or trunc: break
print('neutral seconds',round((i+1)*.02,2),'done',done,'upright',round(info['upright'],3),'height',round(info['height'],3),'distance m',round(distance,3),'reward',round(total,3))
