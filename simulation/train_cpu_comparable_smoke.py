"""CPU PPO short comparison; matching 4096 samples and optimization epochs."""
import json,time,numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv
from stabilized_servo import StabilizedServo

class BenchEnv(ResidualGaitEnv):
 def __init__(self):super().__init__(episode_seconds=1.,disturbance=.02,support_gain=1.875,weight_support=.5)
 def reset(self,*,seed=None,options=None):
  _,info=super().reset(seed=seed,options=options);self.controller=StabilizedServo(self.model,self.data);return self._observation(),info
 def step(self,action):
  obs,reward,term,trunc,info=super().step(action);leg=info['phase_swing']
  if info['foot_contacts'][1-leg]:reward+=(.25 if leg==0 else .1)*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
  return obs,reward,term,trunc,info
def make():return BenchEnv()
if __name__=='__main__':
 torch.set_num_threads(1);e=SubprocVecEnv([make for _ in range(8)],start_method='spawn')
 model=PPO('MlpPolicy',e,device='cpu',n_steps=256,batch_size=256,n_epochs=2,learning_rate=.0001,verbose=0,seed=42)
 start=time.perf_counter();model.learn(total_timesteps=4096);elapsed=time.perf_counter()-start;e.close()
 print('RESULT '+json.dumps(dict(envs=8,steps=4096,seconds=elapsed,env_steps_per_second=4096/elapsed,n_epochs=2)),flush=True)
