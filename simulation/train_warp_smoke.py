"""Short experimental PPO run. Not an approved or deployable walking policy."""
import json,time,numpy as np,torch
from stable_baselines3 import PPO
from warp_vec_env import WarpVecEnv
torch.set_num_threads(1);e=WarpVecEnv(nworld=32,weight=.5,seconds=1.,seed=42)
model=PPO('MlpPolicy',e,device='cuda',n_steps=64,batch_size=256,n_epochs=2,learning_rate=.0001,verbose=0,seed=42)
start=time.perf_counter();model.learn(total_timesteps=4096);elapsed=time.perf_counter()-start
model.save('warp_ppo_smoke.zip');e.close()
print('RESULT '+json.dumps(dict(envs=32,steps=4096,seconds=elapsed,env_steps_per_second=4096/elapsed,
 note='experimental PPO; float32 GPU physics; not validated for hardware')),flush=True)
