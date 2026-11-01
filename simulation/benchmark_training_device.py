"""Measure PPO training throughput without changing existing checkpoints."""
import json,sys,time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv
def make():return ResidualGaitEnv(support_gain=1.5,weight_support=.4)
if __name__=='__main__':
 torch.set_num_threads(1);device=sys.argv[1]
 env=SubprocVecEnv([make for _ in range(8)],start_method='spawn')
 model=PPO.load('focusleft_gpu_131072.zip',env=env,device=device,learning_rate=.0001,verbose=0)
 start=time.perf_counter();model.learn(total_timesteps=4096,reset_num_timesteps=False,progress_bar=False)
 elapsed=time.perf_counter()-start;env.close()
 print(json.dumps(dict(device=device,workers=8,steps=4096,seconds=elapsed,env_steps_per_second=4096/elapsed)),flush=True)
