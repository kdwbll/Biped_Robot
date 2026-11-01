"""Separate GPU PPO trial; validate on CPU before any production use."""
import json,time,torch
from stable_baselines3 import PPO
from warp_vec_env import WarpVecEnv
from train_stabilized_cpu import evaluate
torch.set_num_threads(1);e=WarpVecEnv(nworld=32,weight=.5,seconds=12.,seed=318)
model=PPO('MlpPolicy',e,device='cuda',n_steps=128,batch_size=256,n_epochs=4,learning_rate=.0001,
 policy_kwargs=dict(log_std_init=-1.4),verbose=1,seed=318)
start=time.perf_counter();model.learn(total_timesteps=32768);elapsed=time.perf_counter()-start
model.save('warp_trial_32768.zip');e.close()
assisted=evaluate(model,.5);unassisted=evaluate(model,0.)
result=dict(envs=32,steps=32768,train_seconds=elapsed,train_env_steps_per_second=32768/elapsed,
 cpu_stabilized_assisted=assisted,cpu_stabilized_unassisted=unassisted,
 note='experimental numerical simulator; no physical deployment')
open('warp_trial_32768_audit.json','w',encoding='utf8').write(json.dumps(result,indent=2))
print('RESULT '+json.dumps(result),flush=True)
