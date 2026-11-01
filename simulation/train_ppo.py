"""Exploratory PPO in MuJoCo; no real-world deployment or accuracy guarantee."""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from biped_env import BipedEnv

def evaluate(policy, episodes=3):
    results=[]
    for seed in range(episodes):
        env=BipedEnv(episode_seconds=4); obs,_=env.reset(seed=seed);distance=0.;reward=0.
        for step in range(env.horizon):
            act=np.zeros(5,dtype=np.float32) if policy is None else policy.predict(obs,deterministic=True)[0]
            obs,r,terminated,truncated,info=env.step(act);distance+=info['x_displacement'];reward+=r
            if terminated or truncated: break
        results.append((step+1,distance,reward,terminated))
        env.close()
    return results

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timesteps',type=int,default=10000);parser.add_argument('--output',default='ppo_experiment');args=parser.parse_args()
    if args.timesteps < 256:parser.error('timesteps must be at least 256')
    torch.set_num_threads(1)
    env=BipedEnv(episode_seconds=4)
    model=PPO('MlpPolicy',env,n_steps=256,batch_size=64,n_epochs=4,learning_rate=0.0003,seed=42,device='cpu',verbose=1)
    model.learn(total_timesteps=args.timesteps)
    out=Path(__file__).parent/args.output;model.save(str(out));print('saved:',out.with_suffix('.zip'))
    print('neutral evaluation (steps, displacement m, reward, fell):',evaluate(None))
    print('policy evaluation (steps, displacement m, reward, fell):',evaluate(model))
    env.close()
if __name__=='__main__':main()
