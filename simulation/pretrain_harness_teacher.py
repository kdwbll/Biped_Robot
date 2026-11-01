"""Imitate an assisted alternating demonstration; never intended for hardware."""
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
from supported_stepping_env import SupportedSteppingEnv

ROOT=Path(__file__).parent
def action_for(k):
    s=np.sin(2*np.pi*k*.02);left=max(0.,s);right=max(0.,-s)
    return np.array([-left,left,right,-right,0.],dtype=np.float32)

if __name__=='__main__':
    torch.set_num_threads(1)
    model=PPO.load(str(ROOT/'training_checkpoints/continued_gpu_49152_steps.zip'),device='cuda')
    obs_rows=[];action_rows=[]
    for seed in range(6):
        env=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3.,weight_support=.8)
        obs,_=env.reset(seed=seed)
        for k in range(200):
            act=action_for(k);obs_rows.append(obs);action_rows.append(act)
            obs,_,term,trunc,_=env.step(act)
            if term or trunc:break
        env.close()
        print('collected',len(obs_rows),'samples',flush=True)
    obs=torch.as_tensor(np.asarray(obs_rows),device=model.device)
    target=torch.as_tensor(np.asarray(action_rows),device=model.device)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=3e-4)
    rng=np.random.default_rng(711)
    for epoch in range(80):
        for idx in np.array_split(rng.permutation(len(obs_rows)),max(1,len(obs_rows)//128)):
            prediction=model.policy.get_distribution(obs[idx]).distribution.mean
            loss=((prediction-target[idx])**2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch%20==0:print('epoch',epoch,'loss',float(loss),flush=True)
    path=ROOT/'assisted_imitation_start.zip';model.save(str(path));print('saved',path,flush=True)
