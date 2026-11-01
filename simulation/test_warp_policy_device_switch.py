"""Validate CPU audit inference followed by resumed CUDA PPO updates."""
import numpy as np,torch
from stable_baselines3 import PPO
from warp_vec_env import WarpVecEnv
torch.set_num_threads(1);env=WarpVecEnv(nworld=2,weight=.5,seconds=1.)
model=PPO.load('warp_training_current.zip',env=env,device='cuda',learning_rate=.0001)
obs=env.reset();model.policy.to('cpu');action,_=model.predict(obs,deterministic=True)
assert np.isfinite(action).all() and model.policy.device.type=='cpu'
model.policy.to('cuda');assert model.policy.device.type=='cuda'
# Only test one update's gradient and optimizer state against the GPU policy.
obs_t=torch.as_tensor(obs,device='cuda');actions_t=torch.as_tensor(action,device='cuda')
values,log_prob,entropy=model.policy.evaluate_actions(obs_t,actions_t)
loss=(values.square().mean()-log_prob.mean()*.001);model.policy.optimizer.zero_grad();loss.backward();model.policy.optimizer.step()
print('DEVICE_SWITCH_OK',model.policy.device,float(loss.detach().cpu()),flush=True)
env.close()
