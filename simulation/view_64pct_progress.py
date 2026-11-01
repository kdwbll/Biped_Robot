"""Display the 64%-weight-supported residual policy. Simulation only."""
import time
import torch
import mujoco.viewer
from stable_baselines3 import PPO
from residual_gait_env import ResidualGaitEnv

torch.set_num_threads(1)
model=PPO.load('training_checkpoints/partial_gpu_49152_steps.zip',device='cpu')
env=ResidualGaitEnv(support_gain=2.4,weight_support=.64)
obs,_=env.reset(seed=9001)
print('VIRTUAL ASSISTANCE ENABLED: supports 64% body weight plus artificial balance torque.',flush=True)
print('Not autonomous walking; the simulation resets every 4 seconds.',flush=True)
with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
    while viewer.is_running():
        start=time.monotonic();action,_=model.predict(obs,deterministic=True)
        obs,_,term,trunc,_=env.step(action);viewer.sync()
        if term or trunc:obs,_=env.reset(seed=9001)
        time.sleep(max(0.,.02-(time.monotonic()-start)))
env.close()
