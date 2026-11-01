"""View simulated imitation WITH virtual balance and 80% body-weight support."""
import time
import numpy as np
import torch
import mujoco.viewer
from stable_baselines3 import PPO
from supported_stepping_env import SupportedSteppingEnv

torch.set_num_threads(1)
model=PPO.load('assisted_imitation_start.zip',device='cpu')
env=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3.,weight_support=.8)
obs,_=env.reset(seed=9001)
print('SIMULATION ONLY: virtual balance torque and 80% body-weight support are enabled; auto-reset each 4 seconds.',flush=True)
with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
    while viewer.is_running():
        start=time.monotonic()
        action,_=model.predict(obs,deterministic=True)
        obs,_,term,trunc,_=env.step(action)
        viewer.sync()
        if term or trunc:obs,_=env.reset(seed=9001)
        time.sleep(max(0.,.02-(time.monotonic()-start)))
env.close()
