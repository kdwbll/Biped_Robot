"""View a simulated PPO checkpoint, never connect to robot hardware."""
import argparse,time
from pathlib import Path
import mujoco.viewer
from stable_baselines3 import PPO
from biped_env import BipedEnv
p=argparse.ArgumentParser();p.add_argument('checkpoint',nargs='?',default='ppo_experiment_4096.zip');args=p.parse_args()
env=BipedEnv(episode_seconds=8);model=PPO.load(str(Path(__file__).parent/args.checkpoint),device='cpu');obs,_=env.reset()
with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
 while viewer.is_running():
  start=time.perf_counter();action,_=model.predict(obs,deterministic=True);obs,reward,done,trunc,info=env.step(action)
  if done or trunc:obs,_=env.reset()
  viewer.sync();remaining=.02-(time.perf_counter()-start)
  if remaining>0:time.sleep(remaining)
