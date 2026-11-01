"""Replay latest 50%-supported checkpoint; training runs in separate processes."""
import json,time
from pathlib import Path
import torch
import mujoco.viewer
from stable_baselines3 import PPO
from residual_gait_env import ResidualGaitEnv

ROOT=Path(__file__).parent
torch.set_num_threads(1)
def newest():
    checkpoints=list((ROOT/'training_checkpoints').glob('partial50b_gpu_*_steps.zip'))
    checkpoints+=list((ROOT/'training_checkpoints').glob('focusleft_gpu_*_steps.zip'))
    checkpoints+=list((ROOT/'training_checkpoints').glob('partial40_gpu_*_steps.zip'))
    checkpoints+=list((ROOT/'training_checkpoints').glob('fast_cpu_*_steps.zip'))
    if (ROOT/'continuous_current.zip').exists():checkpoints.append(ROOT/'continuous_current.zip')
    return max(checkpoints,key=lambda p:p.stat().st_mtime) if checkpoints else ROOT/'partial50_gpu_81920.zip'

def support_for(path):
    if path.name=='continuous_current.zip':
        try:return float(json.loads((ROOT/'continuous_state.json').read_text(encoding='utf8'))['weight'])
        except (OSError,ValueError,KeyError):return .35
    if path.name.startswith('fast_cpu_'):
        try:return float(json.loads((ROOT/'fast_curriculum_state.json').read_text(encoding='utf8'))['weight'])
        except (OSError,ValueError,KeyError):return .4
    return .4 if path.name.startswith('partial40_') else .5

path=newest(); model=PPO.load(str(path),device='cpu');path_mtime=path.stat().st_mtime_ns
weight=support_for(path)
env=ResidualGaitEnv(episode_seconds=4.,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
obs,_=env.reset(seed=9001);last_check=0.
print('LIVE CHECKPOINT REPLAY -- SIMULATION ONLY',flush=True)
print(f'{weight:.0%} body weight support and virtual balance are enabled.',flush=True)
print('Viewer runs separately from training; latest saved checkpoint is loaded periodically.',flush=True)
print('Current checkpoint:',path.name,flush=True)
with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
    while viewer.is_running():
        start=time.monotonic()
        if start-last_check>3.:
            last_check=start;new_path=newest()
            if new_path != path or new_path.stat().st_mtime_ns != path_mtime:
                try:
                    candidate=PPO.load(str(new_path),device='cpu')
                    model=candidate;path=new_path;path_mtime=path.stat().st_mtime_ns
                    weight=support_for(path)
                    env.set_support(3.75*weight);env.set_weight_support(weight);obs,_=env.reset(seed=9001)
                    print(f'Loaded checkpoint: {path.name}; virtual weight support: {weight:.0%}',flush=True)
                except (OSError,EOFError,ValueError):
                    pass # A checkpoint might still be writing; retry next cycle.
        action,_=model.predict(obs,deterministic=True)
        obs,_,term,trunc,_=env.step(action);viewer.sync()
        if term or trunc:obs,_=env.reset(seed=9001)
        time.sleep(max(0.,.02-(time.monotonic()-start)))
env.close()
