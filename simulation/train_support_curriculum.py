"""Simulation-only curriculum: decay virtual balance torque; audit without it."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from supported_stepping_env import SupportedSteppingEnv
from continue_step_training import evaluate

ROOT = Path(__file__).parent
START = 49152
DURATION = 65536

def make():
    return SupportedSteppingEnv(episode_seconds=4., disturbance=.02, support_gain=3.)

class Curriculum(BaseCallback):
    def __init__(self):
        super().__init__()
        self.next_eval = START + 16384
    def _on_step(self):
        progress = max(0., min(1., (self.num_timesteps - START) / DURATION))
        # Hold support for the first quarter, then fade gradually to zero.
        gain = 3. * max(0., min(1., (1. - progress) / .75))
        self.training_env.env_method('set_support', gain)
        if self.num_timesteps >= self.next_eval:
            rows = evaluate(self.model)  # SteppingEnv: no virtual force
            summary = dict(timesteps=self.num_timesteps, training_support_gain=gain,
                           mean_forward_m=float(np.mean([r['forward_m'] for r in rows])),
                           falls=sum(r['fell'] for r in rows),
                           max_tilt_deg=max(r['peak_tilt_deg'] for r in rows),
                           supported_swing_frames=sum(r['supported_swing_frames'] for r in rows),
                           episodes=rows)
            (ROOT / f'curriculum_gpu_{self.num_timesteps}_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            print('UNASSISTED AUDIT ' + json.dumps(summary), flush=True)
            self.next_eval += 16384
        return True

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    env = SubprocVecEnv([make for _ in range(4)], start_method='spawn')
    model = PPO.load(str(ROOT/'training_checkpoints/continued_gpu_49152_steps.zip'), env=env, device='cuda', learning_rate=.0001)
    print('training with temporary virtual support; audits without support', flush=True)
    checkpoint = CheckpointCallback(save_freq=2048, save_path=str(ROOT/'training_checkpoints'), name_prefix='curriculum_gpu')
    model.learn(total_timesteps=DURATION, reset_num_timesteps=False, callback=CallbackList([Curriculum(), checkpoint]))
    model.save(str(ROOT/'curriculum_gpu_114688.zip'))
    env.close()
    print('curriculum complete', flush=True)
