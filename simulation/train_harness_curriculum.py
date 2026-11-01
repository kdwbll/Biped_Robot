"""Train with a virtual harness that fades to zero; evaluate unsupported."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from supported_stepping_env import SupportedSteppingEnv
from continue_step_training import evaluate

ROOT=Path(__file__).parent
START=49152
DURATION=98304
def make():
    return SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3.,weight_support=.8)

class HarnessCurriculum(BaseCallback):
    def __init__(self,prefix='harness'):
        super().__init__(); self.next_eval=START+16384; self.prefix=prefix
    def _on_step(self):
        progress=np.clip((self.num_timesteps-START)/DURATION,0,1)
        factor=float(np.clip((1-progress)/.75,0,1))
        # Support is training-only; the fixed audit environment is unassisted.
        self.training_env.env_method('set_support',3*factor)
        self.training_env.env_method('set_weight_support',.8*factor)
        if self.num_timesteps>=self.next_eval:
            rows=evaluate(self.model)
            summary=dict(timesteps=self.num_timesteps,training_weight_support=.8*factor,
                training_balance_gain=3*factor,mean_forward_m=float(np.mean([r['forward_m'] for r in rows])),
                falls=sum(r['fell'] for r in rows),max_tilt_deg=max(r['peak_tilt_deg'] for r in rows),
                supported_swing_frames=sum(r['supported_swing_frames'] for r in rows),episodes=rows)
            (ROOT/f'{self.prefix}_gpu_{self.num_timesteps}_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
            print('UNASSISTED AUDIT '+json.dumps(summary),flush=True); self.next_eval+=16384
        return True

if __name__=='__main__':
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1); env=SubprocVecEnv([make for _ in range(4)],start_method='spawn')
    imitation=len(sys.argv)>1 and sys.argv[1]=='imitation'
    prefix='imitation' if imitation else 'harness'
    source=ROOT/('assisted_imitation_start.zip' if imitation else 'training_checkpoints/continued_gpu_49152_steps.zip')
    model=PPO.load(str(source),env=env,device='cuda',learning_rate=.0001)
    print('training with decaying virtual harness; unsupported audits',flush=True)
    ckpt=CheckpointCallback(save_freq=2048,save_path=str(ROOT/'training_checkpoints'),name_prefix=f'{prefix}_gpu')
    model.learn(total_timesteps=DURATION,reset_num_timesteps=False,callback=CallbackList([HarnessCurriculum(prefix),ckpt]))
    model.save(str(ROOT/f'{prefix}_gpu_147456.zip'));env.close();print('harness curriculum complete',flush=True)
