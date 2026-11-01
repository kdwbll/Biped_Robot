"""Explore small PPO corrections to a reference gait, with decaying harness."""
import json
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from residual_gait_env import ResidualGaitEnv

ROOT=Path(__file__).parent
STEPS=65536
def make():return ResidualGaitEnv()

def audit(model):
    rows=[]
    for seed in (9001,9002,9003,9004,9005):
        e=ResidualGaitEnv(support_gain=0.,weight_support=0.);obs,_=e.reset(seed=seed)
        x0=e.data.xipos[e.body,0];peak=np.zeros(2);frames=np.zeros(2,dtype=int);tilt=0.
        for k in range(200):
            a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
            peak=np.maximum(peak,info['foot_clearance_m'])
            for leg in (0,1):frames[leg]+=int(info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg])
            tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
            if term or trunc:break
        rows.append(dict(seed=seed,seconds=(k+1)*.02,fell=bool(term),forward_m=float(e.data.xipos[e.body,0]-x0),
                         peak_clearance_m=peak.tolist(),supported_frames=frames.tolist(),max_tilt_deg=tilt));e.close()
    return rows

class Curriculum(BaseCallback):
    def __init__(self):super().__init__();self.next_eval=16384
    def _on_step(self):
        factor=float(np.clip((1-self.num_timesteps/STEPS)/.75,0,1))
        self.training_env.env_method('set_support',3*factor)
        self.training_env.env_method('set_weight_support',.8*factor)
        if self.num_timesteps>=self.next_eval:
            rows=audit(self.model)
            result=dict(timesteps=self.num_timesteps,training_weight_support=.8*factor,
                        falls=sum(r['fell'] for r in rows),episodes=rows)
            (ROOT/f'residual_gpu_{self.num_timesteps}_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            print('UNASSISTED AUDIT '+json.dumps(result),flush=True);self.next_eval+=16384
        return True

if __name__=='__main__':
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);env=SubprocVecEnv([make for _ in range(4)],start_method='spawn')
    model=PPO('MlpPolicy',env,device='cuda',n_steps=256,batch_size=128,n_epochs=4,learning_rate=.0001,
              policy_kwargs=dict(log_std_init=-1.4),verbose=1,seed=812)
    ckpt=CheckpointCallback(save_freq=2048,save_path=str(ROOT/'training_checkpoints'),name_prefix='residual_gpu')
    model.learn(total_timesteps=STEPS,callback=CallbackList([Curriculum(),ckpt]))
    model.save(str(ROOT/'residual_gpu_65536.zip'));env.close();print('residual curriculum complete',flush=True)
