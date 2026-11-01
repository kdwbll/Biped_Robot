"""Continue PPO using GPU for the policy and CPU for MuJoCo; simulation only."""
import json,time
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from stepping_env import SteppingEnv
ROOT=Path(__file__).parent
SEEDS=(9001,9002,9003,9004,9005)
def make():return SteppingEnv(episode_seconds=4.0,disturbance=.02)
def evaluate(model):
 rows=[]
 for seed in SEEDS:
  e=make();obs,_=e.reset(seed=seed);x0=float(e.data.xipos[e.body,0]);best=np.zeros(2);swing=0;peak=0.
  for k in range(200):
   a,_=model.predict(obs,deterministic=True);obs,r,term,trunc,info=e.step(a)
   best=np.maximum(best,info.get('foot_clearance_m',[0,0]));swing+=int(info.get('valid_swing',False));peak=max(peak,float(np.degrees(np.arccos(np.clip(info.get('upright',0),-1,1)))))
   if term or trunc:break
  rows.append({'seed':seed,'seconds':(k+1)*.02,'forward_m':float(e.data.xipos[e.body,0]-x0),'peak_clearance_m':best.tolist(),'supported_swing_frames':swing,'peak_tilt_deg':peak,'fell':bool(term)});e.close()
 return rows
class Audit(BaseCallback):
 def __init__(self):super().__init__();self.next_eval=16384
 def _on_step(self):
  if self.num_timesteps>=self.next_eval:
   rows=evaluate(self.model);summary={'timesteps':self.num_timesteps,'mean_forward_m':float(np.mean([r['forward_m'] for r in rows])),'falls':sum(r['fell'] for r in rows),'max_tilt_deg':max(r['peak_tilt_deg'] for r in rows),'supported_swing_frames':sum(r['supported_swing_frames'] for r in rows),'episodes':rows};(ROOT/f'stepping_gpu_{self.num_timesteps}_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print('AUDIT '+json.dumps(summary),flush=True);self.next_eval+=16384
  return True
if __name__=='__main__':
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(1);env=SubprocVecEnv([make for _ in range(4)],start_method='spawn');model=PPO.load(str(ROOT/'training_checkpoints/continued_gpu_49152_steps.zip'),env=env,device='cuda',learning_rate=.0001);print('starting GPU supported-step continuation, CUDA',torch.cuda.get_device_name(0),flush=True)
 # num_timesteps starts at 32768; audit next after 49152.
 audit=Audit();audit.next_eval=65536
 checkpoint=CheckpointCallback(save_freq=2048,save_path=str(ROOT/'training_checkpoints'),name_prefix='stepping_gpu')
 model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=CallbackList([audit,checkpoint]));out=ROOT/'stepping_gpu_114688.zip';model.save(str(out));env.close();print('saved',out,flush=True)

