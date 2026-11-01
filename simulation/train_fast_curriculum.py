"""Eight-worker PPO with automatic, conservative reductions of virtual support."""
import json
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback,CallbackList,CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_gait_env import ResidualGaitEnv
ROOT=Path(__file__).parent
START=163840
DURATION=131072

class FastEnv(ResidualGaitEnv):
 def __init__(self):super().__init__(support_gain=1.5,weight_support=.4)
 def step(self,action):
  obs,reward,term,trunc,info=super().step(action)
  leg=info.get('phase_swing',0)
  if info.get('foot_contacts',[False,False])[1-leg]:
   reward+=(.25 if leg==0 else .10)*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
  return obs,reward,term,trunc,info
def make():return FastEnv()

def evaluate(model,weight,episode_seconds=4.,seeds=(9101,9102,9103)):
 rows=[]
 for seed in seeds:
  e=ResidualGaitEnv(episode_seconds=episode_seconds,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
  obs,_=e.reset(seed=seed);x0=e.data.xipos[e.body,0];run=[0,0];order=[];peak=[0.,0.]
  for k in range(round(episode_seconds/.02)):
   a,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(a)
   for leg in (0,1):
    peak[leg]=max(peak[leg],info['foot_clearance_m'][leg]);good=info['foot_clearance_m'][leg]>.003 and info['foot_contacts'][1-leg]
    run[leg]=run[leg]+1 if good else 0
    if run[leg]==3:order.append(leg)
   if term or trunc:break
  alternating=sum(order[i]!=order[i-1] for i in range(1,len(order)))
  rows.append(dict(seed=seed,seconds=(k+1)*.02,fell=bool(term),forward_m=float(e.data.xipos[e.body,0]-x0),peak_clearance_m=peak,events=order,alternations=alternating));e.close()
 return rows

class Curriculum(BaseCallback):
 def __init__(self):super().__init__();self.next_eval=START+32768;self.weight=.4
 def _on_step(self):
  if self.num_timesteps>=self.next_eval:
   assisted=evaluate(self.model,self.weight);unassisted=evaluate(self.model,0.)
   passed=all(not r['fell'] and r['forward_m']>0 and r['alternations']>=3 for r in assisted)
   evaluated=self.weight
   if passed:self.weight=max(0.,round(self.weight-.05,2))
   self.training_env.env_method('set_support',3.75*self.weight)
   self.training_env.env_method('set_weight_support',self.weight)
   summary=dict(timesteps=self.num_timesteps,evaluated_weight=evaluated,next_training_weight=self.weight,
      passed_gate=passed,assisted=assisted,unassisted=unassisted)
   (ROOT/f'fast_cpu_{self.num_timesteps}_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
   (ROOT/'fast_curriculum_state.json').write_text(json.dumps(dict(weight=self.weight,timesteps=self.num_timesteps),indent=2),encoding='utf8')
   print('CURRICULUM AUDIT '+json.dumps(summary),flush=True);self.next_eval+=32768
  return True

if __name__=='__main__':
 torch.set_num_threads(1);env=SubprocVecEnv([make for _ in range(8)],start_method='spawn')
 model=PPO.load(str(ROOT/'partial40_gpu_163840.zip'),env=env,device='cpu',learning_rate=.0001)
 (ROOT/'fast_curriculum_state.json').write_text(json.dumps(dict(weight=.4,timesteps=START)),encoding='utf8')
 ckpt=CheckpointCallback(save_freq=1024,save_path=str(ROOT/'training_checkpoints'),name_prefix='fast_cpu')
 print('Fast CPU curriculum: 8 environments; support decreases only after gait checks pass.',flush=True)
 model.learn(total_timesteps=DURATION,reset_num_timesteps=False,callback=CallbackList([Curriculum(),ckpt]))
 model.save(str(ROOT/'fast_cpu_294912.zip'));env.close();print('fast curriculum complete',flush=True)
