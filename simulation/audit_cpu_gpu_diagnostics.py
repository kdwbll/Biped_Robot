"""Fixed-action numerical comparisons; no optimizer or policy training."""
import json
from pathlib import Path
import numpy as np
from residual_gait_env import ResidualGaitEnv
from stabilized_servo import StabilizedServo
from warp_vec_env import WarpVecEnv

class CPU(ResidualGaitEnv):
 def reset(self,*,seed=None,options=None):
  _,info=super().reset(seed=seed,options=options);self.controller=StabilizedServo(self.model,self.data);return self._observation(),info
 def step(self,action):
  obs,reward,term,trunc,info=super().step(action);leg=info['phase_swing']
  if info['foot_contacts'][1-leg]:reward+=(.25 if leg==0 else .1)*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
  return obs,reward,term,trunc,info

rows=[]
for weight in (0.,.35):
 for case in ('zero_residual','hip_residual'):
  seed=17;cpu=CPU(episode_seconds=4.,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
  gpu=WarpVecEnv(nworld=1,weight=weight,seconds=4.,seed=seed);cpu.reset(seed=seed);gpu.reset()
  mismatch=0;reward_err=[];obs_err=[];body_err=[];max_overflow=0;reason='completed'
  for k in range(150):
   action=np.zeros(5,dtype=np.float32)
   if case=='hip_residual':action[-1]=.5*np.sin(2*np.pi*k*.02)
   co,cr,ct,cu,ci=cpu.step(action);go,gr,gd,gi=gpu.step(action[None,:]);ginfo=gi[0]
   if gd[0]:go[0]=ginfo['terminal_observation']
   mismatch+=int(ci['foot_contacts']!=ginfo['foot_contacts']);reward_err.append(abs(cr-float(gr[0])))
   obs_err.append(float(np.max(np.abs(co-go[0]))));max_overflow=max(max_overflow,ginfo['overflow'])
   if not gd[0]:body_err.append(float(np.linalg.norm(cpu.data.xipos[cpu.body]-gpu._arrays()['body'][0])))
   if ct or cu or gd[0]:reason=f"CPU terminated={bool(ct or cu)}, GPU terminated={bool(gd[0])}";break
  row=dict(weight_support=weight,case=case,seed=seed,seconds=(k+1)*.02,reason=reason,
   contact_mismatch_frames=mismatch,frames=k+1,reward_mae=float(np.mean(reward_err)),
   obs_max_error=float(max(obs_err)),body_position_max_error_mm=max(body_err,default=0.)*1000,overflow=max_overflow)
  rows.append(row);cpu.close();gpu.close();print(json.dumps(row),flush=True)
(Path(__file__).parent/'cpu_gpu_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
