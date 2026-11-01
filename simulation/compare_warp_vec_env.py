"""Diagnostic CPU/GPU rollout parity; different precision may diverge."""
import json,numpy as np
from warp_vec_env import WarpVecEnv
from residual_gait_env import ResidualGaitEnv
from stabilized_servo import StabilizedServo

class CPU(ResidualGaitEnv):
 def reset(self,*,seed=None,options=None):
  _,info=super().reset(seed=seed,options=options);self.controller=StabilizedServo(self.model,self.data);return self._observation(),info
 def step(self,action):
  obs,reward,term,trunc,info=super().step(action);leg=info['phase_swing']
  if info['foot_contacts'][1-leg]:reward+=(.25 if leg==0 else .1)*float(np.clip(info['foot_clearance_m'][leg]/.006,0,1))
  return obs,reward,term,trunc,info

cpu=CPU(episode_seconds=4.,disturbance=.02,support_gain=1.875,weight_support=.5)
g=WarpVecEnv(nworld=1,weight=.5,seconds=4.,seed=123)
co,_=cpu.reset(seed=123);go=g.reset();rows=[]
for k in range(200):
 a=np.array([.1*np.sin(k*.1),0.,0.,0.,.1*np.cos(k*.1)],dtype=np.float32)
 co,cr,ct,cu,ci=cpu.step(a);go,gr,gd,gi=g.step(a[None,:])
 if gd[0]:go[0]=gi[0]['terminal_observation']
 if k in (0,4,24,49,99,199) or ct or cu or gd[0]:
  row=dict(seconds=(k+1)*.02,obs_max_error=float(np.max(np.abs(co-go[0]))),
   obs_difference=(co-go[0]).tolist(),
   cpu_obs_omega=co[3:6].tolist(),gpu_obs_omega=go[0,3:6].tolist(),
   cpu_recomputed_omega=cpu._observation()[3:6].tolist(),
   cpu_world_omega=cpu.data.cvel[cpu.body,:3].tolist(),gpu_world_omega=g.gd.cvel.numpy()[0,g.body,:3].tolist(),
   cpu_rot=cpu.data.xmat[cpu.body].tolist(),gpu_rot=g.gd.xmat.numpy()[0,g.body].tolist(),
   cpu_qvel_max=float(np.max(np.abs(cpu.data.qvel))),gpu_qvel_max=float(np.max(np.abs(g.gd.qvel.numpy()[0]))),
   cpu_reward=float(cr),gpu_reward=float(gr[0]),cpu_contacts=ci['foot_contacts'],gpu_contacts=gi[0]['foot_contacts'],
   cpu_clearance=ci['foot_clearance_m'],gpu_clearance=gi[0]['foot_clearance_m'],cpu_done=bool(ct or cu),gpu_done=bool(gd[0]))
  rows.append(row);print(json.dumps(row),flush=True)
 if ct or cu or gd[0]:break
cpu.close();g.close()
open('warp_vec_parity.json','w',encoding='utf8').write(json.dumps(rows,indent=2))
