"""Observe substep peaks; stiffness is a hypothesis, not a hardware setting."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from stabilized_servo import StabilizedServo
from theoretical_servo import ServoSpec
from supported_stepping_env import SupportedSteppingEnv

class Instrumented(StabilizedServo):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.peak_qvel=0.;self.peak_shaft=0.;self.peak_torque=0.;self.peak_ctrl=0
 def step(self):
  super().step();self.peak_qvel=max(self.peak_qvel,float(np.max(np.abs(self.data.qvel))))
  self.peak_shaft=max(self.peak_shaft,abs(float(self.data.qvel[-1])))
  self.peak_torque=max(self.peak_torque,float(np.max(np.abs(self.data.ctrl))))
  self.peak_ctrl+=int(np.any(np.abs(self.data.ctrl)>.99*self.spec.stall_torque_nm))

rows=[]
for kp in (1.,2.,4.,8.):
 for dt in (.0005,.00025):
  e=SupportedSteppingEnv(episode_seconds=2.,disturbance=0.,support_gain=0.,weight_support=0.)
  e.model.opt.timestep=dt;e.steps_per_control=round(.02/dt);e.reset(seed=1)
  e.controller=Instrumented(e.model,e.data,replace(ServoSpec(),kp_nm_per_rad=kp));z=float(e.data.xipos[e.body,2])
  for k in range(100):
   _,_,term,trunc,info=e.step(np.zeros(5))
   if term or trunc:break
  c=e.controller;row=dict(kp=kp,timestep=dt,seconds=(k+1)*.02,height_loss_mm=(z-float(e.data.xipos[e.body,2]))*1000,
   peak_qvel_rad_s=c.peak_qvel,peak_shaft_rad_s=c.peak_shaft,peak_torque_nm=c.peak_torque,saturated_substeps=c.peak_ctrl,terminated=bool(term))
  rows.append(row);e.close();print(json.dumps(row),flush=True)
(Path(__file__).parent/'timestep_sensitivity_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
