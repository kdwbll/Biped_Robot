"""Read-only simulation diagnostics; no policy training or hardware control."""
from collections import Counter
from pathlib import Path
import json
import mujoco
import numpy as np
from supported_stepping_env import SupportedSteppingEnv
from stabilized_servo import StabilizedServo

ROOT=Path(__file__).parent

class MeasuredServo(StabilizedServo):
 def __init__(self,model,data):
  super().__init__(model,data);self.substeps=0;self.max_shaft=0.;self.max_motor=np.zeros(5);self.max_torque=np.zeros(5)
  self.ground_pen=0.;self.other_pen=0.;self.eq_error=0.;self.contact_names=Counter()
 def step(self):
  super().step();d=self.data;m=self.model;self.substeps+=1
  self.max_shaft=max(self.max_shaft,float(abs(d.qvel[-1])))
  self.max_motor=np.maximum(self.max_motor,np.abs(d.actuator_velocity))
  self.max_torque=np.maximum(self.max_torque,np.abs(d.ctrl))
  if d.nefc:
   mask=(d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id[:d.nefc]<16)
   if mask.any():self.eq_error=max(self.eq_error,float(np.max(np.abs(d.efc_pos[:d.nefc][mask]))))
  for c in d.contact[:d.ncon]:
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (c.geom1,c.geom2)]
   depth=max(0.,-float(c.dist))
   if 'ground' in names:self.ground_pen=max(self.ground_pen,depth)
   else:self.other_pen=max(self.other_pen,depth)
   if self.substeps%40==0:self.contact_names[' | '.join(sorted(names))]+=1

def command(case,k):
 phase=2*np.pi*k*.02
 if case=='neutral':return np.zeros(5)
 if case=='hip':return np.array([0,0,0,0,.5*np.sin(phase)])
 if case=='legs':return np.array([-.5*max(0,np.sin(phase)),.5*max(0,np.sin(phase)),.5*max(0,-np.sin(phase)),-.5*max(0,-np.sin(phase)),0])
 raise ValueError(case)

def run(case,weight,seed):
 e=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=3.75*weight,weight_support=weight)
 _,info=e.reset(seed=seed);servo=MeasuredServo(e.model,e.data);e.controller=servo
 peak_tilt=0.;start=e.data.xipos[e.body].copy();reason='horizon';first_nonfinite=None
 for k in range(200):
  _,_,term,trunc,info=e.step(command(case,k));up=info.get('upright',np.nan)
  if np.isfinite(up):peak_tilt=max(peak_tilt,float(np.degrees(np.arccos(np.clip(up,-1,1)))))
  if not np.isfinite(e.data.qpos).all() or not np.isfinite(e.data.qvel).all():first_nonfinite=k;reason='nonfinite';break
  if term or trunc:reason=info.get('reason','terminated' if term else 'horizon');break
 row=dict(case=case,seed=seed,weight_support=weight,balance_gain=3.75*weight,
  duration_s=(k+1)*.02,reason=reason,max_tilt_deg=peak_tilt,displacement_m=(e.data.xipos[e.body]-start).tolist(),
  max_shaft_speed_rad_s=servo.max_shaft,max_actuator_velocity_rad_s=servo.max_motor.tolist(),max_torque_nm=servo.max_torque.tolist(),
  max_equality_component_error_mm=servo.eq_error*1000,max_ground_penetration_mm=servo.ground_pen*1000,
  max_other_penetration_mm=servo.other_pen*1000,nonfinite=first_nonfinite is not None,
  sampled_contact_pairs=dict(servo.contact_names.most_common(8)))
 e.close();print(json.dumps(row,ensure_ascii=False),flush=True);return row

if __name__=='__main__':
 rows=[run(case,weight,seed) for weight in (0.,.35) for case in ('neutral','hip','legs') for seed in (17,29)]
 (ROOT/'simulation_verification_diagnostics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf8')
