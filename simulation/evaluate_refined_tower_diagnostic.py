"""Measure candidate's forces and neutral stance; this does not validate coverage."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_refined_tower.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');z0=d.xipos[body,2]
max_normal=0.;max_pair=None;max_tilt=0.;max_speed=np.zeros(5)
for step in range(round(6/m.opt.timestep)):
 servo.step();max_speed=np.maximum(max_speed,np.abs(d.actuator_velocity))
 max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1)))))
 if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('non-finite state')
 if step%20==0:
  for i,c in enumerate(d.contact[:d.ncon]):
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
   if 'ground' in names:continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
   if f[0]>max_normal:max_normal=float(f[0]);max_pair=names
mujoco.mj_forward(m,d)
final=[]
for i,c in enumerate(d.contact[:d.ncon]):
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
 if 'ground' in names:continue
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
 if f[0]>.1:final.append({'pair':names,'normal_force_n':float(f[0]),'penetration_mm':float(-c.dist*1000)})
out={'seconds':d.time,'height_loss_mm':float((z0-d.xipos[body,2])*1000),'max_tilt_deg':max_tilt,
     'final_servo_error_deg':np.rad2deg(servo.target_positions-d.actuator_length).tolist(),
     'max_servo_speed_rad_s':max_speed.tolist(),'sampled_max_self_contact_n':max_normal,
     'sampled_max_self_contact_pair':max_pair,'final_self_contacts':final,
     'warning':'Diagnostic approximation; collision pairs excluded by previous model remain unresolved.'}
(p/'refined_tower_6s_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('height loss mm',round(out['height_loss_mm'],3),'fifth error deg',round(out['final_servo_error_deg'][4],3),
      'sampled peak self contact N',round(max_normal,2),'pair',max_pair,
      'final max self contact N',round(max([x['normal_force_n'] for x in final]+[0]),2))
