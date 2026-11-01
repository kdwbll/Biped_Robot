"""Evaluate neutral stance controller in simulation only; never trains a policy."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
gear_candidate = '--gear' in sys.argv
refined_candidate = '--refined' in sys.argv
post_fix = '--post-fix' in sys.argv
phase_candidate = '--phase' in sys.argv
phase25 = '--phase25' in sys.argv
hip8 = '--hip8' in sys.argv
long_test = '--long' in sys.argv
model_name = ('diagnostic_post_shaft_fix_gear_phase_25.xml' if phase25 else 'diagnostic_post_shaft_fix_gear_phase.xml' if phase_candidate else 'diagnostic_post_shaft_fix.xml' if post_fix else 'diagnostic_refined_tower.xml' if refined_candidate else
              'biped_standing_gear_candidate.xml' if gear_candidate else 'biped_fresh_mass_only_candidate.xml')
cases=[(0.,.00025,6.,False),(1.5,.00025,6.,False),(1.5,.000125,6.,False),
       (1.5,.00025,6.,True)]
if long_test:
 cases=[(1.5,.00025,20.,False),(1.5,.00025,20.,True)]
out=[]
for ki,dt,duration,push in cases:
 m=mujoco.MjModel.from_xml_path(str(p/model_name));m.opt.timestep=dt
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=ki,
       kp_by_servo=[4,4,4,4,8] if hip8 else None,
       ki_by_servo=[ki,ki,ki,ki,6] if hip8 else None)
 z0=float(d.xipos[body,2]);p0=d.xipos[body].copy();max_tilt=0.;max_speed=np.zeros(5);max_torque=np.zeros(5)
 max_eq=0.;max_ground=0.;max_other=0.;max_err=np.zeros(5);peak_integral=0.;max_drift=0.;max_self_force=0.
 for i in range(round(duration/dt)):
  d.xfrc_applied[body,:]=0
  if push and 3. <= d.time < 3.2:d.xfrc_applied[body,1]=1.0
  servo.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError((ki,dt,push,d.time))
  max_speed=np.maximum(max_speed,np.abs(d.actuator_velocity));max_torque=np.maximum(max_torque,np.abs(d.ctrl))
  max_err=np.maximum(max_err,np.abs(servo.target_positions-d.actuator_length));peak_integral=max(peak_integral,float(np.max(np.abs(servo.integral_torque))))
  max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1)))))
  max_drift=max(max_drift,float(np.linalg.norm((d.xipos[body]-p0)[:2])))
  equal=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (d.efc_id<16)
  if np.any(equal):max_eq=max(max_eq,float(np.max(np.abs(d.efc_pos[equal]))))
  for ci,c in enumerate(d.contact[:d.ncon]):
   if c.geom1==ground or c.geom2==ground:max_ground=max(max_ground,float(-c.dist))
   else:
    max_other=max(max_other,float(-c.dist))
    if i%20==0:
     f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);max_self_force=max(max_self_force,float(f[0]))
 mujoco.mj_forward(m,d)
 row={'model':model_name,'ki_nm_per_rad_s':ki,'kp_nm_per_rad':4.,'hip_kp_nm_per_rad':8 if hip8 else 4,
      'hip_ki_nm_per_rad_s':6 if hip8 else ki,'timestep_s':dt,'duration_s':float(d.time),
      'external_push':'1 N in world +y for 0.2 s starting at 3 s' if push else 'none',
      'initial_height_m':z0,'final_height_m':float(d.xipos[body,2]),
      'height_loss_mm':float((z0-d.xipos[body,2])*1000),
      'final_position_error_deg':np.rad2deg(servo.target_positions-d.actuator_length).tolist(),
      'max_position_error_deg':np.rad2deg(max_err).tolist(),
      'max_abs_torque_nm':max_torque.tolist(),'max_abs_speed_rad_s':max_speed.tolist(),
      'peak_integral_torque_nm':peak_integral,'max_tilt_deg':max_tilt,'max_horizontal_drift_mm':max_drift*1000,
      'max_leg_constraint_residual_mm':max_eq*1000,'max_ground_penetration_mm':max_ground*1000,
      'max_other_penetration_mm':max_other*1000,'sampled_max_self_contact_force_n':max_self_force}
 out.append(row)
 print('ki',ki,'dt',dt,'push',push,'sag mm',round(row['height_loss_mm'],3),
       'error deg',round(max(abs(x) for x in row['final_position_error_deg']),3),
       'tilt',round(max_tilt,2),'drift mm',round(max_drift*1000,2),
       'speed',round(max(max_speed),2),'eq mm',round(max_eq*1000,3),flush=True)
result_name=('standing_gear_phase_25_hip8' if hip8 else 'standing_gear_phase_25' if phase25 else 'standing_gear_phase' if phase_candidate else 'standing_post_shaft_fix' if post_fix else 'standing_refined_pi' if refined_candidate else 'standing_gear_pi' if gear_candidate else 'standing_pi')+('_long_audit.json' if long_test else '_candidate_audit.json')
(p/result_name).write_text(json.dumps(out,indent=2),encoding='utf8')
