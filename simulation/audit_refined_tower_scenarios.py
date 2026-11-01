"""Exercise refined collision diagnostic in fixed, low-amplitude scenarios; no training."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
phase='--phase' in sys.argv
phase25='--phase25' in sys.argv
cases=[(.00025,'neutral'),(.000125,'neutral'),(.00025,'push_plus'),(.00025,'push_minus'),(.00025,'hip_command')]
rows=[]
for dt,case in cases:
 m=mujoco.MjModel.from_xml_path(str(p/('diagnostic_post_shaft_fix_gear_phase_25.xml' if phase25 else 'diagnostic_post_shaft_fix_gear_phase.xml' if phase else 'diagnostic_refined_tower.xml')));m.opt.timestep=dt
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 z0=d.xipos[body,2];p0=d.xipos[body].copy();peak_force=0.;pair=None;max_speed=np.zeros(5);max_torque=np.zeros(5);max_tilt=0.;max_eq=0.;max_ground=0.
 for step in range(round(4/dt)):
  d.xfrc_applied[body,:]=0
  if 2.<=d.time<2.2 and case.startswith('push'):d.xfrc_applied[body,1]=1. if case=='push_plus' else -1.
  if case=='hip_command' and d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1510])
  servo.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError((case,dt,d.time))
  max_speed=np.maximum(max_speed,np.abs(d.actuator_velocity));max_torque=np.maximum(max_torque,np.abs(d.ctrl))
  max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1)))))
  eq=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id<16)
  if np.any(eq):max_eq=max(max_eq,float(np.max(np.abs(d.efc_pos[eq]))))
  for i,c in enumerate(d.contact[:d.ncon]):
   if c.geom1==ground or c.geom2==ground:
    max_ground=max(max_ground,float(-c.dist));continue
   if step%20:continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
   if f[0]>peak_force:
    peak_force=float(f[0]);pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
 mujoco.mj_forward(m,d)
 row={'case':case,'timestep_s':dt,'duration_s':d.time,'height_loss_mm':float((z0-d.xipos[body,2])*1000),
      'final_horizontal_drift_mm':float(np.linalg.norm((d.xipos[body]-p0)[:2])*1000),
      'max_tilt_deg':max_tilt,'max_leg_constraint_residual_mm':max_eq*1000,'max_ground_penetration_mm':max_ground*1000,
      'max_servo_speed_rad_s':max_speed.tolist(),'max_servo_torque_nm':max_torque.tolist(),
      'sampled_max_self_contact_n':peak_force,'sampled_max_self_contact_pair':pair,
      'final_servo_error_deg':np.rad2deg(servo.target_positions-d.actuator_length).tolist()}
 rows.append(row)
 print(case,'dt',dt,'sag mm',round(row['height_loss_mm'],3),'tilt',round(max_tilt,2),
       'max self contact N',round(peak_force,2),'pair',pair,'hip error deg',round(row['final_servo_error_deg'][4],3),flush=True)
(p/('gear_phase_25_scenarios_audit.json' if phase25 else 'gear_phase_scenarios_audit.json' if phase else 'refined_tower_scenarios_audit.json')).write_text(json.dumps(rows,indent=2),encoding='utf8')
