"""Isolate unverified hip limits and soft gearing in neutral standing; no training."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
out=[]
for limits,gear_time in [(True,.005),(False,.005),(True,.001),(False,.001)]:
 m=mujoco.MjModel.from_xml_path(str(p/'biped_fresh_mass_only_candidate.xml'));m.opt.timestep=.00025
 joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ['L0_hip_candidate','L1_hip_candidate']]
 if not limits:
  for j in joints:m.jnt_limited[j]=0
 eqids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,n) for n in
        ['opposite_world_rotation_assumption','estimated_pinion_23_gear_50']]
 for i in eqids:m.eq_solref[i,0]=gear_time
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');z0=d.xipos[body,2]
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
 eqpeak=np.zeros(2);over=np.zeros(2);speed=np.zeros(5)
 for _ in range(round(4/m.opt.timestep)):
  servo.step();speed=np.maximum(speed,np.abs(d.actuator_velocity))
  if not np.isfinite(d.qpos).all():raise RuntimeError((limits,gear_time,d.time))
  for k,j in enumerate(joints):
   q=d.qpos[m.jnt_qposadr[j]];lo,hi=m.jnt_range[j];over[k]=max(over[k],lo-q,q-hi)
  for k,eqid in enumerate(eqids):
   match=np.flatnonzero((d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id==eqid))
   if len(match):eqpeak[k]=max(eqpeak[k],abs(d.efc_pos[match[0]]))
 mujoco.mj_forward(m,d)
 row={'limits_enabled':limits,'gear_constraint_time_constant_s':gear_time,'seconds':d.time,
      'height_loss_mm':float((z0-d.xipos[body,2])*1000),
      'max_excess_relative_to_unverified_range_deg':np.rad2deg(over).tolist(),
      'max_gear_constraint_error_deg':np.rad2deg(eqpeak).tolist(),
      'max_servo_speed_rad_s':speed.tolist()}
 out.append(row)
 print('limits',limits,'gear solref',gear_time,'sag',round(row['height_loss_mm'],3),
       'excess deg',np.round(row['max_excess_relative_to_unverified_range_deg'],3),
       'gear residual deg',np.round(row['max_gear_constraint_error_deg'],3),
       'peak speed',round(max(speed),2),flush=True)
(p/'standing_gear_constraints_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf8')
