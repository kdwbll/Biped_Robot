"""Measure fifth-servo bias, unverified hip limits and gearing during stance."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'biped_standing_gear_candidate.xml'))
no_limits='--no-hip-limits' in sys.argv
if no_limits:
 for name in ['L0_hip_candidate','L1_hip_candidate']:
  m.jnt_limited[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)]=0
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ['L0_hip_candidate','L1_hip_candidate','hypothetical_servo_shaft']]
eqids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,n) for n in ['opposite_world_rotation_assumption','estimated_pinion_23_gear_50']]
rows=[]
for t in (2,6,20):
 while d.time < t-m.opt.timestep/2:servo.step()
 mujoco.mj_forward(m,d)
 res=[]
 for eqid in eqids:
  i=np.flatnonzero((d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id==eqid))[0]
  res.append(float(np.rad2deg(d.efc_pos[i])))
 row={'time_s':float(d.time),'servo_error_deg':np.rad2deg(servo.target_positions-d.actuator_length).tolist(),
      'servo_torque_nm':d.ctrl.tolist(),'servo_integral_nm':servo.integral_torque.tolist(),
      'hip_and_shaft_angles_deg':[float(np.rad2deg(d.qpos[m.jnt_qposadr[j]])) for j in joints],
      'gear_residual_deg':res}
 rows.append(row)
 print('time',t,'hip and shaft deg',np.round(row['hip_and_shaft_angles_deg'],3),
       'servo error deg',np.round(row['servo_error_deg'],3),'torque',np.round(row['servo_torque_nm'],3),
       'integral',np.round(row['servo_integral_nm'],3),flush=True)
(p/('standing_servo_bias_no_limits_audit.json' if no_limits else 'standing_servo_bias_audit.json')).write_text(json.dumps(rows,indent=2),encoding='utf8')
