"""Compare soft gear constraints and integration steps in a temporary model."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
cases=[(.00025,.001,False),(.00025,.001,True),(.00025,.0005,True),(.000125,.0005,True)]
out=[]
for dt,tau,strong in cases:
 m=mujoco.MjModel.from_xml_path(str(p/'biped_fresh_mass_only_candidate.xml'));m.opt.timestep=dt
 eqids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,n) for n in
        ['opposite_world_rotation_assumption','estimated_pinion_23_gear_50']]
 for i in eqids:
  m.eq_solref[i,0]=tau
  if strong:m.eq_solimp[i]=[.99,.999,.001,.5,2.]
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');z0=d.xipos[body,2]
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
 err=np.zeros(2);speed=np.zeros(5);torque=np.zeros(5)
 for _ in range(round(4/dt)):
  servo.step();speed=np.maximum(speed,np.abs(d.actuator_velocity));torque=np.maximum(torque,np.abs(d.ctrl))
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError((dt,tau,strong,d.time))
  for k,eqid in enumerate(eqids):
   match=np.flatnonzero((d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id==eqid))
   if len(match):err[k]=max(err[k],abs(d.efc_pos[match[0]]))
 mujoco.mj_forward(m,d)
 row={'dt_s':dt,'gear_time_constant_s':tau,'stronger_constraint_impedance':strong,
      'duration_s':d.time,'height_loss_mm':float((z0-d.xipos[body,2])*1000),
      'max_gear_constraint_error_deg':np.rad2deg(err).tolist(),
      'max_servo_speed_rad_s':speed.tolist(),'max_abs_torque_nm':torque.tolist()}
 out.append(row)
 print('dt',dt,'tau',tau,'strong',strong,'sag',round(row['height_loss_mm'],3),
       'gear error deg',np.round(row['max_gear_constraint_error_deg'],3),
       'peak speed',round(max(speed),2),'peak torque',round(max(torque),2),flush=True)
(p/'gear_solver_fidelity_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
