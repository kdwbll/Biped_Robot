"""Inspect gear equality and control response with one axial pair omitted."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent;out=[]
for filename in ('diagnostic_sector_collision_trial.xml','diagnostic_sector_axial_contact_trial.xml'):
 m=mujoco.MjModel.from_xml_path(str(p/filename));m.opt.timestep=.00025;d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6]);h=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)];h0=d.qpos[h].copy();q0=d.actuator_length[4];peak=0.;maxerr=0.;samples=[];stop=None
 for step in range(round(6/m.opt.timestep)):
  if d.time>=1:servo.set_pulse_us([1500,1500,1500,1500,1480])
  old=d.time;servo.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+.5*m.opt.timestep:stop='nonfinite or reset';break
  hip=np.rad2deg(d.qpos[h]-h0);shaft=float(np.rad2deg(d.actuator_length[4]-q0));error=hip[0]+.5*shaft;maxerr=max(maxerr,abs(error));peak=max(peak,max(abs(hip)))
  if peak>4.5:stop='hip range';break
  if step%4000==3999:samples.append({'time_s':float(d.time),'hip_delta_deg':hip.tolist(),'shaft_delta_deg':shaft,'gear_residual_deg':float(error),'servo_error_deg':float(np.rad2deg(servo.target_positions[4]-d.actuator_length[4])),'torque_nm':float(d.ctrl[4])})
 row={'model':filename,'stop_reason':stop,'max_gear_residual_deg':float(maxerr),'max_hip_deg':float(peak),'samples':samples};out.append(row);print(filename,'stop',stop,'max gear residual',maxerr,'max hip',peak,'final',samples[-1] if samples else None,flush=True)
(p/'axial_contact_gear_response_audit.json').write_text(json.dumps({'results':out,'limitations':'Two six-second ideal-model tests for one command; monitors one linear gear constraint residual only. Does not validate real servo, all contact forces or other maneuvers.'},indent=2),encoding='utf8')
