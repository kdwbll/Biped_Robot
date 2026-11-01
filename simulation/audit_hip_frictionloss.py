"""Sensitivity of assumed hip joint dry friction, with no planar contact duplication.

MuJoCo joint frictionloss is load-independent and is not a calibrated contact
friction/preload model. This script changes in-memory model parameters only.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import time
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[];smoke='--smoke' in sys.argv
for loss in ((.05,) if smoke else (0.,.01,.05,.1)):
 for sign in ((1,) if smoke else (-1,1)):
  m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'));d=mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
  joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
  for joint in joints:m.dof_frictionloss[m.jnt_dofadr[joint]]=loss
  mujoco.mj_forward(m,d)
  friction_type=int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
  friction_values=d.efc_frictionloss[d.efc_type==friction_type].copy()
  if loss>0 and (len(friction_values)!=2 or not np.allclose(friction_values,loss)):
   raise RuntimeError('Expected two active MuJoCo hip friction constraints')
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
      kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
  initial=float(d.actuator_length[4]);ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
  hipaddr=[int(m.jnt_qposadr[j]) for j in joints];hip_initial=d.qpos[hipaddr].copy()
  maxhip=0.;peakforce=0.;peakpair=None;peaktorque=0.;valid=True;reason=None;start=time.perf_counter()
  for step in range(round(6/m.opt.timestep)):
   if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1500+sign*10])
   old=float(d.time);servo.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+m.opt.timestep*.5:
    valid=False;reason='nonfinite or automatic reset';break
   if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
    valid=False;reason='collision or constraint arena exhausted';break
   maxhip=max(maxhip,float(np.max(np.abs(np.degrees(d.qpos[hipaddr]-hip_initial)))))
   peaktorque=max(peaktorque,abs(float(d.ctrl[4])))
   if maxhip>4.5:valid=False;reason='exceeded diagnostic hip range';break
   if step%20==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
     if force[0]>peakforce:
      peakforce=float(force[0]);peakpair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  mujoco.mj_forward(m,d)
  row={'assumed_frictionloss_nm_per_hip':loss,'initial_friction_constraint_count':len(friction_values),'pulse_delta_us':sign*10,'valid':valid,'invalid_reason':reason,
       'duration_s':float(d.time),'actual_shaft_change_deg':float(np.degrees(d.actuator_length[4]-initial)),
       'shaft_target_error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])),
       'max_hip_change_deg':maxhip,'peak_motor_torque_nm':peaktorque,'sampled_peak_self_contact_n':peakforce,
       'sampled_peak_pair':peakpair,'wall_s':time.perf_counter()-start}
  rows.append(row);print(row,flush=True)
out={'results':rows,'limitations':'Friction loss values are hypothetical, equal on both hips, load-independent, and not a measurement of surface friction or preload. Tests cover small motions only; collision exclusions remain. No training.'}
(p/('hip_frictionloss_smoke_audit.json' if smoke else 'hip_frictionloss_sensitivity_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
