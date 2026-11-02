"""Contrast small central-servo steps with grounded and unloaded robot; no training."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec

p=Path(__file__).parent
rows=[]
for condition in ('grounded','unloaded'):
 for delta in (-10,10):
  m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_smooth_hip_trial.xml'))
  m.opt.timestep=.00025
  d=mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m,d,0)
  if condition=='unloaded':
   m.opt.gravity[:]=0
   root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'root_free')
   if root<0:
    root=next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
   d.qpos[m.jnt_qposadr[root]+2]+=.15
  mujoco.mj_forward(m,d)
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
  ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
  hips=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)]
  initial=d.actuator_length[4];hip0=d.qpos[hips].copy(); peak=0.;peak_pair=None;stop=None; samples=[]
  for step in range(round(8/m.opt.timestep)):
   if d.time>=1:servo.set_pulse_us([1500,1500,1500,1500,1500+delta])
   before=d.time;servo.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+m.opt.timestep*.5:stop='nonfinite or reset';break
   if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):stop='contact or constraint capacity';break
   if max(abs(np.rad2deg(d.qpos[hips]-hip0)))>4.5:stop='hip guard';break
   if step%40==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
     if f[0]>peak:peak=float(f[0]);peak_pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
   if step%4000==3999:
    mujoco.mj_forward(m,d)
    samples.append({'time_s':float(d.time),'shaft_deg':float(np.rad2deg(d.actuator_length[4]-initial)),'error_deg':float(np.rad2deg(servo.target_positions[4]-d.actuator_length[4])),'torque_nm':float(d.ctrl[4]),'integral_nm':float(servo.integral_torque[4]),'shaft_speed_rad_s':float(d.actuator_velocity[4]),'hip_change_deg':np.rad2deg(d.qpos[hips]-hip0).tolist()})
  row={'condition':condition,'delta_us':delta,'duration_s':float(d.time),'valid':stop is None,'stop_reason':stop,'sampled_peak_self_contact_n':peak,'sampled_peak_pair':peak_pair,'samples':samples}
  rows.append(row);print(condition,delta,'valid',not stop,'peak',round(peak,3),'last',samples[-1] if samples else None,flush=True)
(p/'hip_ground_load_diagnostic.json').write_text(json.dumps({'results':rows,'limitations':'Unloaded means no gravity and raised root, so multiple effects change together; diagnostic comparison, not proof of single cause. Collision coverage and servo parameters are incomplete.'},indent=2),encoding='utf8')
