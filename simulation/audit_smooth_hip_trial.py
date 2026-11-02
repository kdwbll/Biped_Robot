"""Compare assumed smooth and damped hips under small opposite commands."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
for filename in ('diagnostic_chassis0_collision_trial.xml','diagnostic_smooth_hip_trial.xml'):
 for delta in (-10,10):
  m=mujoco.MjModel.from_xml_path(str(p/filename));d=mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
   kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
  ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');initial=float(d.actuator_length[4])
  hips=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)]
  q0=d.qpos[hips].copy();maxhip=0.;peak=0.;maxspeed=0.;stop=None
  for step in range(round(4/m.opt.timestep)):
   if d.time>=1.:servo.set_pulse_us([1500,1500,1500,1500,1500+delta])
   old=float(d.time);servo.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+m.opt.timestep*.5:
    stop='nonfinite state or automatic reset';break
   if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
    stop='contact or constraint capacity exceeded';break
   maxhip=max(maxhip,float(np.max(np.abs(np.degrees(d.qpos[hips]-q0)))))
   maxspeed=max(maxspeed,abs(float(d.actuator_velocity[4])))
   if maxhip>4.5:stop='exceeded diagnostic hip range';break
   if step%20==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);peak=max(peak,float(f[0]))
  mujoco.mj_forward(m,d)
  row={'model':filename,'pulse_delta_us':delta,'valid':stop is None,'stop_reason':stop,'duration_s':float(d.time),
   'shaft_delta_deg':float(np.degrees(d.actuator_length[4]-initial)),
   'shaft_target_error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])),
   'max_hip_change_deg':maxhip,'max_shaft_speed_rad_s':maxspeed,'sampled_peak_self_contact_n':peak}
  rows.append(row);print(row,flush=True)
(p/'smooth_hip_trial_audit.json').write_text(json.dumps({'results':rows,
 'limitations':'Ideal zero damping is a hypothesis; finite small motions, unmeasured actuator and incomplete collisions. Does not measure actual lubricated friction.'},indent=2),encoding='utf8')
