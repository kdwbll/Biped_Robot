"""Compare partial-collision trial under small opposite hip commands."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
range_trial='--range' in sys.argv
models=('diagnostic_selective_collision_trial.xml','diagnostic_range_collision_trial.xml') if range_trial else ('diagnostic_post_shaft_fix_gear_phase_25.xml','diagnostic_selective_collision_trial.xml')
for filename in models:
 for delta in (-10,10):
  m=mujoco.MjModel.from_xml_path(str(p/filename));d=mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
   kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
  ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');peak=0.;pair=None;valid=True
  initial=servo.initial_positions[4]
  for step in range(round(6/m.opt.timestep)):
   if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1500+delta])
   old=d.time;servo.step()
   if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:valid=False;break
   if step%20==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
     if f[0]>peak:
      peak=float(f[0]);pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  mujoco.mj_forward(m,d)
  row={'model':filename,'pulse_delta_us':delta,'valid':valid,'actual_delta_deg':float(np.degrees(d.actuator_length[4]-initial)),
   'error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])),
   'sampled_peak_self_contact_n':peak,'peak_pair':pair}
  rows.append(row);print(row,flush=True)
(p/('range_collision_motion_audit.json' if range_trial else 'selective_collision_motion_audit.json')).write_text(json.dumps({'results':rows,'limitations':
 'Short limited motion; near/overlapping high-resolution pairs deliberately disabled, remaining geometric coverage not established.'},indent=2),encoding='utf8')
