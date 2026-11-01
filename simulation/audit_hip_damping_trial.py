"""Compare hypothetical central servo damping; no training or hardware control."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
for damping in (.01,.04,.08,.16):
 m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
     kp_by_servo=[4,4,4,4,12],ki_by_servo=[1.5,1.5,1.5,1.5,10],hip_damping=damping)
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 initial=servo.initial_positions[4];peak=0.;valid=True
 for step in range(round(6/m.opt.timestep)):
  if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1510])
  old=d.time;servo.step()
  if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:valid=False;break
  if step%20==0:
   for i,c in enumerate(d.contact[:d.ncon]):
    if ground in (c.geom1,c.geom2):continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);peak=max(peak,float(f[0]))
 mujoco.mj_forward(m,d)
 row={'hip_damping_nm_per_rad_s':damping,'valid':valid,'actual_deg':float(np.degrees(d.actuator_length[4]-initial)),
      'error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])), 'sampled_peak_self_contact_n':peak}
 rows.append(row);print(row,flush=True)
(p/'hip_damping_trial_audit.json').write_text(json.dumps({'results':rows,
 'limitations':'Numerical gain experiments, not measured servo parameters; incomplete collisions; sampled forces.'},indent=2),encoding='utf8')
