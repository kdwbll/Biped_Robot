"""Compare saved implicit-damping model with equivalent in-memory changes."""
from pathlib import Path
from dataclasses import replace
import json,mujoco,numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo
p=Path(__file__).parent
m0=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_damping_trial.xml'));m1=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'))
for name in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50'):
 eq=mujoco.mj_name2id(m0,mujoco.mjtObj.mjOBJ_EQUALITY,name)
 if eq<0:raise RuntimeError('Missing equality')
 m0.eq_solref[eq,0]=.0005
states=[]
for m in (m0,m1):
 m.opt.timestep=.00025;d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
 for step in range(2000):
  if d.time>=.1:s.set_pulse_us([1500,1500,1500,1500,1520])
  previous=d.time;s.step()
  if not np.isfinite(d.qpos).all() or d.time<previous+.5*m.opt.timestep:raise RuntimeError('Numerical failure')
 states.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time)))
q=float(np.max(np.abs(states[0][0]-states[1][0])));v=float(np.max(np.abs(states[0][1]-states[1][1])));u=float(np.max(np.abs(states[0][2]-states[1][2])))
if max(q,v,u)>1e-9:raise RuntimeError(('Variants differ',q,v,u))
out={'duration_s':states[0][3],'max_qpos_difference':q,'max_qvel_difference':v,'max_control_difference':u,'limitations':'Only verifies saved model and in-memory parameters coincide under one short input; not physical validity.'}
(p/'sector_implicit_stiff_equivalence_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
