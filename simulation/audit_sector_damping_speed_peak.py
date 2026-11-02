"""Locate speed peaks under hypothesized servo damping without training."""
from pathlib import Path
from dataclasses import replace
import json,mujoco,numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo
p=Path(__file__).parent;rows=[]
for damp in (.01,.3):
 m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_axial_contact_trial.xml'));m.opt.timestep=.00025
 for leg in (0,1):j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate');m.dof_frictionloss[m.jnt_dofadr[j]]=.1
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=damp);peak=None
 for step in range(round(2/m.opt.timestep)):
  if d.time>=1:s.set_pulse_us([1500,1500,1500,1500,1480])
  before=d.time;s.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+.5*m.opt.timestep:raise RuntimeError('Numerical error')
  speed=float(d.actuator_velocity[4])
  if peak is None or abs(speed)>abs(peak['speed_rad_s']):peak={'time_s':float(d.time),'speed_rad_s':speed,'torque_nm':float(d.ctrl[4]),'shaft_deg':float(np.rad2deg(d.actuator_length[4]-s.initial_positions[4])),'target_deg':float(np.rad2deg(s.target_positions[4]-s.initial_positions[4]))}
 row={'hip_damping':damp,'peak':peak};rows.append(row);print(row,flush=True)
(p/'sector_damping_speed_peak_audit.json').write_text(json.dumps({'results':rows,'limitations':'Two seconds, two hypothetical gains, one command. Speed is simulation output; not physical servo limit or hardware test.'},indent=2),encoding='utf8')
