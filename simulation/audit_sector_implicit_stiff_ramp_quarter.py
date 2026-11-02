"""Sensitivity of implicit candidate to assumed hinge friction; no training."""
from pathlib import Path
from dataclasses import replace
import json,mujoco,numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent;rows=[]
for timestep in (.0000625,):
 friction=.05
 for delta in (-20,):
  m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_damping_trial.xml'));m.opt.timestep=timestep
  for eqname in ('opposite_world_rotation_assumption','candidate_pinion_25_gear_50'):
   eq=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,eqname)
   if eq<0:raise RuntimeError('Missing gear constraint')
   m.eq_solref[eq,0]=.0005
  for leg in (0,1):j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate');m.dof_frictionloss[m.jnt_dofadr[j]]=friction
  d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
  hips=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)];h0=d.qpos[hips].copy();q0=d.actuator_length[4];maxspeed=0.;maxhip=0.;maxgear=0.;stop=None
  for step in range(round(4/m.opt.timestep)):
   if d.time>=1:s.set_pulse_us([1500,1500,1500,1500,1500+delta*min(1.,(d.time-1.)/2.)])
   before=d.time;s.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+.5*m.opt.timestep:stop='nonfinite/reset';break
   if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):stop='capacity';break
   maxspeed=max(maxspeed,abs(float(d.actuator_velocity[4])));maxhip=max(maxhip,float(np.max(np.abs(np.rad2deg(d.qpos[hips]-h0)))))
   maxgear=max(maxgear,abs(float(np.rad2deg((d.qpos[hips[0]]-h0[0])+.5*(d.actuator_length[4]-q0)))))
   if maxhip>4.5:stop='hip range';break
  mujoco.mj_forward(m,d);row={'timestep_s':timestep,'frictionloss_nm_per_hip':friction,'pulse_delta_us':delta,'time_s':float(d.time),'stop_reason':stop,'shaft_delta_deg':float(np.rad2deg(d.actuator_length[4]-q0)),'shaft_error_deg':float(np.rad2deg(s.target_positions[4]-d.actuator_length[4])),'max_hip_deg':maxhip,'max_shaft_speed_rad_s':maxspeed,'max_gear_residual_deg':maxgear};rows.append(row);print(row,flush=True)
out={'results':rows,'gear_solref_s':.0005,'limitations':'A 2 s ramp to -20 us at a quarter time step and one assumed friction value. Joint friction is not measured and sampling does not establish realistic torque/clearance. Not training or hardware.'}
(p/'sector_implicit_stiff_ramp_quarter_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
