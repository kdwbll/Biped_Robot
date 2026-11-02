"""Record peak force direction for the suspected chassis/tower proxy pair."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_collision_trial.xml'));m.opt.timestep=.00025;d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6]);ga,gb=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('dc_chassis_1_1','tower_L0_tower_8')];hip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'L0_hip_candidate');h=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)];initial=d.qpos[h].copy();peak=None
for step in range(round(6/m.opt.timestep)):
 if d.time>=1:s.set_pulse_us([1500,1500,1500,1500,1480])
 old=d.time;s.step()
 if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+.5*m.opt.timestep:raise RuntimeError('Numerical reset')
 if np.any(np.abs(np.rad2deg(d.qpos[h]-initial))>4.5):raise RuntimeError('Hip guard')
 if step%20:continue
 for i,c in enumerate(d.contact[:d.ncon]):
  if {c.geom1,c.geom2}!={ga,gb}:continue
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  if peak is None or f[0]>peak['normal_force_n']:
   normal=np.array(c.frame).reshape(3,3)[0];axis=d.xaxis[hip].copy();point=np.array(c.pos);j1=np.zeros((3,m.nv));j2=np.zeros((3,m.nv));mujoco.mj_jac(m,d,j1,None,point,m.geom_bodyid[ga]);mujoco.mj_jac(m,d,j2,None,point,m.geom_bodyid[gb]);jac=normal@(j2-j1)
   peak={'time_s':float(d.time),'normal_force_n':float(f[0]),'penetration_mm':float(-c.dist*1000),'normal_world':normal.tolist(),'hip_axis_world':axis.tolist(),'normal_axis_abs_dot':float(abs(normal@axis)),'normal_relative_jacobian_max_abs':float(np.max(np.abs(jac))),'hip_deg':np.rad2deg(d.qpos[h]-initial).tolist()}
if peak is None:raise RuntimeError('No sampled contact for requested pair')
out={'pair':['dc_chassis_1_1','tower_L0_tower_8'],'command_delta_us':-20,'sampling_every_steps':20,'peak':peak,'limitations':'Peak sampled at 5 ms intervals for one command and one approximate proxy pair. Contact normal and Jacobian are simulation results, not real contact force or proof over all states.'}
(p/'sector_peak_contact_direction_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(peak,indent=2),flush=True)
