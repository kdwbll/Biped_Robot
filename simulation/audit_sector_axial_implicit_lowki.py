"""Compare small-to-moderate bidirectional commands on sector collision candidate."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec
p=Path(__file__).parent;rows=[]
axial='--axial' in sys.argv
for filename in ('diagnostic_sector_axial_contact_trial.xml',):
 for delta in (-20,20):
  m=mujoco.MjModel.from_xml_path(str(p/filename));m.opt.timestep=.00025
  for leg in (0,1):
   joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate');m.dof_frictionloss[m.jnt_dofadr[joint]]=.1
  shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft');m.dof_damping[m.jnt_dofadr[shaft]]=.301
  d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
  servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,1.5],hip_damping=0.)
  h=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')]) for i in (0,1)];h0=d.qpos[h].copy();initial=d.actuator_length[4];ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');peak=0.;peakpair=None;maxhip=0.;sector_peak=0.;sector_contact_samples=0;stop=None;maxspeed=0.
  for step in range(round(6/m.opt.timestep)):
   if d.time>=1.:servo.set_pulse_us([1500,1500,1500,1500,1500+delta])
   old=float(d.time);servo.step()
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+.5*m.opt.timestep:stop='nonfinite or auto reset';break
   if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):stop='capacity warning';break
   maxhip=max(maxhip,float(np.max(np.abs(np.rad2deg(d.qpos[h]-h0)))))
   if maxhip>4.5:stop='hypothetical hip guard';break
   maxspeed=max(maxspeed,abs(float(d.actuator_velocity[4])))
   if step%20==0:
    for i,c in enumerate(d.contact[:d.ncon]):
     if ground in (c.geom1,c.geom2):continue
     f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
     if f[0]>peak:peak=float(f[0]);peakpair=names
     if any(n and n.startswith('sector_chassis0_') for n in names):sector_contact_samples+=1;sector_peak=max(sector_peak,float(f[0]))
  mujoco.mj_forward(m,d)
  row={'model':filename,'pulse_delta_us':delta,'duration_s':float(d.time),'stop_reason':stop,'shaft_displacement_deg':float(np.rad2deg(d.actuator_length[4]-initial)),'shaft_error_deg':float(np.rad2deg(servo.target_positions[4]-d.actuator_length[4])),'max_hip_change_deg':maxhip,'max_shaft_speed_rad_s':maxspeed,'sampled_peak_self_contact_n':peak,'sampled_peak_pair':peakpair,'sampled_sector_contact_points':sector_contact_samples,'sampled_peak_sector_contact_n':sector_peak}
  rows.append(row);print(row,flush=True)
out={'results':rows,'hypothetical_joint_frictionloss_nm':.1,'hypothetical_implicit_shaft_damping_nm_s_per_rad':.301,'explicit_servo_damping':0.,'hypothetical_hip_integral_gain':1.5,'limitations':'Short controlled diagnostic with 20 us step, sampled forces, unmeasured gains and incomplete collisions. Range check only inside this script, not XML. Not training or real-hardware evidence.'}
(p/'sector_axial_implicit_lowki_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
