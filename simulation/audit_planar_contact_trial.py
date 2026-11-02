"""Measure the experimental planar facets during a short standing test."""
from pathlib import Path
from dataclasses import replace
import json
import time
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;rows=[]
for filename in ('diagnostic_chassis0_collision_trial.xml','diagnostic_cad_plane_contact_trial.xml'):
 m=mujoco.MjModel.from_xml_path(str(p/filename));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
  kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground');body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
 z0=float(d.xipos[body,2]);peak=0.;plane_peak=0.;plane_contacts=0;stop=None;start=time.perf_counter()
 for step in range(round(1/m.opt.timestep)):
  old=float(d.time);servo.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+m.opt.timestep*.5:
   stop='nonfinite state or automatic reset';break
  if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
   stop='contact or constraint arena exhausted';break
  if step%10==0:
   pc=0;total=0.
   for i,c in enumerate(d.contact[:d.ncon]):
    if ground in (c.geom1,c.geom2):continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);peak=max(peak,float(f[0]))
    a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or ''
    b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
    if a.startswith('plane_') and b.startswith('plane_'):pc+=1;total+=float(f[0])
   plane_contacts=max(plane_contacts,pc);plane_peak=max(plane_peak,total)
   if peak>500:stop='sampled internal contact force exceeded 500 N';break
 mujoco.mj_forward(m,d)
 row={'model':filename,'duration_s':float(d.time),'valid':stop is None,'stop_reason':stop,'wall_s':time.perf_counter()-start,
      'height_change_mm':float((d.xipos[body,2]-z0)*1000),'sampled_peak_internal_point_n':peak,
      'sampled_max_planar_contact_count':plane_contacts,'sampled_max_planar_normal_sum_n':plane_peak}
 rows.append(row);print(row,flush=True)
(p/'cad_plane_contact_trial_audit.json').write_text(json.dumps({'results':rows,'limitations':
 'One second simulated neutral standing; contact data sampled every 10 steps, the stop threshold is diagnostic only. Friction and surface compliance are unmeasured.'},indent=2),encoding='utf8')
