"""Inspect current ground contacts and foot proxy state in a fixed standing test."""
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import json
import sys
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
box='--aligned-box' in sys.argv
filename='diagnostic_aligned_foot_proxy_trial.xml' if box else 'diagnostic_range_collision_trial.xml'
m=mujoco.MjModel.from_xml_path(str(p/filename));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
 kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
proxies=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{i}_foot_proxy') for i in (0,1)]
configured=[{'first':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(a)),
             'second':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(b))}
            for a,b in zip(m.pair_geom1,m.pair_geom2) if ground in (a,b) and (a in proxies or b in proxies)]
samples=[]
for step in range(round(4/m.opt.timestep)):
 old=d.time;servo.step()
 if not np.isfinite(d.qpos).all() or d.time<old+m.opt.timestep*.5:raise RuntimeError('Simulation reset')
 if step in (round(1/m.opt.timestep)-1,round(4/m.opt.timestep)-1):
  by_geom=defaultdict(lambda:{'contact_points':0,'sum_normal_n':0.,'min_distance_mm':1e9})
  for i,c in enumerate(d.contact[:d.ncon]):
   if ground not in (c.geom1,c.geom2):continue
   other=c.geom2 if c.geom1==ground else c.geom1
   name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other)
   f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
   r=by_geom[name];r['contact_points']+=1;r['sum_normal_n']+=float(f[0]);r['min_distance_mm']=min(r['min_distance_mm'],float(c.dist*1000))
  mujoco.mj_forward(m,d)
  samples.append({'time_s':float(d.time),'contacts':dict(by_geom),
    'foot_proxy_ground_distance_mm':[float(mujoco.mj_geomDistance(m,d,ground,g,.5,None)*1000) for g in proxies]})
out={'model':filename,'explicit_ground_foot_proxy_pairs':configured,'samples':samples,
 'limitations':'Only two simulation instants under hypothetical gains; rendered mesh geometry used for explicit ground contact may be convexified. Foot proxy clearance alone does not validate its dimensions or pose.'}
(p/('aligned_foot_ground_contact_audit.json' if box else 'ground_contact_geometry_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
