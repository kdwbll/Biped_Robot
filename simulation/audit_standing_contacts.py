"""Inspect simulated contact forces in a neutral stance, not a hardware test."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'biped_standing_gear_candidate.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
for _ in range(round(6/m.opt.timestep)):s.step()
mujoco.mj_forward(m,d)
rows=[]
for i,c in enumerate(d.contact[:d.ncon]):
 force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
 rows.append({'geoms':names,'distance_mm':float(c.dist*1000),'normal_force_n':float(force[0]),
              'contact_point_m':c.pos.tolist()})
rows.sort(key=lambda x:x['normal_force_n'],reverse=True)
(p/'standing_contact_forces_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print('contact count',len(rows),'non-ground contacts with normal force > .1 N:')
for row in (r for r in rows if 'ground' not in r['geoms'] and r['normal_force_n']>.1):
 print(row['geoms'],round(row['normal_force_n'],3),'N',round(row['distance_mm'],4),'mm')
