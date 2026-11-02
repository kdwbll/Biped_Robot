"""Classify excluded proxy closest points relative to the ideal hip cylinder axis."""
from pathlib import Path
import json
import mujoco
import numpy as np
p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_separate_sole_contact_trial.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);q0=d.qpos.copy()
source=json.loads((p/'chassis0_finer_proxy_range_audit.json').read_text(encoding='utf8'))
exclude=[x for x in source['near_pairs'] if x['min_mm']<=.05]
joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft');rows=[]
for row in exclude:
 a,b=(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in row['geoms']);leg=0 if row['geoms'][1].startswith('nomerge_L0_') else 1
 angle=row['min_at_hip_deg'];d.qpos[:]=q0;d.qpos[m.jnt_qposadr[shaft]]+=np.deg2rad(-2*angle)
 for j in joints:d.qpos[m.jnt_qposadr[j]]+=np.deg2rad(angle)
 mujoco.mj_forward(m,d);ends=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,a,b,.003,ends));axis=d.xaxis[joints[leg]];origin=d.xanchor[joints[leg]]
 positions=[]
 for pt in (ends[:3],ends[3:]):
  v=pt-origin;h=np.dot(v,axis);r=np.linalg.norm(v-h*axis);positions.append({'radius_mm':float(r*1000),'axis_offset_mm':float(h*1000)})
 rows.append({'geoms':row['geoms'],'sample_angle_deg':angle,'distance_mm':distance*1000,'closest_points':positions})
near=lambda x: all(18<=a['radius_mm']<=22 for a in x['closest_points'])
nonoverlap=[x for x in rows if x['distance_mm']>=0]
result={'count':len(rows),'nonnegative_count':len(nonoverlap),'near_18_22mm_both_count':sum(near(x) for x in nonoverlap),'results':rows,
 'limitations':'Closest points at the prior sampled minimum, convex proxies only. For overlaps, witnesses may be unreliable. Radius alone does not establish CAD surface identity, bearing contact or safe exclusion. Does not cover other joint motion.'}
(p/'remaining_collision_axis_classification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('remaining',len(rows),'nonnegative',len(nonoverlap),'both witness radii 18-22mm',result['near_18_22mm_both_count'],flush=True)
