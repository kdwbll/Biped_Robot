"""Bounded sweep of finer proxies over the assumed +/-5 degree hip rotation."""
from pathlib import Path
import json
import math
import time
import numpy as np
import mujoco

p=Path(__file__).parent;start=time.perf_counter()
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_finer_full_gap_only.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
chassis=[(i,n) for i,n in enumerate(names) if n and n.startswith('nomerge_chassis5_')]
towers=[(i,n) for i,n in enumerate(names) if n and (n.startswith('nomerge_L0_') or n.startswith('nomerge_L1_'))]
if len(chassis)!=822 or len(towers)!=663:raise RuntimeError('Unexpected finer proxy counts')
joint=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
adr=[int(m.jnt_qposadr[j]) for j in [shaft,*joint]];q0=d.qpos.copy()
# For a rigid tower geom, no point can move farther than 2*r*sin(5deg/2)
# from its initial pose. Prune only if initial proxy separation exceeds this
# bound plus 2mm; a separate margin handles potential close contacts.
candidates=[];largest_radius=0.
for b,bn in towers:
 leg=0 if bn.startswith('nomerge_L0_') else 1
 radius=float(np.linalg.norm(d.geom_xpos[b]-d.xanchor[joint[leg]])+m.geom_rbound[b])
 largest_radius=max(largest_radius,radius)
 max_travel=2*radius*math.sin(math.radians(5)/2);cutoff=.002+max_travel
 if cutoff>=.099:raise RuntimeError('Initial separation query cap is too small')
 for a,an in chassis:
  distance=float(mujoco.mj_geomDistance(m,d,a,b,.1,None))
  if distance<=cutoff:candidates.append((a,an,b,bn))
print('total pairs',len(chassis)*len(towers),'candidate pairs',len(candidates),flush=True)
mins=np.ones(len(candidates));at_angle=np.zeros(len(candidates));initial=None
for angle in np.linspace(-5,5,201):
 d.qpos[:]=q0;d.qpos[adr[0]]+=np.deg2rad(-2*angle);d.qpos[adr[1]]+=np.deg2rad(angle);d.qpos[adr[2]]+=np.deg2rad(angle);mujoco.mj_forward(m,d)
 distances=np.fromiter((mujoco.mj_geomDistance(m,d,a,b,.003,None) for a,_,b,_ in candidates),float,count=len(candidates))
 changed=distances<mins;mins=np.minimum(mins,distances);at_angle[changed]=angle
 if abs(angle)<1e-9:initial=distances.copy()
 if abs(angle-round(angle))<1e-9:print('hip deg',round(float(angle),2),flush=True)
assert initial is not None
near=[{'geoms':[an,bn],'initial_mm':float(initial[i]*1000),'min_mm':float(mins[i]*1000),'min_at_hip_deg':float(at_angle[i])}
      for i,(_,an,_,bn) in enumerate(candidates) if mins[i]<.002]
excluded=[r for r in near if r['min_mm']<=.1]
out={'sample_count':201,'hip_range_deg':[-5,5],'chassis_count':len(chassis),'tower_count':len(towers),
 'total_pairs':len(chassis)*len(towers),'candidate_pairs_after_motion_bound':len(candidates),
 'within_2mm_count':len(near),'unresolved_within_0_1mm_count':len(excluded),
 'overlap_count':sum(r['min_mm']<0 for r in excluded),'minimum_signed_distance_mm':min(r['min_mm'] for r in near),
 'largest_tower_radius_m':largest_radius,'elapsed_s':time.perf_counter()-start,'near_pairs':near,
 'limitations':'Rigid independent hip rotation only; proxy separations are approximate. Motion bound and sampling do not cover soft-limit overshoot, model errors, or other motions; no hardware or training validation.'}
(p/'finer_proxy_range_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print({k:v for k,v in out.items() if k!='near_pairs'},flush=True)
