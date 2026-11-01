"""Sample unmerged torso/tower proxy gaps over the assumed hip range."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_nomerge_proxy_gap_only.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
chassis=[(i,n) for i,n in enumerate(names) if n and n.startswith('nomerge_chassis5_')]
towers=[(i,n) for i,n in enumerate(names) if n and (n.startswith('highres_L0_') or n.startswith('highres_L1_'))]
if len(chassis)!=387 or len(towers)!=192:raise RuntimeError('Unexpected proxy counts')
joint_names=['hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate']
adr=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in joint_names]
pruned='--pruned' in sys.argv
q0=d.qpos.copy();pairs=[(a,an,b,bn) for a,an in chassis for b,bn in towers]
mins=np.ones(len(pairs));initial=None;at_angle=np.zeros(len(pairs))
for angle in np.linspace(-5,5,201):
 d.qpos[:]=q0;d.qpos[adr[0]]+=np.deg2rad(-2*angle);d.qpos[adr[1]]+=np.deg2rad(angle);d.qpos[adr[2]]+=np.deg2rad(angle)
 mujoco.mj_forward(m,d)
 distances=np.fromiter((mujoco.mj_geomDistance(m,d,a,b,.003 if pruned else .001,None) for a,_,b,_ in pairs),float,count=len(pairs))
 changed=distances<mins;mins=np.minimum(mins,distances);at_angle[changed]=angle
 if abs(angle)<1e-9:initial=distances.copy()
assert initial is not None
rows=[{'geoms':[an,bn],'initial_mm':float(initial[i]*1000),'min_mm':float(mins[i]*1000),'min_at_hip_deg':float(at_angle[i])}
      for i,(_,an,_,bn) in enumerate(pairs) if mins[i]<(.002 if pruned else .0001000000001)]
out={'sample_count':201,'hip_range_deg':[-5,5],'pair_count':len(pairs),'selection_threshold_mm':2. if pruned else .1,'selected_pair_count':len(rows),
 'overlapping_count':int(np.sum(mins<0)),'minimum_signed_distance_mm':float(np.min(mins)*1000),'near_pairs':rows,
 'limitations':'Proxy distances at 201 prescribed hip positions; no continuous-motion guarantee or physical CAD clearance measurement.'}
(p/('nomerge_proxy_pruning_audit.json' if pruned else 'nomerge_proxy_range_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print({k:v for k,v in out.items() if k!='near_pairs'},flush=True)
