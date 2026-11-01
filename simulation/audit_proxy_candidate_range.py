"""Measure all high-resolution torso-5/tower pairs over assumed hip range."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_highres_gap_only.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
chassis=[(i,n) for i,n in enumerate(names) if n and n.startswith('highres_chassis5_')]
towers=[(i,n) for i,n in enumerate(names) if n and (n.startswith('highres_L0_') or n.startswith('highres_L1_'))]
if len(chassis)!=96 or len(towers)!=192:raise RuntimeError('Expected complete high-resolution geometry')
joint_names=['hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate']
adr=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in joint_names]
q0=d.qpos.copy();pairs=[(a,an,b,bn) for a,an in chassis for b,bn in towers]
mins=np.ones(len(pairs));initial=None;at_angle=np.zeros(len(pairs))
dense='--dense' in sys.argv
sample_count=201 if dense else 21
for angle in np.linspace(-5,5,sample_count):
 d.qpos[:]=q0
 d.qpos[adr[0]]+=np.deg2rad(-2*angle)
 d.qpos[adr[1]]+=np.deg2rad(angle)
 d.qpos[adr[2]]+=np.deg2rad(angle)
 mujoco.mj_forward(m,d)
 distances=np.fromiter((mujoco.mj_geomDistance(m,d,a,b,.001,None) for a,_,b,_ in pairs),float,count=len(pairs))
 changed=distances<mins;mins=np.minimum(mins,distances);at_angle[changed]=angle
 if abs(angle)<1e-9:initial=distances.copy()
 if not dense or round((angle+5)*10)%20==0:
  print('hip deg',round(float(angle),3),'overlapping convex pairs',int(np.sum(distances<0)),flush=True)
assert initial is not None
rows=[{'geoms':[an,bn],'initial_mm':float(initial[i]*1000),'min_mm':float(mins[i]*1000),'min_at_hip_deg':float(at_angle[i])}
      for i,(_,an,_,bn) in enumerate(pairs) if mins[i]<=.0003]
build=json.loads((p/'selective_collision_trial_build.json').read_text(encoding='utf8'))
excluded={tuple(row['geoms']) for row in build['excluded_at_initial_pose']}
out={'hypothetical_hip_range_deg':[-5,5],'sample_count':sample_count,'pair_count':len(pairs),
     'pairs_with_overlap_in_any_sample':int(np.sum(mins<0)),
     'previously_enabled_pairs_with_sampled_overlap':sum(r['min_mm']<0 and tuple(r['geoms']) not in excluded for r in rows),
     'previously_excluded_pairs_never_overlapping_in_samples':sum(tuple(r['geoms']) in excluded and r['min_mm']>=0 for r in rows),
     'near_pairs':rows,'limitations':f'{sample_count} kinematic samples, proxy distances only; not continuous clearance, contact proof or measured real travel.'}
(p/('proxy_candidate_range_dense_audit.json' if dense else 'proxy_candidate_range_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print({k:v for k,v in out.items() if k!='near_pairs'},flush=True)
