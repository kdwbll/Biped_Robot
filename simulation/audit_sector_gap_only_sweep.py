"""Sweep selected sector and original proxy distances over assumed hip angles."""
from pathlib import Path
import json,math
import mujoco
import numpy as np
p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_gap_only.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);q0=d.qpos.copy()
sector=json.loads((p/'sector_clip_prototype_audit.json').read_text())['results'];mapping={f"nomerge_{x['geom']}":[f"sector_{Path(f).stem}" for f in x['files']] for x in sector}
source=json.loads((p/'sector_gap_only_witness_audit.json').read_text())['results'];joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)];shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft');items=[]
for x in source:
 a,b=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in x['geoms']];pieces=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in mapping[x['geoms'][0]]]
 leg=x['leg'];joint=joints[leg];mount=m.jnt_bodyid[joint];body=m.geom_bodyid[b]
 while body!=mount:
  if body<=0 or m.body_jntnum[body]:raise RuntimeError('Tower proxy not rigidly attached to mount')
  body=m.body_parentid[body]
 rootbody=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
 for geom in [a,*pieces]:
  body=m.geom_bodyid[geom]
  while body!=rootbody:
   if body<=0 or m.body_jntnum[body]:raise RuntimeError('Chassis proxy moves relative to robot root')
   body=m.body_parentid[body]
 axis=d.xaxis[joint];v=d.geom_xpos[b]-d.xanchor[joint];rad=float(np.linalg.norm(v-axis*np.dot(v,axis))+m.geom_rbound[b])
 items.append((a,b,pieces,rad))
angles=np.linspace(-5,5,201);old=np.full(len(items),np.inf);new=old.copy();at=np.zeros(len(items))
for angle in angles:
 d.qpos[:]=q0;d.qpos[m.jnt_qposadr[shaft]]+=np.deg2rad(-2*angle)
 for j in joints:d.qpos[m.jnt_qposadr[j]]+=np.deg2rad(angle)
 mujoco.mj_forward(m,d)
 for i,(a,b,pieces,rad) in enumerate(items):
  value=float(mujoco.mj_geomDistance(m,d,a,b,.003,None));candidate=min(float(mujoco.mj_geomDistance(m,d,g,b,.003,None)) for g in pieces)
  old[i]=min(old[i],value)
  if candidate<new[i]:new[i]=candidate;at[i]=angle
step=np.deg2rad(angles[1]-angles[0]);rows=[]
for x,(_,_,_,rad),xold,xnew,angle in zip(source,items,old,new,at):
 bound=2*rad*math.sin(step/4);rows.append({'geoms':x['geoms'],'leg':x['leg'],'old_min_mm':float(xold*1000),'sector_min_mm':float(xnew*1000),'sector_min_at_deg':float(angle),'interpolation_motion_bound_mm':float(bound*1000),'bounded_lower_mm':float((xnew-bound)*1000)})
out={'sample_count':len(angles),'hip_range_deg':[-5,5],'pair_count':len(rows),'positive_lower_bound_count':sum(x['bounded_lower_mm']>0 for x in rows),'lower_bound_min_mm':min(x['bounded_lower_mm'] for x in rows),'results':rows,'limitations':'Only 57 selected proxy pairs, rigid ideal hip motion and 201 samples plus geometric interpolation bound. Does not include other proxies, controller overshoot, other contacts, floating manufacturing tolerances, or dynamics. Sector meshes remain inactive.'}
(p/'sector_gap_only_sweep_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('pairs',len(rows),'positive lower bounds',out['positive_lower_bound_count'],'minimum lower mm',out['lower_bound_min_mm'],flush=True)
