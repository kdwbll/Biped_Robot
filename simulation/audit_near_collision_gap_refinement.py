"""Refine excluded-pair gaps and bound interpolation in the assumed hip range."""
from pathlib import Path
import json,math
import mujoco
import numpy as np
p=Path(__file__).parent
model=mujoco.MjModel.from_xml_path(str(p/'diagnostic_separate_sole_contact_trial.xml'));d=mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,d,0);mujoco.mj_forward(model,d);initial=d.qpos.copy()
raw=json.loads((p/'chassis0_finer_proxy_range_audit.json').read_text(encoding='utf8'))
selected=[x for x in raw['near_pairs'] if .05<x['min_mm']<=.1]
if len(selected)!=30:raise RuntimeError('Unexpected candidate selection')
joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
shaft=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
ids=[];radii=[]
for x in selected:
 a,b=(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in x['geoms'])
 if min(a,b)<0:raise RuntimeError('Missing geometry')
 leg=0 if x['geoms'][1].startswith('nomerge_L0_') else 1
 joint=joints[leg];mount=model.jnt_bodyid[joint];body=model.geom_bodyid[b]
 while body!=mount:
  if body<=0 or model.body_jntnum[body]:raise RuntimeError('Proxy is not rigid on hip mount')
  body=model.body_parentid[body]
 rootbody=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot_free');body=model.geom_bodyid[a]
 while body!=rootbody:
  if body<=0 or model.body_jntnum[body]:raise RuntimeError('Chassis proxy moves relative to root')
  body=model.body_parentid[body]
 v=d.geom_xpos[b]-d.xanchor[joint];axis=d.xaxis[joint];r=float(np.linalg.norm(v-axis*np.dot(v,axis))+model.geom_rbound[b])
 ids.append((a,b));radii.append(r)
angles=np.linspace(-5.,5.,2001);mins=np.full(len(ids),np.inf);at=np.zeros(len(ids))
for angle in angles:
 d.qpos[:]=initial;d.qpos[model.jnt_qposadr[shaft]]+=np.deg2rad(-2*angle)
 for j in joints:d.qpos[model.jnt_qposadr[j]]+=np.deg2rad(angle)
 mujoco.mj_forward(model,d)
 dist=np.array([mujoco.mj_geomDistance(model,d,a,b,.003,None) for a,b in ids]);change=dist<mins;mins=np.minimum(mins,dist);at[change]=angle
step_rad=np.deg2rad(angles[1]-angles[0]);rows=[]
for x,r,minimum,angle in zip(selected,radii,mins,at):
 interpolation_bound_m=2*r*math.sin(step_rad/4)
 rows.append({'geoms':x['geoms'],'radius_bound_m':r,'sample_min_mm':minimum*1000,'min_at_deg':angle,'interpolation_motion_bound_mm':interpolation_bound_m*1000,'certified_proxy_gap_lower_bound_mm':(minimum-interpolation_bound_m)*1000})
out={'model':'diagnostic_separate_sole_contact_trial.xml','hip_range_deg':[-5,5],'sample_count':len(angles),'results':rows,'limitations':'Conservative rigid-proxy interpolation bound for ideal kinematics only; numerical tolerances, motion outside assumed +/-5 degrees, non-rigid model errors, CAD manufacturing and unmodelled collisions are not covered.'}
(p/'near_collision_gap_refinement_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('pairs',len(rows),'positive lower bound',sum(x['certified_proxy_gap_lower_bound_mm']>0 for x in rows),'minimum lower bound mm',min(x['certified_proxy_gap_lower_bound_mm'] for x in rows),flush=True)
