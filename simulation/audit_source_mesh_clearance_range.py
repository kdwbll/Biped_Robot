"""Estimate source STL surface distances during sampled hip rotation with FCL."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);q0=d.qpos.copy()
ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate')]
adr=[int(m.jnt_qposadr[x]) for x in ids]
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in source.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
chassis=[parts[0],parts[5]]
for x in chassis:x.apply_scale(.001)
towers=[]
for filename in ('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml'):
 root=ET.parse(p/filename).getroot()
 chunks=trimesh.load_mesh(p/root.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in chunks if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001);towers.append(tower)
root_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
legs=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{i}_part0') for i in (0,1)]
def pose(body):
 matrix=np.eye(4);matrix[:3,:3]=d.xmat[body].reshape(3,3);matrix[:3,3]=d.xpos[body];return matrix
rows=[]
for angle in np.linspace(-5,5,21):
 d.qpos[:]=q0;d.qpos[adr[0]]+=np.deg2rad(-2*angle);d.qpos[adr[1]]+=np.deg2rad(angle);d.qpos[adr[2]]+=np.deg2rad(angle);mujoco.mj_forward(m,d)
 for ci,mesh in zip((0,5),chassis):
  cm=trimesh.collision.CollisionManager();cm.add_object('chassis',mesh,transform=pose(root_body))
  for leg,tower in enumerate(towers):
   tr=pose(legs[leg]);collision=cm.in_collision_single(tower,transform=tr)
   distance=float(cm.min_distance_single(tower,transform=tr))*1000
   rows.append({'hip_delta_deg':float(angle),'chassis_source_index':ci,'leg':leg,'triangle_surface_intersection':bool(collision),'surface_distance_mm':distance})
 print('hip deg',angle,flush=True)
summary=[]
for ci in (0,5):
 for leg in (0,1):
  subset=[r for r in rows if r['chassis_source_index']==ci and r['leg']==leg]
  summary.append({'chassis_source_index':ci,'leg':leg,'min_surface_distance_mm':min(r['surface_distance_mm'] for r in subset),
    'max_surface_distance_mm':max(r['surface_distance_mm'] for r in subset),'intersecting_sample_count':sum(r['triangle_surface_intersection'] for r in subset)})
out={'sample_count':21,'hip_range_deg':[-5,5],'summary':summary,'samples':rows,
 'limitations':'FCL triangle surface distance from STL, not exact SolidWorks solid distance; mesh tessellation and containment are not ruled out. Only sampled hip poses.'}
(p/'source_mesh_clearance_range_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
