"""Compare high-force convex proxies to their original, non-convex STL components.

Uses FCL triangle-mesh intersection at the same MuJoCo pose; not an exact CAD
interference test. The triangle meshes come from the same sources used to build
the existing decompositions. No contacts are changed.
"""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
src=next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL'))
source=trimesh.load_mesh(src,process=True)
chassis=[]
for c in source.split(only_watertight=False,repair=False):
 if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001:
  c.apply_scale(.001);chassis.append(c)
towers=[]
for leg,filename in enumerate(('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml')):
 root=ET.parse(p/filename).getroot();mesh=p/root.find("./asset/mesh[@name='mesh0']").get('file')
 parts=trimesh.load_mesh(mesh,process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in parts if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces))
 tower.apply_scale(.001);towers.append(tower)
contacts=json.loads((p/'standing_contact_forces_audit.json').read_text(encoding='utf8'))
suspects=sorted({tuple(sorted(r['geoms'])) for r in contacts if 'ground' not in r['geoms'] and r['normal_force_n']>100})
m=mujoco.MjModel.from_xml_path(str(p/'biped_standing_gear_candidate.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
def pose(body):
 t=np.eye(4);t[:3,:3]=d.xmat[body].reshape(3,3);t[:3,3]=d.xpos[body];return t
def measure():
 result=[]
 for a,b in suspects:
  cgeom=a if a.startswith('dc_chassis_') else b
  tgeom=b if a.startswith('dc_chassis_') else a
  idx=int(cgeom.split('_')[2]);leg=int(tgeom.split('_')[1][1:])
  ci=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,cgeom)
  ti=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,tgeom)
  manager=trimesh.collision.CollisionManager()
  manager.add_object('chassis_source',chassis[idx],transform=pose(m.geom_bodyid[ci]))
  collision,details=manager.in_collision_single(towers[leg],transform=pose(m.geom_bodyid[ti]),return_data=True)
  same=[r for r in contacts if tuple(sorted(r['geoms']))==(a,b)]
  contact=max(same,key=lambda x:x['normal_force_n'])
  world=np.array(contact['contact_point_m'])
  cb=m.geom_bodyid[ci];tb=m.geom_bodyid[ti]
  local_c=(world-d.xpos[cb])@d.xmat[cb].reshape(3,3)
  local_t=(world-d.xpos[tb])@d.xmat[tb].reshape(3,3)
  _,dc,_=trimesh.proximity.closest_point_naive(chassis[idx],local_c[None,:])
  _,dt,_=trimesh.proximity.closest_point_naive(towers[leg],local_t[None,:])
  result.append({'proxy_pair':[a,b],'source_chassis_piece':idx,'tower_leg':leg,
                 'source_mesh_triangles_intersect':bool(collision),
                 'reported_contact_count':len(details),
                 'fcl_max_reported_depth_m':max([float(x.depth) for x in details]+[0.]),
                 'distance_from_logged_6s_proxy_contact_to_chassis_mesh_m':float(dc[0]),
                 'distance_from_logged_6s_proxy_contact_to_tower_mesh_m':float(dt[0])})
 return result
out={'warning':'STL triangle collision at simulated pose, not exact CAD; contact and FCL penetration depth are approximate.',
     'source_chassis_piece_count':len(chassis),'suspect_pair_count':len(suspects),'poses':{}}
out['poses']['initial']=measure()
for _ in range(round(6/m.opt.timestep)):servo.step()
mujoco.mj_forward(m,d)
out['poses']['six_seconds']=measure()
(p/'proxy_vs_original_mesh_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
for pose_name, rows in out['poses'].items():
 print(pose_name)
 for r in rows:print(r['proxy_pair'],'original triangles intersect',r['source_mesh_triangles_intersect'],
                     'FCL depth mm',round(r['fcl_max_reported_depth_m']*1000,4),
                     'distance from logged contact to source meshes mm',
                     round(r['distance_from_logged_6s_proxy_contact_to_chassis_mesh_m']*1000,3),
                     round(r['distance_from_logged_6s_proxy_contact_to_tower_mesh_m']*1000,3))
