"""Compare CAD closest-distance points to the assembled simulation STL surfaces."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p=Path(__file__).parent
cad=json.loads((p/'solidworks_closest_distance_audit.json').read_text(encoding='utf8'))
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in source.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
chassis={i:parts[i] for i in (0,5)}
for mesh in chassis.values():mesh.apply_scale(.001)
towers=[]
for filename in ('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml'):
 root=ET.parse(p/filename).getroot();chunks=trimesh.load_mesh(p/root.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in chunks if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001);towers.append(tower)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');rows=[]
for row in cad['results']:
 leg=row['leg'];body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part0')
 # CAD assembly coordinates are compared under the assumed chassis-source-to-root mapping.
 points=[np.asarray(row['point1_m']),np.asarray(row['point2_m'])]
 world=[d.xmat[root].reshape(3,3)@q+d.xpos[root] for q in points]
 local_chassis=(world[0]-d.xpos[root])@d.xmat[root].reshape(3,3)
 local_tower=(world[1]-d.xpos[body])@d.xmat[body].reshape(3,3)
 distances=[]
 for mesh,point in ((chassis[row['chassis_source_index']],local_chassis),(towers[leg],local_tower)):
  _,dist,_=trimesh.proximity.closest_point_naive(mesh,point[None,:]);distances.append(float(dist[0])*1000)
 rows.append({'chassis_source_index':row['chassis_source_index'],'leg':leg,'cad_distance_mm':row['distance_mm'],
              'cad_witness_to_corresponding_stl_surface_mm':distances,'simulation_world_witness_m':[x.tolist() for x in world]})
out={'pose':'keyframe 0 vs saved CAD pose','results':rows,'limitations':'Assumes CAD assembly coordinates match chassis STL frame; STL tessellation and candidate leg phase produce discrepancies. A small point-to-mesh distance does not establish surface normals or full contact patch.'}
(p/'cad_contact_mapping_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
