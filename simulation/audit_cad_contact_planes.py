"""Inspect STL facets on CAD-identified chassis/tower contact planes."""
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
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
source=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in source.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
torso=parts[0];torso.apply_scale(.001)
def to_root(vertices,body):
 world=vertices@d.xmat[body].reshape(3,3).T+d.xpos[body]
 return (world-d.xpos[root])@d.xmat[root].reshape(3,3)
def facets(mesh,body,z):
 v=to_root(mesh.vertices,body);tri=v[mesh.faces]
 mask=np.max(np.abs(tri[:,:,2]-z),axis=1)<.00001
 selected=tri[mask];cross=np.cross(selected[:,1]-selected[:,0],selected[:,2]-selected[:,0]);area=np.linalg.norm(cross,axis=1)/2
 return {'triangle_count':int(mask.sum()),'facet_area_mm2':float(area.sum()*1e6),
         'positive_normal_count':int(np.sum(cross[:,2]>0)),'negative_normal_count':int(np.sum(cross[:,2]<0))}
rows=[]
for leg,filename in enumerate(('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml')):
 r=next(x for x in cad['results'] if x['chassis_source_index']==0 and x['leg']==leg)
 z=float(r['point1_m'][2]);rootxml=ET.parse(p/filename).getroot()
 chunks=trimesh.load_mesh(p/rootxml.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in chunks if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part0')
 rows.append({'leg':leg,'cad_plane_z_m':z,'chassis_facets':facets(torso,root,z),'tower_facets':facets(tower,body,z)})
out={'results':rows,'plane_tolerance_mm':.01,'limitations':'Only coplanar STL facets at one initial pose; total facet area is not the overlap/contact patch and can include unrelated coplanar faces.'}
(p/'cad_contact_planes_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
