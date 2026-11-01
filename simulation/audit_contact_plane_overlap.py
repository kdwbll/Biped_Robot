"""Compute the sampled overlap of opposing CAD-identified STL contact faces."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
src=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in src.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
chassis=parts[0];chassis.apply_scale(.001)
def facets(mesh,body):
 world=mesh.vertices@d.xmat[body].reshape(3,3).T+d.xpos[body]
 v=(world-d.xpos[root])@d.xmat[root].reshape(3,3);tri=v[mesh.faces]
 cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 return tri[(np.max(np.abs(tri[:,:,2]-.136),axis=1)<.00001)&(np.abs(cross[:,2])>1e-14),:,:2]
def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def area(poly):
 if len(poly)<3:return 0.
 return abs(sum(cross2(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly))))/2
def intersect(subject,clip):
 poly=[x.copy() for x in subject]
 if cross2(clip[1]-clip[0],clip[2]-clip[0])<0:clip=clip[::-1]
 for i in range(3):
  if not poly:break
  a=clip[i];b=clip[(i+1)%3];edge=b-a;new=[]
  for j,current in enumerate(poly):
   prev=poly[j-1];dc=cross2(edge,current-a);dp=cross2(edge,prev-a)
   inc=dc>=-1e-14;inp=dp>=-1e-14
   if inc!=inp:
    denom=dp-dc
    if abs(denom)>1e-20:new.append(prev+(current-prev)*(dp/denom))
   if inc:new.append(current)
  poly=new
 return area(poly)
ch=facets(chassis,root);chlo=ch.min(axis=1);chhi=ch.max(axis=1)
rows=[]
for leg,filename in enumerate(('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml')):
 xml=ET.parse(p/filename).getroot();chunks=trimesh.load_mesh(p/xml.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in chunks if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part0');tris=facets(tower,body)
 total=0.;reverse_total=0.;positive=0
 for tri in tris:
  lo=tri.min(axis=0);hi=tri.max(axis=0)
  candidates=np.flatnonzero(np.all(chhi>=lo-1e-12,axis=1)&np.all(chlo<=hi+1e-12,axis=1))
  for idx in candidates:
   a=intersect(ch[idx],tri)
   if a>1e-18:
    total+=a;reverse_total+=intersect(tri,ch[idx]);positive+=1
 if abs(total-reverse_total)>1e-10:raise RuntimeError('Triangle clipping order check failed')
 rows.append({'leg':leg,'chassis_triangles':len(ch),'tower_triangles':len(tris),
              'positive_triangle_intersections':positive,'projected_overlap_area_mm2':total*1e6,
              'reverse_order_difference_mm2':abs(total-reverse_total)*1e6})
out={'pose':'initial keyframe','plane_z_assembly_m':.136,'results':rows,
 'limitations':'Overlap of STL triangles in projection on one plane at one pose, not proof of load-bearing contact, tolerances, normals under motion, or mechanical compliance.'}
(p/'contact_plane_overlap_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
