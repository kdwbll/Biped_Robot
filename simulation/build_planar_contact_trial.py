"""Build a CAD-plane triangular-prism contact experiment, not a production model.

The planar faces were identified using CAD closest distances and STL facets.
Prisms are extruded inward from that plane. Friction is a placeholder; motion
filtering assumes +/-5 degree rigid hip rotation. Existing contacts are retained.
"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p=Path(__file__).parent;src=p/'diagnostic_chassis0_collision_trial.xml'
tree=ET.parse(src);root=tree.getroot();assets=root.find('asset');contact=root.find('contact')
m=mujoco.MjModel.from_xml_path(str(src));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
robot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');rr=d.xmat[robot].reshape(3,3);rx=d.xpos[robot]
mesh=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in mesh.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
chassis=parts[0];chassis.apply_scale(.001)
outdir=p/'cad_plane_collision_meshes';outdir.mkdir(exist_ok=True)
prism_faces=np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
def facets(mesh,body,sign):
 rb=d.xmat[body].reshape(3,3);xb=d.xpos[body]
 world=mesh.vertices@rb.T+xb;local=(world-rx)@rr;tri=local[mesh.faces]
 normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 mask=(np.max(np.abs(tri[:,:,2]-.136),axis=1)<.00001)&(normal[:,2]*sign>1e-14)
 tri=tri[mask];tri[:,:,2]=.136
 return tri
def build(triangles,body_name,prefix,sign):
 body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body_name);rb=d.xmat[body_id].reshape(3,3);xb=d.xpos[body_id]
 body=next(x for x in root.iter('body') if x.get('name')==body_name);names=[]
 extrude_root=np.array([0.,0.,sign*.0001]);extrude_local=rb.T@rr@extrude_root
 for i,tri in enumerate(triangles):
  world=tri@rr.T+rx;local=(world-xb)@rb
  vertices=np.vstack((local,local+extrude_local))*1000
  part=trimesh.Trimesh(vertices=vertices,faces=prism_faces,process=True)
  name=f'plane_{prefix}_{i:04d}';path=outdir/(name+'.stl');part.export(path)
  ET.SubElement(assets,'mesh',name=name,file='cad_plane_collision_meshes/'+path.name,scale='.001 .001 .001')
  ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
  names.append(name)
 return names
chtri=facets(chassis,robot,-1);chnames=build(chtri,'robot_free','C',1)
chlo=chtri[:,:,:2].min(axis=1);chhi=chtri[:,:,:2].max(axis=1);counts=[];allpairs=0
for leg,filename in enumerate(('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml')):
 xml=ET.parse(p/filename).getroot();chunks=trimesh.load_mesh(p/xml.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in chunks if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part0');tri=facets(tower,body,1)
 names=build(tri,f'L{leg}_part0',f'L{leg}',-1)
 joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate')
 anchor=(d.xanchor[joint]-rx)@rr
 for i,t in enumerate(tri):
  radius=float(np.max(np.linalg.norm(t-anchor,axis=1)))
  margin=2*radius*math.sin(math.radians(5)/2)+.002
  lo=t[:,:2].min(axis=0)-margin;hi=t[:,:2].max(axis=0)+margin
  candidates=np.flatnonzero(np.all(chhi>=lo,axis=1)&np.all(chlo<=hi,axis=1))
  for idx in candidates:
   ET.SubElement(contact,'pair',geom1=chnames[idx],geom2=names[i],solref='.001 1',solimp='.99 .999 .001',friction='.0001 .0001 .0001 .0001 .0001')
   allpairs+=1
 counts.append({'leg':leg,'triangle_count':len(tri)})
root.set('model','diagnostic_cad_plane_contact_trial')
path=p/'diagnostic_cad_plane_contact_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
out={'chassis_triangles':len(chtri),'legs':counts,'added_plane_pairs':allpairs,'compiled_pairs':model.npair,'initial_contacts':data.ncon,
 'thickness_mm':.1,'assumed_friction':.0001,
 'limitations':'Experimental STL facet prisms, initial CAD contact plane and bounded hip range only. Friction, physical thickness and solver response are unverified; existing contacts may duplicate. Not for training or hardware.'}
(p/'cad_plane_contact_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
