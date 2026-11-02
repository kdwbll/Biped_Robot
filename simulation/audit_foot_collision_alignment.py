"""Compare raw foot mesh bounds with proposed box proxy at initial pose."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

p=Path(__file__).parent;filename='diagnostic_range_collision_trial.xml'
root=ET.parse(p/filename).getroot();m=mujoco.MjModel.from_xml_path(str(p/filename));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
rows=[]
for leg in (0,1):
 mesh=root.find(f"./asset/mesh[@name='L{leg}_mesh6']")
 source=trimesh.load_mesh(p/mesh.get('file'),process=False)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part6')
 proxy=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{leg}_foot_proxy')
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 vertices=source.vertices*.001@d.xmat[body].reshape(3,3).T+d.xpos[body]
 signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
 corners=(signs*m.geom_size[proxy])@d.geom_xmat[proxy].reshape(3,3).T+d.geom_xpos[proxy]
 vmin=vertices.min(axis=0);vmax=vertices.max(axis=0);pmin=corners.min(axis=0);pmax=corners.max(axis=0)
 bottom=vertices[vertices[:,2]<=vmin[2]+.0002,:2]
 if len(bottom)<3:raise RuntimeError('Too few vertices at the foot bottom')
 bottom_area=float(ConvexHull(bottom).volume)
 proxy_area=float(ConvexHull(corners[:,:2]).volume)
 rows.append({'leg':leg,'mesh_world_bounds_mm':(np.array([vmin,vmax])*1000).tolist(),
  'proxy_world_bounds_mm':(np.array([pmin,pmax])*1000).tolist(),
  'proxy_minus_mesh_bottom_mm':float((pmin[2]-vmin[2])*1000),
  'near_lowest_0_2mm_vertex_count':len(bottom),'near_lowest_mesh_hull_area_mm2':bottom_area*1e6,
  'box_projected_hull_area_mm2':proxy_area*1e6,
  'proxy_ground_signed_distance_mm':float(mujoco.mj_geomDistance(m,d,ground,proxy,1,None)*1000),
  'proxy_floor_pair_present':any(ground in (a,b) and proxy in (a,b) for a,b in zip(m.pair_geom1,m.pair_geom2))})
out={'pose':'initial keyframe','results':rows,
 'limitations':'World axis bounds from raw mesh and oriented box, not support polygon comparison or validation across foot motion. Mesh includes all component geometry.'}
(p/'foot_collision_alignment_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
